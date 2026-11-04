// Phonon in the browser on WebGPU: load a model once, then speak text in any
// voice of Gradium's catalog in the model's languages, with the time of every
// word. The model's files (files.ts) stay in the browser's cache once downloaded.
import {
  readVoice,
  unpackModel,
  type PhononModel,
} from "./checkpoint";
import {
  CATALOG_URL,
  DEFAULT_VOICE,
  MODEL_CACHE,
  MODELS,
  tokenizerUrl,
  voiceUrl,
  type Language,
  type ModelFiles,
  type ModelId,
} from "./files";
import { PhononEngine, TEXT_MAX } from "./engine";
import { paceForSpeed } from "./pace";
import { SAMPLE_RATE, WordTimes, wordRanges, type Span, type Word } from "./timing";
import { Tokenizer } from "./tokenizer";

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(MODEL_CACHE);
  } catch {
    /* A page that keeps no cache downloads every time. */
    return null;
  }
}

/** Kept in the cache when it has room; a full cache still leaves the bytes in use. */
async function keep(cache: Cache | null, url: string, bytes: Uint8Array<ArrayBuffer>) {
  try {
    await cache?.put(url, new Response(bytes));
  } catch {
    /* Nothing to do. */
  }
}

async function download(url: string, progress?: (percent: number) => void) {
  const response = await fetch(url);
  if (!response.ok || !response.body)
    throw new Error(`The voice model could not download (${response.status}).`);
  const total = Number(response.headers.get("content-length")) || 0;
  const parts: Uint8Array[] = [];
  let loaded = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.byteLength;
    if (total && progress) progress(Math.min(100, Math.round((loaded / total) * 100)));
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) {
    bytes.set(p, o);
    o += p.byteLength;
  }
  return bytes;
}

/** A file from the cache, or downloaded and then kept. */
async function cached(url: string): Promise<ArrayBuffer> {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) return hit.arrayBuffer();
  const bytes = await download(url);
  await keep(cache, url, bytes);
  return bytes.buffer;
}

/** The model's weights, from the cache, or downloaded (reporting progress) and kept. */
async function weights(m: ModelFiles, progress: (percent: number) => void): Promise<PhononModel> {
  const cache = await openCache();
  const hit = await cache?.match(m.weights);
  if (hit) return unpackModel(await hit.arrayBuffer());
  const bytes = await download(m.weights, progress);
  await keep(cache, m.weights, bytes);
  return unpackModel(bytes.buffer);
}

/** Text cut where it must be so each part fits the model's text window:
 *  between sentences, then between words, then inside a word. */
function parts(text: string, count: (s: string) => number): string[] {
  if (count(text) <= TEXT_MAX) return [text];
  const out: string[] = [];
  let part = "";
  const add = (piece: string) => {
    if (part && count(part + piece) > TEXT_MAX) {
      out.push(part);
      part = "";
    }
    if (count(piece) <= TEXT_MAX) part += piece;
    else if (/\s/.test(piece.trim())) for (const word of piece.split(/(?<=\s)/)) add(word);
    else for (let i = 0; i < piece.length; i += 32) add(piece.slice(i, i + 32));
  };
  for (const sentence of text.match(/[^.!?]+(?:[.!?]+|$)\s*/g) ?? [text]) add(sentence);
  if (part) out.push(part);
  return out.map((p) => p.trim()).filter(Boolean);
}

/** A voice of Gradium's catalog. */
export interface Voice {
  name: string;
  id: string;
  language: Language;
  locale: string;
  gender: string;
  description: string;
}

/** Speech for a text: its 24 kHz samples and when each word, and each token in it, is heard. */
export interface Speech {
  audio: Float32Array;
  sampleRate: number;
  words: Word[];
  tokens: Span[];
}

export interface SpeakOptions {
  /** A catalog voice's name ("Delia") or id. The model's `defaultVoice` when not given. */
  voice?: string;
  /** Speaking speed, from about 0.77 to 1.3. */
  speed?: number;
  /** The sampling seed: the same seed and text give the same audio. */
  seed?: number;
  /** Each chunk of audio as soon as it is made: the first comes within milliseconds. */
  onAudio?: (samples: Float32Array) => void;
  /** Each word with its times, in order, as soon as nothing still to come can change them: once
   *  the token after it has its start, or the text has ended. They are the same as the result's
   *  `words`. With audio made much faster than real time, that is before the word is heard. A
   *  long word, or a slow device, can make a word arrive after its sound has begun. */
  onWord?: (word: Word) => void;
  signal?: AbortSignal;
}

/** The model, ready to speak. */
export class Phonon {
  private readonly engine: PhononEngine;
  private readonly tokenizer: Tokenizer;
  private readonly catalog: Voice[];
  /** Settles when the GPU device is lost (a reset): load again to go on. */
  readonly lost: Promise<unknown>;

  /** The voice `speak` uses when none is given: the default of the model's first language. */
  readonly defaultVoice: string;

  private readonly lag: number;

  private constructor(
    engine: PhononEngine,
    tokenizer: Tokenizer,
    catalog: Voice[],
    languages: Language[],
    lag: number,
    lost: Promise<unknown>,
  ) {
    this.engine = engine;
    this.tokenizer = tokenizer;
    this.catalog = catalog;
    this.defaultVoice = DEFAULT_VOICE[languages[0]];
    this.lag = lag;
    this.lost = lost;
  }

  /** Loads a model (files.ts MODELS, Phonon 31M by default): a download the first time, then
   *  the browser's cache. `progress` hears the download in percent. */
  static async load({
    model = "phonon-31m-en",
    progress,
  }: { model?: ModelId; progress?: (percent: number) => void } = {}): Promise<Phonon> {
    const files: ModelFiles = MODELS[model];
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
    if (!adapter) throw new Error("This browser has no WebGPU.");
    const subgroups = subgroups32(adapter);
    const device = await adapter.requestDevice({ requiredFeatures: subgroups ? ["subgroups"] : [] });
    const [blob, tokenizer, catalog] = await Promise.all([
      weights(files, progress ?? (() => {})),
      cached(tokenizerUrl(files)).then((b) => new Tokenizer(JSON.parse(new TextDecoder().decode(b)))),
      cached(CATALOG_URL).then((b) =>
        new TextDecoder()
          .decode(b)
          .split("\n")
          .filter(Boolean)
          .map((line) => JSON.parse(line) as Voice & { voice_id: string })
          .filter((v) => files.languages.includes(v.language))
          .map((v) => ({
            name: v.name,
            id: v.voice_id,
            language: v.language,
            locale: v.locale,
            gender: v.gender,
            description: v.description,
          })),
      ),
    ]);
    const engine = new PhononEngine(device, blob, { subgroups, align: files.align });
    await engine.ready();
    return new Phonon(engine, tokenizer, catalog, files.languages, files.align.lag, device.lost);
  }

  /** The catalog's voices in the model's languages. */
  get voices(): readonly Voice[] {
    return this.catalog;
  }

  /** The words `speak` gives for a text, in order, each with its tokens' text, as the tokenizer
   *  writes them: a page can lay them out before the times arrive. */
  words(text: string): { text: string; tokens: string[] }[] {
    return this.parts(text).flatMap((part) => {
      const pieces = this.tokenizer.encode(part).map((id) => this.tokenizer.pieces[id].replaceAll("▁", ""));
      return wordRanges(this.tokenizer.encode(part).map((id) => this.tokenizer.pieces[id])).map((w) => ({
        text: w.text,
        tokens: pieces.slice(w.from, w.to),
      }));
    });
  }

  private parts(text: string) {
    return parts(text, (s) => this.tokenizer.encode(s).length);
  }

  /** Text to speech, in parts that fit the model, joined. */
  async speak(text: string, options: SpeakOptions = {}): Promise<Speech> {
    const { voice = this.defaultVoice, speed = 1, seed = 1, onAudio, onWord, signal } = options;
    const v = this.catalog.find((c) => c.name === voice || c.id === voice);
    if (!v) throw new Error(`No voice ${voice}`);
    const latents = this.engine.hasVoice(v.id) ? undefined : readVoice(await cached(voiceUrl(v.id)));
    const audio: Float32Array[] = [];
    const words: Word[] = [];
    const tokens: Span[] = [];
    let at = 0;
    // Each part has its own seed: the noise depends on the seed and the frame, which starts at 0.
    for (const [i, part] of this.parts(text).entries()) {
      signal?.throwIfAborted();
      const ids = this.tokenizer.encode(part);
      const shift = <S extends Span>(s: S): S => ({ ...s, start: s.start + at, end: s.end + at });
      const shiftWord = (w: Word): Word => ({ ...shift(w), tokens: w.tokens.map(shift) });
      const times = new WordTimes(
        ids.map((id) => this.tokenizer.pieces[id]),
        this.lag,
        onWord && ((w) => onWord(shiftWord(w))),
      );
      const out = await this.engine.synthesize(ids, v.id, {
        seed: seed + i,
        words: part.split(/\s+/).filter(Boolean).length,
        pace: paceForSpeed(speed),
        latents,
        onRows: (rows) => times.addRows(rows),
        onChunk: (chunk) => {
          times.addAudio(chunk);
          onAudio?.(chunk);
        },
        cancelled: () => !!signal?.aborted,
      });
      signal?.throwIfAborted();
      const pcm = join(out!.chunks);
      const timed = times.finish();
      words.push(...timed.words.map(shiftWord));
      tokens.push(...timed.tokens.map(shift));
      audio.push(pcm);
      at += pcm.length / SAMPLE_RATE;
    }
    return { audio: join(audio), sampleRate: SAMPLE_RATE, words, tokens };
  }
}

function join(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of parts) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** 32-lane subgroups: the GEMVs sum with them where every subgroup has exactly 32 lanes. */
function subgroups32(adapter: GPUAdapter): boolean {
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number };
  return adapter.features.has("subgroups") && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
}

export { TEXT_MAX } from "./engine";
export { DEFAULT_VOICE, MODELS, type Language, type ModelId } from "./files";
export type { Span, Word } from "./timing";
