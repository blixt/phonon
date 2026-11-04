// Phonon in WebGPU. The LM (lm.ts) makes one 80 ms latent per step with
// no round trip to the page, so many steps go out in one submit, and the
// codec (codec.ts) turns latents into audio in chunks as soon as they exist:
// small chunks first, for the first sound, then 8 frames at a time. The GPU
// sees the end of speech itself, so work a submit runs past it costs little.
// Results come back behind the GPU, a few submits deep. A voice is the
// prompt rows the LM reads first: a catalog voice's speaker latents.
import { paceVectors } from "./pace";
import { lmLayers, toF16, type PhononModel } from "./checkpoint";
import { Graph, encode } from "./graph";
import { STATE } from "./kernels";
import { F_MAX, H, LanguageModel, PROMPT, S_MAX, TEXT_MAX } from "./lm";
import { CHUNKS, Codec, FRAME } from "./codec";

export { TEXT_MAX };
/** Submits in flight. */
const DEPTH = 3;
const TEMPERATURE = 0.3;
/** The reference's end of a text: 1 frame after the end-of-speech frame, or 3 for texts of at
 *  most 4 words, and at most (tokens / 3 + 2) seconds of frames. */
const FRAMES_AFTER_EOS = 1;
const FRAMES_AFTER_EOS_SHORT = 3;
const SHORT_TEXT_WORDS = 4;
const FRAMES_PER_SECOND = 12.5;
const COPY_DST = 0x08;
const MAP_READ = 0x01;
/** Each submit's readback: the state, then its chunk's audio. */
const AUDIO_AT = 16;

export class PhononEngine {
  private readonly g: Graph;
  private readonly model: PhononModel;
  private readonly lm: LanguageModel;
  private readonly codec: Codec;
  /** Each voice's prompt keys and values (lm.ts). */
  private readonly voices = new Map<string, GPUBuffer>();
  private readonly stages: GPUBuffer[];
  private pace = Number.NaN;

  constructor(
    device: GPUDevice,
    model: PhononModel,
    { subgroups, align }: { subgroups: boolean; align: { layer: number; head: number } },
  ) {
    this.model = model;
    this.g = new Graph(device, model, subgroups);
    this.lm = new LanguageModel(this.g, lmLayers(model), align);
    this.codec = new Codec(this.g, this.lm.lats);
    const most = Math.max(...CHUNKS);
    this.stages = Array.from({ length: DEPTH + 1 }, () =>
      device.createBuffer({ size: AUDIO_AT + most * FRAME * 4, usage: COPY_DST | MAP_READ }),
    );
    this.setPace(0);
  }

  /** Settles when every shader has compiled. */
  ready() {
    return this.g.ready();
  }

  /** Registers a voice by its speaker latents, [125][512]. */
  addVoice(name: string, latents: Float32Array) {
    if (latents.length !== PROMPT * H)
      throw new Error(`Voice ${name} has ${latents.length} values`);
    this.voices.set(name, this.lm.voice(latents));
  }
  hasVoice(name: string) {
    return this.voices.has(name);
  }

  /** The LM's pace (pace.ts): its condition and BOS row are written into the blob. */
  setPace(delta: number) {
    if (delta === this.pace) return;
    this.pace = delta;
    const { cond, bos } = paceVectors(this.model.pace, delta);
    for (const [name, values] of [
      ["cond", cond],
      ["bos_embed", bos],
    ] as const)
      this.g.device.queue.writeBuffer(
        this.g.weights,
        this.model.tensors[name].offset,
        toF16(values),
      );
  }

  /** Text ids to 24 kHz audio in one voice, in chunks as they are made. Null when cancelled. */
  async synthesize(
    ids: number[],
    voice: string,
    {
      seed = 1,
      words = Infinity,
      cancelled = () => false,
      onChunk,
    }: {
      seed?: number;
      /** The text's words: a short text keeps more frames after the end of speech. */
      words?: number;
      cancelled?: () => boolean;
      /** Each chunk of audio as it is made. */
      onChunk?: (samples: Float32Array) => void;
    } = {},
  ): Promise<Synthesis | null> {
    const kv = this.voices.get(voice);
    if (!kv) throw new Error(`Unknown voice ${voice}`);
    if (!ids.length) return { chunks: [], align: new Float32Array(0) };
    if (ids.length > TEXT_MAX) throw new Error(`Text too long: ${ids.length} tokens`);
    const { device } = this.g;
    const { lm, codec } = this;
    const maxFrames = Math.min(
      F_MAX,
      S_MAX - PROMPT - ids.length - 1,
      Math.ceil((ids.length / 3 + 2) * FRAMES_PER_SECOND),
    );
    const after = words <= SHORT_TEXT_WORDS ? FRAMES_AFTER_EOS_SHORT : FRAMES_AFTER_EOS;
    let enc = device.createCommandEncoder();
    codec.reset(enc);
    lm.start(enc, ids, kv, seed, TEMPERATURE, after);
    device.queue.submit([enc.finish()]);
    let issued = 1;
    let decoded = 0;
    let made = 0;
    let end: number | null = null;
    const chunks: Float32Array[] = [];
    const free = [...this.stages];
    const inflight: {
      stage: GPUBuffer;
      f0: number;
      size: number;
      n: number;
      ready: Promise<void>;
    }[] = [];
    // One submit: the steps the next chunk needs, then that chunk.
    const issue = () => {
      const stage = free.pop()!;
      enc = device.createCommandEncoder();
      const size = CHUNKS[Math.min(made, CHUNKS.length - 1)];
      const target = Math.min(maxFrames, end ?? Math.max(issued, decoded + size));
      const steps = [];
      for (; issued < target; issued++) steps.push(...lm.step);
      if (steps.length) encode(enc, steps);
      const avail = end ?? issued;
      let f0 = -1;
      if (decoded < avail && (decoded + size <= avail || end !== null)) {
        f0 = decoded;
        codec.encodeChunk(enc, f0, size);
        decoded += size;
        made++;
      }
      enc.copyBufferToBuffer(this.g.state, 0, stage, 0, 16);
      if (f0 >= 0) enc.copyBufferToBuffer(codec.audio, 0, stage, AUDIO_AT, size * FRAME * 4);
      device.queue.submit([enc.finish()]);
      inflight.push({ stage, f0, size, n: issued, ready: stage.mapAsync(MAP_READ) });
    };
    const drain = async () => {
      const job = inflight.shift()!;
      await job.ready;
      const m = job.stage.getMappedRange();
      // The frames to keep, once the GPU has seen the end of speech.
      const keep = new Uint32Array(m, 0, 4)[STATE.end];
      if (end === null && keep) end = Math.min(keep, maxFrames);
      if (end === null && job.n >= maxFrames) end = maxFrames;
      if (job.f0 >= 0 && (end === null || job.f0 < end)) {
        const frames = Math.min(job.size, (end ?? Infinity) - job.f0);
        const chunk = new Float32Array(m, AUDIO_AT, frames * FRAME).slice();
        chunks.push(chunk);
        onChunk?.(chunk);
      }
      job.stage.unmap();
      free.push(job.stage);
    };
    const finished = () => end !== null && decoded >= end && issued >= end;
    for (;;) {
      while (!cancelled() && inflight.length < DEPTH && !finished()) issue();
      if (!inflight.length) break;
      await drain();
    }
    if (cancelled()) return null;
    return { chunks, align: await this.alignment(end!, ids.length) };
  }

  /** The alignment head's scores for `frames` frames over `n` text tokens. */
  private async alignment(frames: number, n: number): Promise<Float32Array> {
    const { device } = this.g;
    const bytes = frames * TEXT_MAX * 4;
    const read = device.createBuffer({ size: bytes, usage: COPY_DST | MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.lm.align, 0, read, 0, bytes);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(MAP_READ);
    const rows = new Float32Array(read.getMappedRange());
    const out = new Float32Array(frames * n);
    for (let f = 0; f < frames; f++) out.set(rows.subarray(f * TEXT_MAX, f * TEXT_MAX + n), f * n);
    read.destroy();
    return out;
  }
}

/** A text's audio, in the chunks it was made in, and the alignment head's scores over its
 *  tokens: one row per 80 ms frame. */
export interface Synthesis {
  chunks: Float32Array[];
  align: Float32Array;
}
