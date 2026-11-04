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
/** Each submit's readback: the state, its chunk's audio, then the alignment rows of the frames
 *  it made (at most 8 steps and frame 0's row). */
const AUDIO_AT = 16;
const ROWS = 16;

export class PhononEngine {
  private readonly g: Graph;
  private readonly model: PhononModel;
  private readonly lm: LanguageModel;
  private readonly codec: Codec;
  /** Each voice's prompt keys and values (lm.ts). */
  private readonly voices = new Map<string, GPUBuffer>();
  private readonly stages: GPUBuffer[];
  private readonly rowsAt: number;
  private pace = Number.NaN;
  /** Settles when the text in progress has drained. */
  private turn: Promise<unknown> = Promise.resolve();

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
    this.rowsAt = AUDIO_AT + most * FRAME * 4;
    this.stages = Array.from({ length: DEPTH + 1 }, () =>
      device.createBuffer({ size: this.rowsAt + ROWS * TEXT_MAX * 4, usage: COPY_DST | MAP_READ }),
    );
    this.setPace(0);
  }

  /** Settles when every shader has compiled. */
  ready() {
    return this.g.ready();
  }

  /** Registers a voice by its speaker latents, [125][512]. Its prefill uses the LM's caches, so
   *  it runs in the queue, before a text (`synthesize`'s `latents`). */
  private addVoice(name: string, latents: Float32Array) {
    if (latents.length !== PROMPT * H)
      throw new Error(`Voice ${name} has ${latents.length} values`);
    this.voices.set(name, this.lm.voice(latents));
  }
  hasVoice(name: string) {
    return this.voices.has(name);
  }

  /** The LM's pace (pace.ts): its condition and BOS row are written into the blob, so it is set
   *  in the queue, before a text (`synthesize`'s `pace`). */
  private setPace(delta: number) {
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

  /** Text ids to 24 kHz audio in one voice, in chunks as they are made, with the alignment
   *  head's scores of each kept frame over the text as they are made. Null when cancelled. One
   *  text at a time: a call waits until the one before has drained, cancelled or not. */
  synthesize(...args: Parameters<PhononEngine["run"]>): ReturnType<PhononEngine["run"]> {
    const next = this.turn.then(() => this.run(...args));
    this.turn = next.catch(() => {});
    return next;
  }

  private async run(
    ids: number[],
    voice: string,
    {
      seed = 1,
      words = Infinity,
      pace = 0,
      latents,
      cancelled = () => false,
      onChunk,
      onRows,
    }: {
      seed?: number;
      /** The LM's pace (pace.ts). */
      pace?: number;
      /** The voice's speaker latents, [125][512], the first time it is used. */
      latents?: Float32Array;
      /** The text's words: a short text keeps more frames after the end of speech. */
      words?: number;
      cancelled?: () => boolean;
      /** Each chunk of audio as it is made. */
      onChunk?: (samples: Float32Array) => void;
      /** The alignment rows of the next kept frames, one score per text token each. */
      onRows?: (rows: Float32Array) => void;
    } = {},
  ): Promise<Synthesis | null> {
    if (!this.voices.has(voice)) {
      if (!latents) throw new Error(`Unknown voice ${voice}`);
      this.addVoice(voice, latents);
    }
    const kv = this.voices.get(voice)!;
    this.setPace(pace);
    if (!ids.length) return { chunks: [] };
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
    let rowsCopied = 0;
    let end: number | null = null;
    const chunks: Float32Array[] = [];
    const free = [...this.stages];
    const inflight: {
      stage: GPUBuffer;
      f0: number;
      size: number;
      n: number;
      /** The frames whose alignment rows it carries. */
      r0: number;
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
      const r0 = rowsCopied;
      if (issued > r0) {
        if (issued - r0 > ROWS) throw new Error(`${issued - r0} alignment rows in one submit`);
        const bytes = TEXT_MAX * 4;
        enc.copyBufferToBuffer(lm.align, r0 * bytes, stage, this.rowsAt, (issued - r0) * bytes);
        rowsCopied = issued;
      }
      device.queue.submit([enc.finish()]);
      inflight.push({ stage, f0, size, n: issued, r0, ready: stage.mapAsync(MAP_READ) });
    };
    const drain = async () => {
      const job = inflight.shift()!;
      await job.ready;
      const m = job.stage.getMappedRange();
      // The frames to keep, once the GPU has seen the end of speech.
      const keep = new Uint32Array(m, 0, 4)[STATE.end];
      if (end === null && keep) end = Math.min(keep, maxFrames);
      if (end === null && job.n >= maxFrames) end = maxFrames;
      // Rows come before audio: a frame's words can need the head's later frames, never its audio.
      const r1 = Math.min(job.n, end ?? Infinity);
      if (onRows && r1 > job.r0) {
        const n = ids.length;
        const rows = new Float32Array(m, this.rowsAt, (job.n - job.r0) * TEXT_MAX);
        const out = new Float32Array((r1 - job.r0) * n);
        for (let r = 0; r < r1 - job.r0; r++) out.set(rows.subarray(r * TEXT_MAX, r * TEXT_MAX + n), r * n);
        onRows(out);
      }
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
    return { chunks };
  }
}

/** A text's audio, in the chunks it was made in. */
export interface Synthesis {
  chunks: Float32Array[];
}
