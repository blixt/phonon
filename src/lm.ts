// Phonon's flow LM as ops: a transformer (6 layers or 9) over the voice prompt, the
// text and the audio latents, and the flow head that samples each next
// latent. A voice's 125 prompt rows go through the layers once (`voice`), and
// a text then copies their keys and values in. The prefill runs the text and
// BOS rows padded to a multiple of 64: one fixed op set per size, so a text
// only writes its ids. A step feeds the position in the state buffer and
// makes one frame. The KV cache is f16.
import * as K from "./kernels";
import { encode, type Graph, type Op } from "./graph";

export const H = 512;
const FF = 2048;
export const PROMPT = 125;
/** Positions the LM keeps: the voice prompt, the text and the frames. */
export const S_MAX = 1024;
/** Frames one synthesis may make (61 s). */
export const F_MAX = 768;
export const TEXT_MAX = 240;
/** Prefill sizes: the text and BOS rows, padded. */
const PREFILL = [64, 128, 192, 256];
/** Token ids past any real one: BOS, and padding (a zero row). */
const BOS = 0xfffffffe;
const PAD = 0xffffffff;
const LN_EPS = "1e-5";
const FLOW_LN_EPS = "1e-6";
const ROPE_THETA = "10000.0";
const EOS_THRESHOLD = "4.0";
const SPLITS = 8;
const KV_BYTES = 2;

/** Per-row activations of a transformer. */
export interface Rows {
  x: GPUBuffer;
  h: GPUBuffer;
  qkv: GPUBuffer;
  q: GPUBuffer;
  att: GPUBuffer;
  ff: GPUBuffer;
}
export function rowBuffers(g: Graph, rows: number): Rows {
  const r = (n: number) => g.buffer(rows * n * 4);
  return { x: r(H), h: r(H), qkv: r(3 * H), q: r(H), att: r(H), ff: r(FF) };
}

/** One transformer layer over `rows` rows. Row m sits at position pos0 + m, past the cache's
 *  own position (kernels.ts Cache). `out` takes the layer's result in place of `B.x`. */
export function layerRows(
  g: Graph,
  pre: string,
  o: {
    B: Rows;
    rows: number;
    kc: GPUBuffer;
    vc: GPUBuffer;
    cache: K.Cache;
    pos0: number;
    out?: { buf: GPUBuffer; yo: number };
  },
): Op[] {
  const { B, rows, cache } = o;
  const gate = cache.kind === "codec" ? "chunk" : "none";
  const at = g.u(K.UNIFORMS.rows, { rows, pos0: o.pos0 });
  return [
    g.layernorm(B.x, B.h, rows, `${pre}.ln1.w`, `${pre}.ln1.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.qkv`, B.qkv, { M: rows, K: H, N: 3 * H, gate }),
    g.op(
      K.ropeKV({ theta: ROPE_THETA, cache }),
      [B.qkv, B.q, o.kc, o.vc, g.state, at],
      [rows, 1, 1],
    ),
    g.op(
      K.attention({ cache }),
      [B.q, o.kc, o.vc, B.att, g.state, at],
      [Math.ceil(rows / 16), 8, 1],
    ),
    ...g.gemm(B.att, `${pre}.o`, B.x, { M: rows, K: H, N: H, res: "add", rbuf: B.x, gate }),
    g.layernorm(B.x, B.h, rows, `${pre}.ln2.w`, `${pre}.ln2.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.fc1`, B.ff, { M: rows, K: H, N: FF, act: "gelu", gate }),
    ...g.gemm(B.ff, `${pre}.fc2`, o.out?.buf ?? B.x, {
      M: rows,
      K: FF,
      N: H,
      res: "add",
      rbuf: B.x,
      yo: o.out?.yo ?? 0,
      gate,
    }),
  ];
}

export class LanguageModel {
  /** The latents made so far, one row of 32 per frame. */
  readonly lats: GPUBuffer;
  /** One step: a latent, then the next position. */
  readonly step: Op[];
  /** The alignment head's scores over the text, one row of TEXT_MAX per frame. */
  readonly align: GPUBuffer;
  private readonly g: Graph;
  private readonly rows: Rows;
  private readonly kc: GPUBuffer[];
  private readonly vc: GPUBuffer[];
  private readonly ids: GPUBuffer;
  /** The noise's uniform (kernels.ts UNIFORMS.noise: seed, then scale), written per text. */
  private readonly noise: GPUBuffer;
  /** The BOS row's output on its way to row 0, where the step's tail reads. */
  private readonly bos: GPUBuffer;
  private readonly voiceIn: GPUBuffer;
  private readonly voiceOps: Op[];
  private readonly prefills: Map<number, Op[]>;
  private readonly tail: Op[];
  private readonly cond: GPUBuffer;
  private readonly ada: GPUBuffer;
  private readonly fh: GPUBuffer;
  private readonly ft: GPUBuffer;
  private readonly z: GPUBuffer;
  private readonly lat: GPUBuffer;
  private readonly parts: GPUBuffer;
  private readonly depth: number;
  /** The head that follows the text as it speaks (files.ts), its scores kept for word times. */
  private readonly alignLayer: number;
  private readonly alignAt: K.Align;

  constructor(g: Graph, depth: number, align: { layer: number; head: number }) {
    this.g = g;
    this.depth = depth;
    this.alignLayer = align.layer;
    this.alignAt = { lo: PROMPT, width: TEXT_MAX, head: align.head };
    const rows = PREFILL.at(-1)!;
    this.rows = rowBuffers(g, rows);
    this.kc = Array.from({ length: depth }, () => g.buffer(S_MAX * H * KV_BYTES));
    this.vc = Array.from({ length: depth }, () => g.buffer(S_MAX * H * KV_BYTES));
    this.cond = g.buffer(H * 4);
    this.ada = g.buffer(10240 * 4);
    this.fh = g.buffer(H * 4);
    this.ft = g.buffer(H * 4);
    this.z = g.buffer(32 * 4);
    this.lat = g.buffer(32 * 4);
    this.lats = g.buffer(F_MAX * 32 * 4);
    this.ids = g.buffer(rows * 4);
    this.bos = g.buffer(H * 4);
    this.parts = g.buffer(SPLITS * 8 * 66 * 4);
    this.voiceIn = g.buffer(PROMPT * H * 4);
    this.align = g.buffer(F_MAX * TEXT_MAX * 4);
    this.noise = g.u(K.UNIFORMS.noise, {});
    this.tail = this.tailOps();
    this.step = [...Array.from({ length: depth }, (_, i) => this.stepLayer(i)).flat(), ...this.tail];
    this.voiceOps = [
      ...g.gemm(this.voiceIn, "prompt_proj", this.rows.x, { M: PROMPT, K: H, N: H, gate: "none" }),
      ...this.layers(PROMPT, 0),
    ];
    const em = g.q8("embed");
    this.prefills = new Map(
      PREFILL.map((n) => [
        n,
        [
          g.op(
            K.embedRows({ bos: BOS, pad: PAD }),
            [
              this.ids,
              g.weights,
              this.rows.x,
              g.u(K.UNIFORMS.embedRows, { eo: em.wo, eso: em.wso, bo: g.half("bos_embed") }),
            ],
            [(n * H) / 256, 1, 1],
          ),
          ...this.layers(n, PROMPT, true),
        ],
      ]),
    );
  }

  /** The layers over `rows` rows from position `pos0`, in the prefill's way, and for a text's
   *  prefill frame 0's alignment row. */
  private layers(rows: number, pos0: number, text = false): Op[] {
    const { g } = this;
    return Array.from({ length: this.depth }, (_, i) => [
      ...layerRows(g, `l${i}`, {
        B: this.rows,
        rows,
        kc: this.kc[i],
        vc: this.vc[i],
        cache: { kind: "lm" },
        pos0,
      }),
      ...(text && i === this.alignLayer
        ? [g.op(K.alignRow(this.alignAt), [this.rows.q, this.kc[i], this.align, g.state], [1, 1, 1])]
        : []),
    ]).flat();
  }

  /** A voice's prompt rows through every layer, its keys and values kept in a new buffer. */
  voice(latents: Float32Array): GPUBuffer {
    const { device } = this.g;
    const bytes = PROMPT * H * KV_BYTES;
    const kv = this.g.buffer(2 * this.depth * bytes);
    device.queue.writeBuffer(
      this.voiceIn,
      0,
      latents.buffer,
      latents.byteOffset,
      latents.byteLength,
    );
    const enc = device.createCommandEncoder();
    encode(enc, this.voiceOps);
    for (let i = 0; i < this.depth; i++) {
      enc.copyBufferToBuffer(this.kc[i], 0, kv, 2 * i * bytes, bytes);
      enc.copyBufferToBuffer(this.vc[i], 0, kv, (2 * i + 1) * bytes, bytes);
    }
    device.queue.submit([enc.finish()]);
    return kv;
  }

  /** Sets up a text in a voice (its keys and values from `voice`) and encodes its prefill, which
   *  makes frame 0. The state's position starts on BOS, and frame 0's step moves it past. */
  start(
    enc: GPUCommandEncoder,
    ids: number[],
    voice: GPUBuffer,
    seed: number,
    temperature: number,
    framesAfterEos: number,
  ) {
    const { device } = this.g;
    const n = PREFILL.find((p) => p > ids.length)!;
    const padded = new Uint32Array(n).fill(PAD);
    padded.set(ids);
    padded[ids.length] = BOS;
    device.queue.writeBuffer(this.ids, 0, padded);
    const noise = new DataView(new ArrayBuffer(16));
    noise.setUint32(0, seed >>> 0, true);
    noise.setFloat32(4, Math.sqrt(temperature), true);
    device.queue.writeBuffer(this.noise, 0, noise.buffer);
    const state = new Uint32Array(8);
    state[K.STATE.pos] = PROMPT + ids.length;
    state[K.STATE.after] = framesAfterEos;
    state[K.STATE.text] = ids.length;
    device.queue.writeBuffer(this.g.state, 0, state);
    const bytes = PROMPT * H * KV_BYTES;
    for (let i = 0; i < this.depth; i++) {
      enc.copyBufferToBuffer(voice, 2 * i * bytes, this.kc[i], 0, bytes);
      enc.copyBufferToBuffer(voice, (2 * i + 1) * bytes, this.vc[i], 0, bytes);
    }
    // The rows past BOS are padding: later rows never reach earlier ones, and each step writes
    // its own key and value before it reads the cache, so their keys and values are never read.
    encode(enc, this.prefills.get(n)!);
    enc.copyBufferToBuffer(this.rows.x, ids.length * H * 4, this.bos, 0, H * 4);
    enc.copyBufferToBuffer(this.bos, 0, this.rows.x, 0, H * 4);
    encode(enc, this.tail);
  }

  /** A step's layer: one row at the state's position, attention split over the keys. */
  private stepLayer(i: number): Op[] {
    const { g } = this;
    const B = this.rows;
    const pre = `l${i}`;
    return [
      g.gemv(B.x, `${pre}.qkv`, B.qkv, {
        K: H,
        N: 3 * H,
        ln: { w: `${pre}.ln1.w`, b: `${pre}.ln1.b`, eps: LN_EPS },
      }),
      i === this.alignLayer
        ? g.op(
            K.attnSplit({ splits: SPLITS, theta: ROPE_THETA, sg: g.sg, align: this.alignAt }),
            [B.qkv, this.kc[i], this.vc[i], this.parts, g.state, this.align],
            [SPLITS, 8, 1],
          )
        : g.op(
            K.attnSplit({ splits: SPLITS, theta: ROPE_THETA, sg: g.sg }),
            [B.qkv, this.kc[i], this.vc[i], this.parts, g.state],
            [SPLITS, 8, 1],
          ),
      g.gemv(g.zero, `${pre}.o`, B.x, {
        K: H,
        N: H,
        res: "add",
        rbuf: B.x,
        combine: { splits: SPLITS, parts: this.parts },
      }),
      g.gemv(B.x, `${pre}.fc1`, B.ff, {
        K: H,
        N: FF,
        act: "gelu",
        ln: { w: `${pre}.ln2.w`, b: `${pre}.ln2.b`, eps: LN_EPS },
      }),
      g.gemv(B.ff, `${pre}.fc2`, B.x, { K: FF, N: H, res: "add", rbuf: B.x }),
    ];
  }

  /** From the final hidden state (row 0 of x): the flow head samples the next latent, then its
   *  EOS logit, the next input row and the next position. */
  private tailOps(): Op[] {
    const { g } = this;
    const x = this.rows.x;
    const blocks = Array.from({ length: 6 }, (_, i) => {
      const a = i * 1536;
      return [
        g.gemv(this.fh, `fh${i}.fc1.w`, this.ft, {
          K: H,
          N: H,
          bias: `fh${i}.fc1.b`,
          act: "silu",
          ln: {
            w: `fh${i}.ln.w`,
            b: `fh${i}.ln.b`,
            eps: FLOW_LN_EPS,
            mod: this.ada,
            ho: a,
            so: a + 512,
          },
        }),
        g.gemv(this.ft, `fh${i}.fc2.w`, this.fh, {
          K: H,
          N: H,
          bias: `fh${i}.fc2.b`,
          res: "gate",
          rbuf: this.fh,
          gbuf: this.ada,
          go: a + 1024,
        }),
      ];
    }).flat();
    const fa = 6 * 1536;
    return [
      g.gemv(x, "fh.cond.w", this.cond, {
        K: H,
        N: H,
        bias: "fh.cond.b",
        act: "silu",
        ln: { w: "norm.w", b: "norm.b", eps: LN_EPS },
      }),
      g.gemv(this.cond, "fh.ada.w", this.ada, { K: H, N: 10240, bias: "fh.ada.b" }),
      g.op(K.noise(), [this.z, g.state, this.noise], [1, 1, 1]),
      g.gemv(this.z, "fh.in.w", this.fh, { K: 32, N: H, bias: "fh.in.b" }),
      ...blocks,
      g.gemv(this.fh, "fh.out.w", this.lat, {
        K: H,
        N: 32,
        bias: "fh.out.b",
        res: "add",
        rbuf: this.z,
        ln: { eps: FLOW_LN_EPS, mod: this.ada, ho: fa, so: fa + 512 },
      }),
      g.op(
        K.finishStep({ eps: LN_EPS, threshold: EOS_THRESHOLD }),
        [
          x,
          g.weights,
          this.lat,
          this.lats,
          g.state,
          g.u(K.UNIFORMS.finishStep, {
            ewo: g.half("eos.w"),
            ebo: g.half("eos.b"),
            ngo: g.half("norm.w"),
            nbo: g.half("norm.b"),
          }),
        ],
        [1, 1, 1],
      ),
      g.gemv(this.lat, "latent_proj", x, { K: 32, N: H, res: "vec", vname: "cond" }),
    ];
  }
}
