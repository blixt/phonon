// The backends' ops side by side: every op of graph.ts, with the model's real weights and fixed
// inputs, on WebGPU (gpu.ts) and on the CPU (cpu.ts), and how far apart their outputs are. A
// kernel that changes on one side only shows here. bench/ops.mjs runs it in headless Chrome.
import { unpackModel, toF16, type PhononModel } from "../src/checkpoint";
import { CpuGraph, instantiate, simdLevel, TILES, type Variant } from "../src/cpu";
import { GpuGraph } from "../src/gpu";
import { STATE, type Graph } from "../src/graph";

type Fill = (i: number) => number;
/** A fixed, varied input: no two cases share their values. */
const wave = (seed: number, scale = 1): Fill => (i) => scale * Math.sin(i * 0.731 + seed * 1.37) * Math.cos(i * 0.0137 + seed);

interface Case {
  name: string;
  /** Builds the case on a backend: its ops, the state before them, and the outputs to compare
   *  (f32 values, or the LM's cache in the backend's own format). */
  build<B, O>(g: Graph<B, O>, t: Tools<B>): { ops: O[]; state?: Partial<Record<keyof typeof STATE, number>>; out: Out<B>[] };
  /** Largest difference allowed, relative to the largest value of the output: 1e-4, or more
   *  where WebGPU keeps the LM's cache in f16 or computes sines of large angles. */
  tol?: number;
}
interface Out<B> {
  name: string;
  buf: B;
  count: number;
  /** In the LM cache's format: f16 on WebGPU, f32 on the CPU. */
  kv?: boolean;
  offset?: number;
}
interface Tools<B> {
  /** A buffer of `n` f32 values. */
  f32(n: number, fill?: Fill): B;
  /** A buffer of the LM cache's values, in the backend's format. */
  kv(n: number, fill: Fill): B;
  u32(values: number[]): B;
}

const H = 512;
const theta = "10000.0";
const cases: Case[] = [
  {
    name: "gemv qkv, LayerNorm",
    build: (g, t) => {
      const y = t.f32(1536);
      return { ops: [g.gemv(t.f32(H, wave(1)), "l0.qkv", y, { K: H, N: 1536, ln: { w: "l0.ln1.w", b: "l0.ln1.b", eps: "1e-5" } })], out: [{ name: "y", buf: y, count: 1536 }] };
    },
  },
  {
    name: "gemv fc1, LayerNorm, GELU",
    build: (g, t) => {
      const y = t.f32(2048);
      return { ops: [g.gemv(t.f32(H, wave(2, 3)), "l1.fc1", y, { K: H, N: 2048, act: "gelu", ln: { w: "l1.ln2.w", b: "l1.ln2.b", eps: "1e-5" } })], out: [{ name: "y", buf: y, count: 2048 }] };
    },
  },
  {
    name: "gemv fc2, residual in place",
    build: (g, t) => {
      const y = t.f32(H, wave(3));
      return { ops: [g.gemv(t.f32(2048, wave(4, 0.3)), "l2.fc2", y, { K: 2048, N: H, res: "add", rbuf: y })], out: [{ name: "y", buf: y, count: H }] };
    },
  },
  {
    name: "gemv o, merged attention splits",
    build: (g, t) => {
      const parts = t.f32(8 * 8 * 66, (i) => (i % 66 === 64 ? 2 + Math.sin(i) : i % 66 === 65 ? 3 + Math.cos(i) : Math.sin(i * 0.3)));
      const y = t.f32(H, wave(5));
      return { ops: [g.gemv(null, "l0.o", y, { K: H, N: H, res: "add", rbuf: y, combine: { splits: 8, parts } })], out: [{ name: "y", buf: y, count: H }] };
    },
  },
  {
    name: "gemv flow block: adaLN, SiLU, gate",
    build: (g, t) => {
      const ada = t.f32(10240, wave(6, 0.5));
      const fh = t.f32(H, wave(7));
      const ft = t.f32(H);
      return {
        ops: [
          g.gemv(fh, "fh2.fc1.w", ft, { K: H, N: H, bias: "fh2.fc1.b", act: "silu", ln: { w: "fh2.ln.w", b: "fh2.ln.b", eps: "1e-6", mod: ada, ho: 3072, so: 3584 } }),
          g.gemv(ft, "fh2.fc2.w", fh, { K: H, N: H, bias: "fh2.fc2.b", res: "gate", rbuf: fh, gbuf: ada, go: 4096 }),
        ],
        out: [{ name: "ft", buf: ft, count: H }, { name: "fh", buf: fh, count: H }],
      };
    },
  },
  {
    name: "gemv K = 32, and the latent projection",
    build: (g, t) => {
      const z = t.f32(32, wave(8));
      const fh = t.f32(H);
      const x = t.f32(H);
      return {
        ops: [g.gemv(z, "fh.in.w", fh, { K: 32, N: H, bias: "fh.in.b" }), g.gemv(z, "latent_proj", x, { K: 32, N: H, res: "vec", vname: "cond" })],
        out: [{ name: "fh", buf: fh, count: H }, { name: "x", buf: x, count: H }],
      };
    },
  },
  {
    name: "gemv flow output: modulation only",
    build: (g, t) => {
      const ada = t.f32(10240, wave(9, 0.5));
      const z = t.f32(32, wave(10));
      const lat = t.f32(32);
      return {
        ops: [g.gemv(t.f32(H, wave(11)), "fh.out.w", lat, { K: H, N: 32, bias: "fh.out.b", res: "add", rbuf: z, ln: { eps: "1e-6", mod: ada, ho: 9216, so: 9728 } })],
        out: [{ name: "lat", buf: lat, count: 32 }],
      };
    },
  },
  {
    name: "gemm linear, 37 rows",
    build: (g, t) => {
      const y = t.f32(37 * 1536);
      return { ops: g.gemm(t.f32(37 * H, wave(12)), "d0.qkv", y, { M: 37, K: H, N: 1536, gate: "none" }), out: [{ name: "y", buf: y, count: 37 * 1536 }] };
    },
  },
  {
    name: "gemm linear, GELU, then residual",
    build: (g, t) => {
      const h = t.f32(21 * 2048);
      const x = t.f32(21 * H, wave(14));
      return {
        ops: [
          ...g.gemm(t.f32(21 * H, wave(13)), "d1.fc1", h, { M: 21, K: H, N: 2048, act: "gelu", gate: "none" }),
          ...g.gemm(h, "d1.fc2", x, { M: 21, K: 2048, N: H, res: "add", rbuf: x, gate: "none" }),
        ],
        out: [{ name: "h", buf: h, count: 21 * 2048 }, { name: "x", buf: x, count: 21 * H }],
      };
    },
  },
  {
    name: "conv k = 7 into a stage's rows",
    build: (g, t) => {
      const y = t.f32(22 * H, wave(15));
      return {
        ops: g.gemm(t.f32(25 * H, wave(16)), "s0.w", y, { M: 19, K: 7 * H, N: H, bias: "s0.b", load: "conv", conv: { cin: H, k: 7, dil: 1, elu: false }, yo: 3 * H, gate: "chunk" }),
        out: [{ name: "y", buf: y, count: 22 * H }],
      };
    },
  },
  {
    name: "conv, ELU in, residual from a stage",
    build: (g, t) => {
      const res = t.f32(31 * 256, wave(17));
      const y = t.f32(30 * 256);
      return {
        ops: g.gemm(t.f32(30 * 128, wave(18, 2)), "s1r2.w", y, { M: 30, K: 128, N: 256, bias: "s1r2.b", load: "conv", conv: { cin: 128, k: 1, dil: 1, elu: true }, res: "add", rbuf: res, ro: 256, gate: "chunk" }),
        out: [{ name: "y", buf: y, count: 30 * 256 }],
      };
    },
  },
  {
    name: "transposed conv, stride 5, ELU in",
    build: (g, t) => {
      const y = t.f32((2 + 13 * 5) * 128, wave(19));
      return {
        ops: g.gemm(t.f32(14 * 256, wave(20, 2)), "s2.w", y, { M: 13, K: 512, N: 128, bias: "s2.b", load: "convT", conv: { cin: 256, elu: true, stride: 5 }, yo: 2 * 128, gate: "chunk" }),
        out: [{ name: "y", buf: y, count: (2 + 13 * 5) * 128 }],
      };
    },
  },
  {
    name: "f16 convs: k = 1 with residual, and one output channel",
    build: (g, t) => {
      const res = t.f32(42 * 64, wave(21));
      const y = t.f32(40 * 64);
      const audio = t.f32(38);
      return {
        ops: [
          ...g.gemm(t.f32(40 * 32, wave(22, 2)), "s3r2.w", y, { M: 40, K: 32, N: 64, bias: "s3r2.b", load: "conv", conv: { cin: 32, k: 1, dil: 1, elu: true }, res: "add", rbuf: res, ro: 128, gate: "chunk" }),
          ...g.gemm(y, "s4.w", audio, { M: 38, K: 192, N: 1, bias: "s4.b", load: "conv", conv: { cin: 64, k: 3, dil: 1, elu: true }, gate: "chunk" }),
        ],
        out: [{ name: "y", buf: y, count: 40 * 64 }, { name: "audio", buf: audio, count: 38 }],
      };
    },
  },
  {
    name: "LayerNorm rows",
    build: (g, t) => {
      const y = t.f32(5 * H);
      return { ops: [g.layernorm(t.f32(5 * H, wave(23, 4)), y, 5, "d0.ln1.w", "d0.ln1.b", "1e-5", "none")], out: [{ name: "y", buf: y, count: 5 * H }] };
    },
  },
  {
    name: "RoPE into the LM cache",
    tol: 2e-3,
    build: (g, t) => {
      const q = t.f32(7 * H);
      const kc = t.kv(140 * H, () => 0);
      const vc = t.kv(140 * H, () => 0);
      return {
        ops: [g.ropeKV(t.f32(7 * 1536, wave(24)), q, kc, vc, { rows: 7, pos0: 130 }, { theta, cache: { kind: "lm" } })],
        out: [{ name: "q", buf: q, count: 7 * H }, { name: "k", buf: kc, count: 7 * H, kv: true, offset: 130 * H }, { name: "v", buf: vc, count: 7 * H, kv: true, offset: 130 * H }],
      };
    },
  },
  {
    name: "RoPE into the codec ring",
    tol: 2e-3,
    build: (g, t) => {
      const q = t.f32(9 * H);
      const kc = t.f32(512 * H);
      const vc = t.f32(512 * H);
      return {
        ops: [g.ropeKV(t.f32(9 * 1536, wave(25)), q, kc, vc, { rows: 9, pos0: 0 }, { theta, cache: { kind: "codec", window: 251, ring: 512 } })],
        state: { chunk: 1504 },
        out: [{ name: "q", buf: q, count: 9 * H }, { name: "k", buf: kc, count: 512 * H }, { name: "v", buf: vc, count: 512 * H }],
      };
    },
  },
  {
    name: "attention of LM rows",
    tol: 2e-3,
    build: (g, t) => {
      const out = t.f32(20 * H);
      return {
        ops: [g.attention(t.f32(20 * H, wave(26)), t.kv(150 * H, wave(27)), t.kv(150 * H, wave(28)), out, { rows: 20, pos0: 125 }, { cache: { kind: "lm" } })],
        out: [{ name: "out", buf: out, count: 20 * H }],
      };
    },
  },
  {
    name: "windowed attention over the codec ring",
    build: (g, t) => {
      const out = t.f32(16 * H);
      return {
        ops: [g.attention(t.f32(16 * H, wave(29)), t.f32(512 * H, wave(30)), t.f32(512 * H, wave(31)), out, { rows: 16, pos0: 0 }, { cache: { kind: "codec", window: 251, ring: 512 } })],
        state: { chunk: 704 },
        out: [{ name: "out", buf: out, count: 16 * H }],
      };
    },
  },
  {
    name: "an LM step's split attention, with alignment",
    tol: 2e-3,
    build: (g, t) => {
      const kc = t.kv(400 * H, wave(32));
      const vc = t.kv(400 * H, wave(33));
      const parts = t.f32(8 * 8 * 66);
      const al = t.f32(8 * 240);
      return {
        ops: [g.attnStep(t.f32(1536, wave(34, 3)), kc, vc, parts, { splits: 8, theta, align: { at: { lo: 125, width: 240, head: 3 }, buf: al } })],
        state: { pos: 301, frame: 2, text: 50 },
        out: [
          { name: "parts", buf: parts, count: 8 * 8 * 66 },
          { name: "align", buf: al, count: 3 * 240 },
          { name: "k", buf: kc, count: H, kv: true, offset: 301 * H },
          { name: "v", buf: vc, count: H, kv: true, offset: 301 * H },
        ],
      };
    },
  },
  {
    name: "frame 0's alignment row",
    tol: 2e-3,
    build: (g, t) => {
      const al = t.f32(240);
      return { ops: [g.alignRow(t.f32(64 * H, wave(35)), t.kv(200 * H, wave(36)), al, { lo: 125, width: 240, head: 1 })], state: { pos: 125 + 40, text: 40 }, out: [{ name: "align", buf: al, count: 40 }] };
    },
  },
  {
    name: "embedding rows, BOS and padding",
    build: (g, t) => {
      const x = t.f32(64 * H, wave(37));
      const ids = Array.from({ length: 64 }, (_, i) => (i < 30 ? (i * 37) % 4000 : i === 30 ? 0xfffffffe : 0xffffffff));
      return { ops: [g.embedRows(t.u32(ids), x, 64, { bos: 0xfffffffe, pad: 0xffffffff })], out: [{ name: "x", buf: x, count: 64 * H }] };
    },
  },
  {
    name: "noise",
    build: (g, t) => {
      const z = t.f32(32);
      return { ops: [g.noise(z)], state: { frame: 17, seed: 12345, scale: Math.sqrt(0.3) }, out: [{ name: "z", buf: z, count: 32 }], tol: 1e-4 };
    },
  },
  {
    name: "the end of a step",
    build: (g, t) => {
      const lats = t.f32(8 * 32);
      return {
        ops: [g.finishStep(t.f32(H, wave(38)), t.f32(32, wave(39)), lats, { eps: "1e-5", threshold: "4.0" })],
        state: { pos: 300, frame: 5, after: 1 },
        out: [{ name: "lats", buf: lats, count: 8 * 32 }],
      };
    },
  },
  {
    name: "upsampled latents",
    build: (g, t) => {
      const y = t.f32(32 * H);
      return { ops: [g.upsample(t.f32(20 * 32, wave(40)), y, 32)], state: { chunk: 16 * 9 }, out: [{ name: "y", buf: y, count: 32 * H }] };
    },
  },
  {
    name: "stages' context to their fronts",
    build: (g, t) => {
      const a = t.f32((6 + 32) * H, wave(41));
      const b = t.f32((2 + 6 * 32) * 256, wave(42));
      return {
        ops: [g.shiftRows([{ ctx: 6, rate: 1, ch: H, buf: a }, { ctx: 2, rate: 6, ch: 256, buf: b }], 32)],
        out: [{ name: "a", buf: a, count: (6 + 32) * H }, { name: "b", buf: b, count: (2 + 6 * 32) * 256 }],
      };
    },
  },
];

function f16ToF32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15);
    out[i] = h & 0x8000 ? -v : v;
  }
  return out;
}

async function runCase<B, O>(g: Graph<B, O>, c: Case): Promise<Record<string, Float32Array>> {
  const tools: Tools<B> = {
    f32: (n, fill) => {
      const b = g.buffer(n * 4);
      if (fill) g.write(b, 0, Float32Array.from({ length: n }, (_, i) => fill(i)));
      return b;
    },
    kv: (n, fill) => {
      const b = g.buffer(n * g.kvBytes);
      const v = Float32Array.from({ length: n }, (_, i) => fill(i));
      g.write(b, 0, g.kvBytes === 2 ? toF16(v) : v);
      return b;
    },
    u32: (values) => {
      const b = g.buffer(values.length * 4);
      g.write(b, 0, Uint32Array.from(values));
      return b;
    },
  };
  const { ops, state, out } = c.build(g, tools);
  const st = new Uint32Array(8);
  for (const [k, v] of Object.entries(state ?? {})) {
    if (k === "scale") new Float32Array(st.buffer)[STATE.scale] = v;
    else st[STATE[k as keyof typeof STATE]] = v;
  }
  g.write(g.state, 0, st);
  await g.ready();
  const enc = g.encoder();
  enc.run(ops);
  const sizes = out.map((o) => o.count * (o.kv ? g.kvBytes : 4));
  const stage = g.staging(sizes.reduce((a, b) => a + b, 0) + 32);
  let at = 0;
  for (const [i, o] of out.entries()) {
    enc.read(o.buf, (o.offset ?? 0) * (o.kv ? g.kvBytes : 4), stage, at, sizes[i]);
    at += sizes[i];
  }
  enc.read(g.state, 0, stage, at, 32);
  enc.submit();
  await stage.map();
  const bytes = stage.range().slice(0);
  stage.unmap();
  const result: Record<string, Float32Array> = {};
  at = 0;
  for (const [i, o] of out.entries()) {
    result[o.name] = o.kv && g.kvBytes === 2 ? f16ToF32(new Uint16Array(bytes, at, o.count)) : new Float32Array(bytes, at, o.count);
    at += sizes[i];
  }
  result.state = Float32Array.from(new Uint32Array(bytes, at, 8));
  return result;
}

export async function compareBackends(packed: ArrayBuffer, wasm: Uint8Array<ArrayBuffer>) {
  const model: PhononModel = unpackModel(packed);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU");
  // The GEMVs' subgroup sums where the engine would use them (index.ts).
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number };
  const sg = adapter.features.has("subgroups") && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
  const device = await adapter.requestDevice({ requiredFeatures: sg ? ["subgroups"] : [] });
  const gpu = new GpuGraph(device, model, sg);
  const { exports, memory } = await instantiate(wasm, false, model.blob.byteLength + 64 * 2 ** 20);
  const cpu = new CpuGraph(exports, model, { threads: 1, heap: Math.ceil((exports.__heap_base.value as number) / 64) * 64 });
  const variants: Variant[] = [
    ...TILES.map((tile) => ({ tile, gemv: "f32" as const })),
    { tile: TILES[0], gemv: "int16" },
  ];
  const label = (v: Variant) => `${v.tile.dot ? "dot" : "broadcast"} ${v.tile.rows}x${v.tile.cols}, ${v.gemv} GEMV`;
  const rows = [];
  for (const c of cases) {
    const a = await runCase(gpu, c);
    for (const v of variants) {
      // A variant only changes the matrix products: the other ops run once.
      const products = /^(gemv|gemm|conv|transposed|f16)/.test(c.name);
      if (v !== variants[0] && !products) continue;
      if (v.gemv === "int16" && !c.name.startsWith("gemv")) continue;
      if (v.tile !== TILES[0] && v.gemv === "f32" && c.name.startsWith("gemv")) continue;
      cpu.tune(v);
      const b = await runCase(cpu, c);
      for (const name of Object.keys(a)) {
        const x = a[name];
        const y = b[name];
        let max = 0;
        let diff = 0;
        let bad = 0;
        for (let i = 0; i < x.length; i++) {
          max = Math.max(max, Math.abs(x[i]));
          const d = Math.abs(x[i] - y[i]);
          if (!(d <= diff)) diff = Number.isNaN(d) ? Infinity : Math.max(diff, d);
          if (Number.isNaN(x[i]) !== Number.isNaN(y[i])) bad++;
        }
        const rel = max ? diff / max : diff;
        // int16 inputs to the GEMV round each value to 1/32767 of its group's largest.
        const tol = name === "state" ? 0 : v.gemv === "int16" ? 1e-3 : (c.tol ?? 1e-4);
        rows.push({ case: c.name, variant: products ? label(v) : "", output: name, values: x.length, max, diff, rel, ok: rel <= tol && !bad });
      }
    }
  }
  device.destroy();
  return { simd: simdLevel(), subgroups: sg, memory: memory.buffer.byteLength, rows };
}
