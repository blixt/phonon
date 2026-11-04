// The CPU backend (graph.ts): every buffer is a range of one WebAssembly memory, and every op a
// descriptor of u32 words in it that the module (cpu/) reads, written once when the graph is
// built. A submit runs its ops in one call into the module. With shared memory, pool threads in
// other workers each take part of every op (cpu/src/pool.rs). The module has no allocator: this
// file lays out the memory, from the module's heap base up.
import type { Tensor } from "./checkpoint";
import type * as G from "./graph";
import { S_MAX } from "./lm";

export interface CpuBuf {
  ptr: number;
  size: number;
}
export type CpuOp = number;

export interface CpuExports {
  memory: WebAssembly.Memory;
  run(list: number, n: number): void;
  setup(threads: number, scratch: number, per: number): void;
  rope_table(freqs: number, out: number, n: number): void;
  worker(i: number): void;
  __heap_base: WebAssembly.Global;
  __stack_pointer: WebAssembly.Global;
}

/** Op kinds and their fields, in the order the module reads them (cpu/src/ops.rs). */
const KIND = { copy: 0, clear: 1, gemv: 2, gemm: 3, layernorm: 4, rope: 5, attention: 6, attnStep: 7, alignRow: 8, embed: 9, noise: 10, finish: 11, upsample: 12, shift: 13, elu: 14 } as const;
const FIELDS = {
  gemv: ["x", "xo", "w", "ws", "y", "K", "N", "act", "bias", "res", "r", "g", "go", "v", "ln", "lnw", "lnb", "eps", "mod", "so", "ho", "splits", "parts", "state"],
  gemm: ["x", "w", "ws", "y", "M", "K", "N", "act", "bias", "res", "r", "yo", "ro", "load", "cin", "stride", "gate", "state"],
  layernorm: ["x", "y", "rows", "w", "b", "eps", "gate", "state"],
  rope: ["qkv", "q", "kc", "vc", "rows", "pos0", "freqs", "codec", "ring", "state", "table", "tableRows"],
  attention: ["q", "kc", "vc", "out", "rows", "pos0", "codec", "window", "ring", "state"],
  attnStep: ["qkv", "kc", "vc", "parts", "splits", "freqs", "align", "al", "lo", "width", "head", "state", "table", "tableRows"],
  alignRow: ["q", "kc", "al", "lo", "head", "state"],
  embed: ["ids", "x", "rows", "bos", "pad", "w", "ws", "bosv"],
  noise: ["z", "state"],
  finish: ["x", "lat", "lats", "state", "ew", "eb", "nw", "nb", "eps", "threshold"],
  upsample: ["lats", "y", "T", "w", "ws", "mean", "std", "state"],
  elu: ["src", "dst", "count", "gate", "state"],
} as const;
type Kind = keyof typeof FIELDS;
type Values<K extends Kind> = Partial<Record<(typeof FIELDS)[K][number], number>>;

const ACT: Record<G.Act, number> = { none: 0, gelu: 1, silu: 2 };
const GATE: Record<G.Gate, number> = { none: 0, step: 1, chunk: 2 };
const LOAD = { linear: 0, conv: 1, convT: 2 } as const;
const PAGE = 65536;
/** Each thread's scratch: a GEMM's packed weights (K x 8 floats) and attention scores. */
const SCRATCH_PER = 160 * 1024;
/** Each pool thread's stack. */
export const STACK = 256 * 1024;

const f32bits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0];

class Staging implements G.Staging {
  readonly bytes: ArrayBuffer;
  constructor(bytes: number) {
    this.bytes = new ArrayBuffer(bytes);
  }
  map() {
    // A task, not a microtask, so the worker hears a cancel between submits.
    return new Promise<void>((done) => {
      const c = new MessageChannel();
      c.port1.onmessage = () => {
        c.port1.close();
        done();
      };
      c.port2.postMessage(null);
    });
  }
  range() {
    return this.bytes;
  }
  unmap() {}
}

type Command = { op: CpuOp } | { read: () => void };

class Encoder implements G.Encoder<CpuBuf, CpuOp> {
  private readonly g: CpuGraph;
  private readonly commands: Command[] = [];
  constructor(g: CpuGraph) {
    this.g = g;
  }
  run(ops: CpuOp[]) {
    for (const op of ops) this.commands.push({ op });
  }
  copy(src: CpuBuf, so: number, dst: CpuBuf, dO: number, bytes: number) {
    this.commands.push({ op: this.g.temp(KIND.copy, [src.ptr + so, dst.ptr + dO, bytes]) });
  }
  clear(buf: CpuBuf, offset: number, bytes: number) {
    this.commands.push({ op: this.g.temp(KIND.clear, [buf.ptr + offset, bytes]) });
  }
  read(src: CpuBuf, so: number, dst: G.Staging, dO: number, bytes: number) {
    this.commands.push({
      read: () => new Uint8Array((dst as Staging).bytes, dO, bytes).set(this.g.bytes(src.ptr + so, bytes)),
    });
  }
  submit() {
    const ops: CpuOp[] = [];
    for (const c of this.commands) {
      if ("op" in c) ops.push(c.op);
      else {
        this.g.exec(ops.splice(0));
        c.read();
      }
    }
    this.g.exec(ops);
    this.g.endSubmit();
  }
}

export class CpuGraph implements G.Graph<CpuBuf, CpuOp> {
  readonly weights: CpuBuf;
  readonly state: CpuBuf;
  readonly depth = 1;
  readonly kvBytes = 4;
  readonly threads: number;
  private readonly x: CpuExports;
  private readonly tensors: Record<string, Tensor>;
  private top: number;
  private list: CpuBuf;
  /** Descriptors of one submit's copies and clears, reused by the next. */
  private readonly temps: CpuBuf;
  private tempAt = 0;
  private readonly freqs = new Map<string, number>();
  private table = 0;
  private tableTheta = "";
  /** ELU'd inputs of convolutions: one buffer for all, and the descriptor fields that name it. */
  private elu: CpuBuf | null = null;
  private readonly eluRefs: number[] = [];

  /** `x` is the instantiated module, `stacks` the bytes the pool threads' stacks took from the
   *  heap base, and `blob` the weights, copied into the memory. */
  constructor(x: CpuExports, model: { blob: Uint8Array; tensors: Record<string, Tensor> }, o: { threads: number; heap: number }) {
    this.x = x;
    this.tensors = model.tensors;
    this.threads = o.threads;
    this.top = o.heap;
    this.weights = this.buffer(model.blob.byteLength);
    this.bytes(this.weights.ptr, model.blob.byteLength).set(model.blob);
    this.state = this.buffer(32);
    this.list = this.buffer(4 * 1024);
    this.temps = this.buffer(4 * 4096);
    const scratch = this.buffer(SCRATCH_PER * o.threads);
    x.setup(o.threads, scratch.ptr, SCRATCH_PER);
  }

  /** Bytes from the heap, 64-byte aligned, zeroed. The memory grows as needed. */
  buffer(size: number): CpuBuf {
    const ptr = Math.ceil(this.top / 64) * 64;
    this.top = ptr + Math.max(size, 4);
    const mem = this.x.memory;
    if (this.top > mem.buffer.byteLength) mem.grow(Math.ceil((this.top - mem.buffer.byteLength) / PAGE));
    return { ptr, size };
  }
  /** The memory's bytes (a fresh view: the memory can grow). */
  bytes(ptr: number, n: number) {
    return new Uint8Array(this.x.memory.buffer, ptr, n);
  }
  write(buf: CpuBuf, offset: number, data: ArrayBufferView<ArrayBuffer>) {
    this.bytes(buf.ptr + offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  }
  encoder() {
    return new Encoder(this);
  }
  staging(bytes: number) {
    return new Staging(bytes);
  }
  async ready() {}

  /** Runs ops now, in one call. */
  exec(ops: CpuOp[]) {
    if (!ops.length) return;
    if (ops.length * 4 > this.list.size) this.list = this.buffer(ops.length * 8);
    new Uint32Array(this.x.memory.buffer, this.list.ptr, ops.length).set(ops);
    this.x.run(this.list.ptr, ops.length);
  }
  endSubmit() {
    this.tempAt = 0;
  }
  /** A descriptor for this submit only. */
  temp(kind: number, values: number[]): CpuOp {
    const words = 1 + values.length;
    if ((this.tempAt + words) * 4 > this.temps.size) throw new Error("Too many copies in one submit");
    const ptr = this.temps.ptr + this.tempAt * 4;
    new Uint32Array(this.x.memory.buffer, ptr, words).set([kind, ...values]);
    this.tempAt += words;
    return ptr;
  }

  private desc<K extends Kind>(kind: K, values: Values<K>, extra: number[] = []): CpuOp {
    const fields = FIELDS[kind] as readonly string[];
    const given = values as Record<string, number | undefined>;
    const words = [KIND[kind], ...fields.map((f) => given[f] ?? 0), ...extra];
    const b = this.buffer(words.length * 4);
    new Uint32Array(this.x.memory.buffer, b.ptr, words.length).set(words.map((v) => v >>> 0));
    return b.ptr;
  }
  private tensor(name: string): Tensor {
    const t = this.tensors[name];
    if (!t) throw new Error(`No tensor ${name}`);
    return t;
  }
  /** A tensor's address, and its scales' (0 for f16). */
  private t(name: string) {
    const t = this.tensor(name);
    return { w: this.weights.ptr + t.offset, ws: t.scales === undefined ? 0 : this.weights.ptr + t.scales };
  }
  private half(name?: string) {
    if (!name) return 0;
    const t = this.tensor(name);
    if (t.scales !== undefined) throw new Error(`${name} is int8`);
    return this.weights.ptr + t.offset;
  }
  private q8(name: string) {
    const t = this.t(name);
    if (!t.ws) throw new Error(`${name} is not int8`);
    return t;
  }
  /** The 32 rotary frequencies of `theta` in f32, and a table of their cos and sin for the
   *  LM's positions. */
  private rope(theta: string) {
    let f = this.freqs.get(theta);
    if (f === undefined) {
      const b = this.buffer(32 * 4);
      new Float32Array(this.x.memory.buffer, b.ptr, 32).set(
        Array.from({ length: 32 }, (_, d) => Math.fround(Number(theta) ** (-d / 32))),
      );
      f = b.ptr;
      this.freqs.set(theta, f);
    }
    if (!this.table) {
      const t = this.buffer(S_MAX * 64 * 4);
      this.x.rope_table(f, t.ptr, S_MAX);
      this.table = t.ptr;
      this.tableTheta = theta;
    }
    const table = this.tableTheta === theta ? this.table : 0;
    return { freqs: f, table, tableRows: table ? S_MAX : 0 };
  }

  gemv(x: CpuBuf | null, wname: string, y: CpuBuf, o: G.Gemv<CpuBuf>): CpuOp {
    const { ln, combine } = o;
    const res = o.res ?? "none";
    const w = this.q8(wname);
    return this.desc("gemv", {
      x: x?.ptr,
      xo: o.xo,
      ...w,
      y: y.ptr,
      K: o.K,
      N: o.N,
      act: ACT[o.act ?? "none"],
      bias: this.half(o.bias),
      res: { none: 0, add: 1, add_y: 1, gate: 2, gate_y: 2, vec: 3 }[res],
      r: o.rbuf?.ptr,
      g: o.gbuf?.ptr,
      go: o.go,
      v: this.half(o.vname),
      ln: ln ? 1 | (ln.w ? 2 : 0) | (ln.mod ? 4 : 0) : 0,
      lnw: this.half(ln?.w),
      lnb: this.half(ln?.b),
      eps: ln ? f32bits(Number(ln.eps)) : 0,
      mod: ln?.mod?.ptr,
      so: ln?.so,
      ho: ln?.ho,
      splits: combine?.splits,
      parts: combine?.parts.ptr,
      state: this.state.ptr,
    });
  }

  gemm(x: CpuBuf, wname: string, y: CpuBuf, o: G.Gemm<CpuBuf>): CpuOp[] {
    const load = o.load ?? "linear";
    const conv = o.conv;
    const cin = conv?.cin ?? 0;
    const gate = GATE[o.gate];
    const ops: CpuOp[] = [];
    let src = x.ptr;
    if (conv?.elu && load !== "linear") {
      // A convolution reads its M + k - 1 input rows (a transposed one M + 1), ELU'd first.
      const rows = load === "conv" ? o.M + (conv.k ?? 1) - 1 : o.M + 1;
      const count = rows * cin;
      if (!this.elu || this.elu.size < count * 4) {
        this.elu = this.buffer(count * 4);
        const words = new Uint32Array(this.x.memory.buffer);
        for (const at of this.eluRefs) words[at / 4] = this.elu.ptr;
      }
      const op = this.desc("elu", { src: x.ptr, dst: this.elu.ptr, count, gate, state: this.state.ptr });
      this.eluRefs.push(op + 8);
      ops.push(op);
      src = this.elu.ptr;
    }
    const op = this.desc("gemm", {
      x: src,
      ...this.t(wname),
      y: y.ptr,
      M: o.M,
      K: o.K,
      N: o.N,
      act: ACT[o.act ?? "none"],
      bias: this.half(o.bias),
      res: o.res && o.res !== "none" ? 1 : 0,
      r: o.rbuf?.ptr ?? y.ptr,
      yo: o.yo,
      ro: o.ro,
      load: LOAD[load],
      cin,
      stride: conv?.stride ?? 1,
      gate,
      state: this.state.ptr,
    });
    if (src !== x.ptr) this.eluRefs.push(op + 4);
    ops.push(op);
    return ops;
  }

  layernorm(x: CpuBuf, y: CpuBuf, rows: number, w: string, b: string, eps: string, gate: G.Gate): CpuOp {
    return this.desc("layernorm", {
      x: x.ptr,
      y: y.ptr,
      rows,
      w: this.half(w),
      b: this.half(b),
      eps: f32bits(Number(eps)),
      gate: GATE[gate],
      state: this.state.ptr,
    });
  }

  ropeKV(qkv: CpuBuf, q: CpuBuf, kc: CpuBuf, vc: CpuBuf, at: G.Rows, o: { theta: string; cache: G.Cache }): CpuOp {
    const codec = o.cache.kind === "codec";
    return this.desc("rope", {
      qkv: qkv.ptr,
      q: q.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      rows: at.rows,
      pos0: at.pos0,
      ...this.rope(o.theta),
      codec: codec ? 1 : 0,
      ring: o.cache.kind === "codec" ? o.cache.ring : 0,
      state: this.state.ptr,
    });
  }

  attention(q: CpuBuf, kc: CpuBuf, vc: CpuBuf, out: CpuBuf, at: G.Rows, o: { cache: G.Cache }): CpuOp {
    const c = o.cache;
    return this.desc("attention", {
      q: q.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      out: out.ptr,
      rows: at.rows,
      pos0: at.pos0,
      codec: c.kind === "codec" ? 1 : 0,
      window: c.kind === "codec" ? c.window : 0,
      ring: c.kind === "codec" ? c.ring : 0,
      state: this.state.ptr,
    });
  }

  attnStep(
    qkv: CpuBuf,
    kc: CpuBuf,
    vc: CpuBuf,
    parts: CpuBuf,
    o: { splits: number; theta: string; align?: { at: G.Align; buf: CpuBuf } },
  ): CpuOp {
    return this.desc("attnStep", {
      qkv: qkv.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      parts: parts.ptr,
      splits: o.splits,
      ...this.rope(o.theta),
      align: o.align ? 1 : 0,
      al: o.align?.buf.ptr,
      lo: o.align?.at.lo,
      width: o.align?.at.width,
      head: o.align?.at.head,
      state: this.state.ptr,
    });
  }

  alignRow(q: CpuBuf, kc: CpuBuf, al: CpuBuf, at: G.Align): CpuOp {
    return this.desc("alignRow", { q: q.ptr, kc: kc.ptr, al: al.ptr, lo: at.lo, head: at.head, state: this.state.ptr });
  }

  embedRows(ids: CpuBuf, x: CpuBuf, rows: number, o: { bos: number; pad: number }): CpuOp {
    return this.desc("embed", { ids: ids.ptr, x: x.ptr, rows, bos: o.bos, pad: o.pad, ...this.q8("embed"), bosv: this.half("bos_embed") });
  }

  noise(z: CpuBuf): CpuOp {
    return this.desc("noise", { z: z.ptr, state: this.state.ptr });
  }

  finishStep(x: CpuBuf, lat: CpuBuf, lats: CpuBuf, o: { eps: string; threshold: string }): CpuOp {
    return this.desc("finish", {
      x: x.ptr,
      lat: lat.ptr,
      lats: lats.ptr,
      state: this.state.ptr,
      ew: this.half("eos.w"),
      eb: this.half("eos.b"),
      nw: this.half("norm.w"),
      nb: this.half("norm.b"),
      eps: f32bits(Number(o.eps)),
      threshold: f32bits(Number(o.threshold)),
    });
  }

  upsample(lats: CpuBuf, y: CpuBuf, T: number): CpuOp {
    return this.desc("upsample", {
      lats: lats.ptr,
      y: y.ptr,
      T,
      ...this.q8("up"),
      mean: this.half("lat_mean"),
      std: this.half("lat_std"),
      state: this.state.ptr,
    });
  }

  shiftRows(stages: G.Stage<CpuBuf>[], T: number): CpuOp {
    const words = [KIND.shift, T, stages.length, ...stages.flatMap((s) => [s.buf.ptr, s.ctx, s.rate, s.ch])];
    const b = this.buffer(words.length * 4);
    new Uint32Array(this.x.memory.buffer, b.ptr, words.length).set(words);
    return b.ptr;
  }
}
