// GPU plumbing for the Phonon engine: ops (a pipeline, its bindings and a
// dispatch) over one weight blob. Everything an engine runs is built once,
// when it starts, so no request allocates. Pipelines compile in the
// background (`ready`), each with a layout read from its shader's own
// declarations, so bind groups exist before the compile ends.
import * as K from "./kernels";
import type { Tensor } from "./checkpoint";

// WebGPU's flag values: TypeScript's DOM types lack the constants' globals.
const STORAGE = 0x80;
const UNIFORM = 0x40;
const COPY_DST = 0x08;
const COPY_SRC = 0x04;
const COMPUTE = 0x4;
/** Split-K partials of the largest GEMM (the decoder's last transposed convolution). */
const SCRATCH = 8 << 20;

interface Pipe {
  layout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline | null;
}
export interface Op {
  pipe: Pipe;
  bindGroup: GPUBindGroup;
  groups: [number, number, number];
}
export type Binding = GPUBuffer | GPUBufferBinding;

export function encode(enc: GPUCommandEncoder, ops: Op[]) {
  const pass = enc.beginComputePass();
  for (const o of ops) {
    pass.setPipeline(o.pipe.pipeline!);
    pass.setBindGroup(0, o.bindGroup);
    pass.dispatchWorkgroups(...o.groups);
  }
  pass.end();
}

/** A shader's bind group layout, from its `@binding(i) var<storage|uniform, ...>` lines. */
function layoutOf(device: GPUDevice, code: string): GPUBindGroupLayout {
  const entries = [
    ...code.matchAll(/@binding\((\d+)\) var<(storage|uniform)(?:, (read_write|read))?>/g),
  ].map(([, b, kind, access]): GPUBindGroupLayoutEntry => ({
    binding: Number(b),
    visibility: COMPUTE,
    buffer: {
      type:
        kind === "uniform" ? "uniform" : access === "read_write" ? "storage" : "read-only-storage",
    },
  }));
  return device.createBindGroupLayout({ entries });
}

/** A residual into the output itself reads the output (one buffer cannot be bound twice). */
function residual(res: K.Residual, rbuf: GPUBuffer | undefined, ybuf: GPUBuffer) {
  return rbuf === ybuf ? { res: `${res}_y` as K.Residual, rbuf: undefined } : { res, rbuf };
}

export class Graph {
  readonly device: GPUDevice;
  readonly weights: GPUBuffer;
  /** The LM's and codec's state, slots as kernels.ts STATE names them. */
  readonly state: GPUBuffer;
  /** Bound where a kernel's binding goes unused. */
  readonly zero: GPUBuffer;
  /** 32-lane subgroups for the GEMV sums. */
  readonly sg: boolean;
  private readonly tensors: Record<string, Tensor>;
  private readonly pipes = new Map<string, Pipe>();
  private readonly compiling: Promise<void>[] = [];
  private scratchBuf: GPUBuffer | null = null;

  constructor(
    device: GPUDevice,
    model: { blob: Uint8Array<ArrayBuffer>; tensors: Record<string, Tensor> },
    sg: boolean,
  ) {
    this.device = device;
    this.tensors = model.tensors;
    this.sg = sg;
    this.weights = this.buffer(model.blob.byteLength, model.blob);
    this.state = this.buffer(32);
    this.zero = this.buffer(256);
  }

  buffer(size: number, data?: ArrayBufferView<ArrayBuffer>) {
    const b = this.device.createBuffer({
      size: Math.max(256, Math.ceil(size / 256) * 256),
      usage: STORAGE | COPY_DST | COPY_SRC,
    });
    if (data) this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }
  /** A uniform of a kernel's `fields` (kernels.ts UNIFORMS), 0 where `values` has none. */
  u<F extends K.Fields>(fields: F, values: Partial<Record<F[number], number>>) {
    const given: Partial<Record<string, number>> = values;
    const words = new Uint32Array(Math.max(4, Math.ceil(fields.length / 4) * 4));
    fields.forEach((f, i) => (words[i] = given[f] ?? 0));
    const b = this.device.createBuffer({ size: words.byteLength, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(b, 0, words);
    return b;
  }
  private tensor(name: string): Tensor {
    const t = this.tensors[name];
    if (!t) throw new Error(`No tensor ${name}`);
    return t;
  }
  /** An f16 tensor's offset in halves. */
  half(name: string) {
    const t = this.tensor(name);
    if (t.scales !== undefined) throw new Error(`${name} is int8`);
    return t.offset / 2;
  }
  /** A tensor's offset in u32 words, and its scales' offset in halves (int8 only). */
  words(name: string) {
    const t = this.tensor(name);
    return {
      wo: t.offset / 4,
      wso: t.scales === undefined ? 0 : t.scales / 2,
      q8: t.scales !== undefined,
    };
  }
  /** An int8 tensor's offsets, for the kernels that only read int8. */
  q8(name: string) {
    const w = this.words(name);
    if (!w.q8) throw new Error(`${name} is not int8`);
    return w;
  }
  /** Settles when every pipeline built so far has compiled. */
  async ready() {
    while (this.compiling.length) await Promise.all(this.compiling.splice(0));
  }
  op(code: string, bindings: Binding[], groups: [number, number, number]): Op {
    let pipe = this.pipes.get(code);
    if (!pipe) {
      const layout = layoutOf(this.device, code);
      const p: Pipe = { layout, pipeline: null };
      this.compiling.push(
        this.device
          .createComputePipelineAsync({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
          })
          .then((pipeline) => {
            p.pipeline = pipeline;
          }),
      );
      this.pipes.set(code, p);
      pipe = p;
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipe.layout,
      entries: bindings.map((b, i) => ({
        binding: i,
        resource: b instanceof GPUBuffer ? { buffer: b } : b,
      })),
    });
    return { pipe, bindGroup, groups };
  }
  /** Split-K partials. Dispatches run in order, so every GEMM's partials share one buffer. */
  private scratch(bytes: number): GPUBufferBinding {
    if (bytes > SCRATCH) throw new Error(`Split-K partials of ${bytes} bytes`);
    this.scratchBuf ??= this.buffer(SCRATCH);
    return { buffer: this.scratchBuf, offset: 0, size: Math.ceil(bytes / 256) * 256 };
  }

  /** y = epilogue(W x) for one row (x at element `xo`), optionally normalizing x first.
   *  A step's work, so it returns at once past the end. */
  gemv(
    x: GPUBuffer,
    wname: string,
    ybuf: GPUBuffer,
    o: {
      K: number;
      N: number;
      xo?: number;
      act?: K.Act;
      bias?: string;
      res?: K.Residual;
      rbuf?: GPUBuffer;
      /** gate: y += g[go + n] * acc, g in `gbuf` */
      gbuf?: GPUBuffer;
      go?: number;
      /** vec: y = acc + the blob's tensor `vname` */
      vname?: string;
      ln?: { w?: string; b?: string; eps: string; mod?: GPUBuffer; so?: number; ho?: number };
      /** Merge this many attention splits from `parts` as the input. */
      combine?: { splits: number; parts: GPUBuffer };
    },
  ): Op {
    const { ln, combine } = o;
    const { res, rbuf } = residual(o.res ?? "none", o.rbuf, ybuf);
    const w = this.q8(wname);
    const code = K.gemv({
      K: o.K,
      N: o.N,
      act: o.act,
      bias: !!o.bias,
      res,
      combine: combine?.splits,
      ln: ln && { affine: !!ln.w, modulate: !!ln.mod, eps: ln.eps },
      sg: this.sg,
    });
    const u = this.u(K.UNIFORMS.gemv, {
      xo: o.xo,
      go: o.go,
      so: ln?.so,
      ho: ln?.ho,
      wo: w.wo,
      bo: o.bias ? this.half(o.bias) : 0,
      vo: o.vname ? this.half(o.vname) : 0,
      lgo: ln?.w ? this.half(ln.w) : 0,
      lbo: ln?.b ? this.half(ln.b) : 0,
      wso: w.wso,
    });
    return this.op(
      code,
      [
        x,
        this.weights,
        ybuf,
        combine?.parts ?? rbuf ?? this.zero,
        u,
        o.gbuf ?? this.zero,
        ln?.mod ?? this.zero,
        this.state,
      ],
      [Math.ceil(o.N / K.gemvShape(o.K).rowsPerGroup), 1, 1],
    );
  }

  /** Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. */
  gemm(
    xbuf: GPUBuffer,
    wname: string,
    ybuf: GPUBuffer,
    o: {
      M: number;
      K: number;
      N: number;
      gate: K.Gate;
      act?: K.Act;
      bias?: string;
      res?: K.Residual;
      rbuf?: GPUBuffer;
      /** Output and residual offsets, in elements. */
      yo?: number;
      ro?: number;
      load?: "linear" | "conv" | "convT";
      conv?: K.ConvShape;
    },
  ): Op[] {
    const { M, K: k, N: n, load = "linear", conv, yo = 0, ro = 0, gate } = o;
    const { res, rbuf } = residual(o.res ?? "none", o.rbuf, ybuf);
    const w = this.words(wname);
    const phases = load === "convT" ? conv!.stride! : 1;
    // Row tiles are capped at 4, so long inputs share the split counts of shorter ones.
    const tiles = Math.ceil(n / 64) * Math.min(4, Math.ceil(M / 64)) * phases;
    const splits = Math.max(1, Math.min(Math.floor(160 / tiles), Math.floor(k / 128)));
    const groups: [number, number, number] = [
      Math.ceil(n / 64),
      Math.ceil(M / 64),
      phases * splits,
    ];
    const bo = o.bias ? this.half(o.bias) : 0;
    const shape = { K: k, N: n, load, conv, q8: w.q8, gate };
    const P = K.UNIFORMS.gemm;
    if (splits === 1) {
      const code = K.gemm({ ...shape, act: o.act, bias: !!o.bias, res });
      return [
        this.op(
          code,
          [
            xbuf,
            this.weights,
            ybuf,
            rbuf ?? this.zero,
            this.u(P, { M, yo, ro, wo: w.wo, bo, wso: w.wso }),
            this.state,
          ],
          groups,
        ),
      ];
    }
    const rowsOut = M * phases;
    const part = this.scratch(splits * rowsOut * n * 4);
    return [
      this.op(
        K.gemm({ ...shape, splits }),
        [
          xbuf,
          this.weights,
          part,
          this.zero,
          this.u(P, { M, rows: rowsOut, wo: w.wo, wso: w.wso }),
          this.state,
        ],
        groups,
      ),
      this.op(
        K.gemmReduce({ N: n, splits, act: o.act, bias: !!o.bias, res, gate }),
        [
          part,
          ybuf,
          this.weights,
          rbuf ?? this.zero,
          this.u(K.UNIFORMS.gemmReduce, { rows: rowsOut, yo, ro, stride: rowsOut, bo }),
          this.state,
        ],
        [Math.ceil((rowsOut * n) / 256), 1, 1],
      ),
    ];
  }

  layernorm(
    x: GPUBuffer,
    y: GPUBuffer,
    rows: number,
    w: string,
    b: string,
    eps: string,
    gate: K.Gate,
  ): Op {
    return this.op(
      K.layernorm({ eps, gate }),
      [
        x,
        y,
        this.weights,
        this.u(K.UNIFORMS.layernorm, { go: this.half(w), bo: this.half(b) }),
        this.state,
      ],
      [rows, 1, 1],
    );
  }
}
