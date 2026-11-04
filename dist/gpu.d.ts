import * as K from "./kernels";
import type { Tensor } from "./checkpoint";
import type * as G from "./graph";
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
export declare function encode(enc: GPUCommandEncoder, ops: Op[]): void;
declare class Staging implements G.Staging {
    readonly buf: GPUBuffer;
    private ready;
    constructor(device: GPUDevice, bytes: number);
    map(): Promise<void>;
    range(): ArrayBuffer;
    unmap(): void;
}
declare class Encoder implements G.Encoder<GPUBuffer, Op> {
    private readonly device;
    private readonly enc;
    constructor(device: GPUDevice);
    run(ops: Op[]): void;
    copy(src: GPUBuffer, so: number, dst: GPUBuffer, dO: number, bytes: number): void;
    clear(buf: GPUBuffer, offset: number, bytes: number): void;
    read(src: GPUBuffer, so: number, dst: G.Staging, dO: number, bytes: number): void;
    submit(): void;
}
export declare class GpuGraph implements G.Graph<GPUBuffer, Op> {
    readonly device: GPUDevice;
    readonly weights: GPUBuffer;
    readonly state: GPUBuffer;
    readonly depth = 3;
    readonly kvBytes = 2;
    /** Bound where a kernel's binding goes unused. */
    readonly zero: GPUBuffer;
    /** 32-lane subgroups for the GEMV sums. */
    readonly sg: boolean;
    private readonly tensors;
    private readonly pipes;
    private readonly compiling;
    private scratchBuf;
    constructor(device: GPUDevice, model: {
        blob: Uint8Array<ArrayBuffer>;
        tensors: Record<string, Tensor>;
    }, sg: boolean);
    buffer(size: number, data?: ArrayBufferView<ArrayBuffer>): GPUBuffer;
    write(buf: GPUBuffer, offset: number, data: ArrayBufferView<ArrayBuffer>): void;
    encoder(): Encoder;
    staging(bytes: number): Staging;
    /** A uniform of a kernel's `fields` (kernels.ts UNIFORMS), 0 where `values` has none. */
    u<F extends K.Fields>(fields: F, values: Partial<Record<F[number], number>>): GPUBuffer;
    private tensor;
    /** An f16 tensor's offset in halves. */
    half(name: string): number;
    /** A tensor's offset in u32 words, and its scales' offset in halves (int8 only). */
    words(name: string): {
        wo: number;
        wso: number;
        q8: boolean;
    };
    /** An int8 tensor's offsets, for the kernels that only read int8. */
    q8(name: string): {
        wo: number;
        wso: number;
        q8: boolean;
    };
    /** Settles when every pipeline built so far has compiled. */
    ready(): Promise<void>;
    op(code: string, bindings: Binding[], groups: [number, number, number]): Op;
    /** Split-K partials. Dispatches run in order, so every GEMM's partials share one buffer. */
    private scratch;
    gemv(x: GPUBuffer | null, wname: string, ybuf: GPUBuffer, o: G.Gemv<GPUBuffer>): Op;
    gemm(xbuf: GPUBuffer, wname: string, ybuf: GPUBuffer, o: G.Gemm<GPUBuffer>): Op[];
    layernorm(x: GPUBuffer, y: GPUBuffer, rows: number, w: string, b: string, eps: string, gate: G.Gate): Op;
    ropeKV(qkv: GPUBuffer, q: GPUBuffer, kc: GPUBuffer, vc: GPUBuffer, at: G.Rows, o: {
        theta: string;
        cache: G.Cache;
    }): Op;
    attention(q: GPUBuffer, kc: GPUBuffer, vc: GPUBuffer, out: GPUBuffer, at: G.Rows, o: {
        cache: G.Cache;
    }): Op;
    attnStep(qkv: GPUBuffer, kc: GPUBuffer, vc: GPUBuffer, parts: GPUBuffer, o: {
        splits: number;
        theta: string;
        align?: {
            at: G.AlignRows;
            buf: GPUBuffer;
        };
    }): Op;
    alignRow(q: GPUBuffer, kc: GPUBuffer, al: GPUBuffer, at: G.AlignRows): Op;
    embedRows(ids: GPUBuffer, x: GPUBuffer, rows: number, o: {
        bos: number;
        pad: number;
    }): Op;
    noise(z: GPUBuffer): Op;
    finishStep(x: GPUBuffer, lat: GPUBuffer, lats: GPUBuffer, o: {
        eps: string;
        threshold: string;
    }): Op;
    upsample(lats: GPUBuffer, y: GPUBuffer, T: number): Op;
    shiftRows(stages: G.Stage<GPUBuffer>[], T: number): Op;
}
export {};
