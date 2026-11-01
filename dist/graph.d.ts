import * as K from "./kernels";
import type { Tensor } from "./checkpoint";
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
export declare class Graph {
    readonly device: GPUDevice;
    readonly weights: GPUBuffer;
    /** The LM's and codec's state, slots as kernels.ts STATE names them. */
    readonly state: GPUBuffer;
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
    /** y = epilogue(W x) for one row (x at element `xo`), optionally normalizing x first.
     *  A step's work, so it returns at once past the end. */
    gemv(x: GPUBuffer, wname: string, ybuf: GPUBuffer, o: {
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
        ln?: {
            w?: string;
            b?: string;
            eps: string;
            mod?: GPUBuffer;
            so?: number;
            ho?: number;
        };
        /** Merge this many attention splits from `parts` as the input. */
        combine?: {
            splits: number;
            parts: GPUBuffer;
        };
    }): Op;
    /** Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. */
    gemm(xbuf: GPUBuffer, wname: string, ybuf: GPUBuffer, o: {
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
    }): Op[];
    layernorm(x: GPUBuffer, y: GPUBuffer, rows: number, w: string, b: string, eps: string, gate: K.Gate): Op;
}
export {};
