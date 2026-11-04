import type { Tensor } from "./checkpoint";
import type * as G from "./graph";
/** A GEMM tile (dot products of rows x 4 or 2 x 8 columns, or broadcasts of rows x 8 or 16)
 *  and the GEMV's arithmetic. */
export interface Variant {
    tile: {
        dot: boolean;
        rows: number;
        cols: number;
    };
    gemv: "f32" | "int16";
}
export declare const TILES: Variant["tile"][];
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
    set_tile(dot: number, rows: number, cols: number): void;
    set_gemv(mode: number): void;
    __heap_base: WebAssembly.Global;
    __stack_pointer: WebAssembly.Global;
}
/** Each pool thread's stack. */
export declare const STACK: number;
export declare function simdLevel(): "relaxed" | "simd" | null;
/** The module with its memory import made unshared, for a page that is not cross-origin
 *  isolated: one flag byte in the import section. */
export declare function unshare(bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer>;
/** The module, compiled and instantiated over a new memory of at least `bytes`: shared when
 *  `shared`, else with its memory import made unshared. */
export declare function instantiate(wasm: Uint8Array<ArrayBuffer>, shared: boolean, bytes: number): Promise<{
    module: WebAssembly.Module;
    memory: WebAssembly.Memory;
    exports: {
        run(list: number, n: number): void;
        setup(threads: number, scratch: number, per: number): void;
        rope_table(freqs: number, out: number, n: number): void;
        worker(i: number): void;
        set_tile(dot: number, rows: number, cols: number): void;
        set_gemv(mode: number): void;
        __heap_base: WebAssembly.Global;
        __stack_pointer: WebAssembly.Global;
        memory: WebAssembly.Memory;
    };
}>;
declare class Staging implements G.Staging {
    readonly bytes: ArrayBuffer;
    constructor(bytes: number);
    map(): Promise<void>;
    range(): ArrayBuffer;
    unmap(): void;
}
declare class Encoder implements G.Encoder<CpuBuf, CpuOp> {
    private readonly g;
    private readonly commands;
    constructor(g: CpuGraph);
    run(ops: CpuOp[]): void;
    copy(src: CpuBuf, so: number, dst: CpuBuf, dO: number, bytes: number): void;
    clear(buf: CpuBuf, offset: number, bytes: number): void;
    read(src: CpuBuf, so: number, dst: G.Staging, dO: number, bytes: number): void;
    submit(): void;
}
export declare class CpuGraph implements G.Graph<CpuBuf, CpuOp> {
    readonly weights: CpuBuf;
    readonly state: CpuBuf;
    readonly depth = 1;
    readonly kvBytes = 4;
    readonly threads: number;
    private readonly x;
    private readonly tensors;
    private top;
    private list;
    /** Descriptors of one submit's copies and clears, reused by the next. */
    private readonly temps;
    private tempAt;
    private readonly freqs;
    private table;
    private tableTheta;
    /** ELU'd inputs of convolutions: one buffer for all, and the descriptor fields that name it. */
    private elu;
    private readonly eluRefs;
    /** `x` is the instantiated module, `stacks` the bytes the pool threads' stacks took from the
     *  heap base, and `blob` the weights, copied into the memory. */
    constructor(x: CpuExports, model: {
        blob: Uint8Array;
        tensors: Record<string, Tensor>;
    }, o: {
        threads: number;
        heap: number;
    });
    /** Bytes from the heap, 64-byte aligned, zeroed. The memory grows as needed. */
    buffer(size: number): CpuBuf;
    /** The memory's bytes (a fresh view: the memory can grow). */
    bytes(ptr: number, n: number): Uint8Array<ArrayBuffer>;
    write(buf: CpuBuf, offset: number, data: ArrayBufferView<ArrayBuffer>): void;
    encoder(): Encoder;
    staging(bytes: number): Staging;
    ready(): Promise<void>;
    readonly labels: Map<number, string>;
    readonly times: Map<string, number>;
    /** Kernel variants that compute the same ops (cpu/src/pool.rs), for the fastest on a device. */
    tune(v: Variant): void;
    /** The fastest GEMM tile here: each of TILES on two of the codec's products, best of three. */
    fastestTile(): Variant["tile"];
    /** Runs ops now, in one call. */
    exec(ops: CpuOp[]): void;
    endSubmit(): void;
    /** A descriptor for this submit only. */
    temp(kind: number, values: number[]): CpuOp;
    private desc;
    /** The name the next ops are timed under, in a profile build. */
    label: string;
    private tensor;
    /** A tensor's address, and its scales' (0 for f16). */
    private t;
    private half;
    private q8;
    /** The 32 rotary frequencies of `theta` in f32, and a table of their cos and sin for the
     *  LM's positions. */
    private rope;
    gemv(x: CpuBuf | null, wname: string, y: CpuBuf, o: G.Gemv<CpuBuf>): CpuOp;
    gemm(x: CpuBuf, wname: string, y: CpuBuf, o: G.Gemm<CpuBuf>): CpuOp[];
    layernorm(x: CpuBuf, y: CpuBuf, rows: number, w: string, b: string, eps: string, gate: G.Gate): CpuOp;
    ropeKV(qkv: CpuBuf, q: CpuBuf, kc: CpuBuf, vc: CpuBuf, at: G.Rows, o: {
        theta: string;
        cache: G.Cache;
        gate?: G.Gate;
    }): CpuOp;
    attention(q: CpuBuf, kc: CpuBuf, vc: CpuBuf, out: CpuBuf, at: G.Rows, o: {
        cache: G.Cache;
        gate?: G.Gate;
    }): CpuOp;
    attnStep(qkv: CpuBuf, kc: CpuBuf, vc: CpuBuf, parts: CpuBuf, o: {
        splits: number;
        theta: string;
        align?: {
            at: G.Align;
            buf: CpuBuf;
        };
    }): CpuOp;
    alignRow(q: CpuBuf, kc: CpuBuf, al: CpuBuf, at: G.Align): CpuOp;
    embedRows(ids: CpuBuf, x: CpuBuf, rows: number, o: {
        bos: number;
        pad: number;
    }): CpuOp;
    noise(z: CpuBuf): CpuOp;
    finishStep(x: CpuBuf, lat: CpuBuf, lats: CpuBuf, o: {
        eps: string;
        threshold: string;
    }): CpuOp;
    upsample(lats: CpuBuf, y: CpuBuf, T: number): CpuOp;
    shiftRows(stages: G.Stage<CpuBuf>[], T: number): CpuOp;
}
export {};
