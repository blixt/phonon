import { type Graph } from "./graph";
/** Frames per decoded chunk, the sizes a synthesis goes through. */
export declare const CHUNKS: number[];
/** Audio samples per frame. */
export declare const FRAME = 1920;
export declare class Codec {
    /** The last chunk's samples, up to C_MAX * FRAME floats. */
    readonly audio: GPUBuffer;
    private readonly g;
    private readonly ops;
    private readonly stages;
    constructor(g: Graph, lats: GPUBuffer);
    /** Before a text's first chunk: no context yet. */
    reset(enc: GPUCommandEncoder): void;
    /** Decodes frames [f0, f0 + size) into `audio`. The position goes to the queue ahead of the
     *  submit that holds this encoder, so a submit holds one chunk at most. */
    encodeChunk(enc: GPUCommandEncoder, f0: number, size: number): void;
}
