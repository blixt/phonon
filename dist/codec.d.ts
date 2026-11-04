import { type Encoder, type Graph } from "./graph";
/** Frames per decoded chunk, the sizes a synthesis goes through. */
export declare const CHUNKS: number[];
/** Audio samples per frame. */
export declare const FRAME = 1920;
export declare class Codec<B, O> {
    /** The last chunk's samples, up to C_MAX * FRAME floats. */
    readonly audio: B;
    private readonly g;
    private readonly ops;
    /** Each stage: `ctx` rows of left context (a reading convolution's (kernel - 1) * dilation),
     *  then `rate` rows per 200 Hz position. */
    private readonly stages;
    constructor(g: Graph<B, O>, lats: B);
    /** Before a text's first chunk: no context yet. */
    reset(enc: Encoder<B, O>): void;
    /** Decodes frames [f0, f0 + size) into `audio`. The position goes to the queue ahead of the
     *  submit that holds this encoder, so a submit holds one chunk at most. */
    encodeChunk(enc: Encoder<B, O>, f0: number, size: number): void;
}
