export type Act = "none" | "gelu" | "silu";
/** none | add (y = r + acc) | add_y (y += acc) | gate (y = r + g * acc) |
 *  gate_y (y += g * acc) | vec (y = acc + v, v from the blob) */
export type Residual = "none" | "add" | "add_y" | "gate" | "gate_y" | "vec";
/** The state buffer's u32 slots: the position the next LM step feeds, the frame it makes, the
 *  frames to keep once the end of speech is seen (0 before), the first 200 Hz position of the
 *  codec's chunk, the text's length in tokens, and the frames to keep after the end-of-speech
 *  frame. */
export declare const STATE: {
    readonly pos: 0;
    readonly frame: 1;
    readonly end: 2;
    readonly chunk: 3;
    readonly text: 4;
    readonly after: 5;
};
/** Where the alignment head's scores go: one row per frame over the text tokens, which start
 *  at position `lo`, `width` floats per row. */
export interface Align {
    lo: number;
    width: number;
}
/** Which work returns at once past the end: none, an LM step (its frame) or a codec chunk (its
 *  first frame). */
export type Gate = "none" | "step" | "chunk";
/** The two transformers' key and value caches: the LM's, f16 at fixed positions, or the
 *  codec's, an f32 ring at the chunk's position over a window, its work gated past the end. */
export type Cache = {
    kind: "lm";
} | {
    kind: "codec";
    window: number;
    ring: number;
};
/** Each kernel's uniform fields, in order: its WGSL struct and the values graph.ts writes. */
export declare const UNIFORMS: {
    readonly layernorm: readonly ["go", "bo"];
    readonly rows: readonly ["rows", "pos0"];
    readonly embedRows: readonly ["eo", "eso", "bo"];
    readonly noise: readonly ["seed", "scale"];
    readonly upsample: readonly ["T", "wo", "wso", "mo", "so"];
    readonly shiftRows: readonly ["src", "count"];
    readonly gemm: readonly ["M", "yo", "ro", "rows", "wo", "bo", "wso"];
    readonly gemmReduce: readonly ["rows", "yo", "ro", "stride", "bo"];
    readonly gemv: readonly ["xo", "yo", "ro", "go", "so", "ho", "wo", "bo", "vo", "lgo", "lbo", "wso"];
    readonly finishStep: readonly ["ewo", "ebo", "ngo", "nbo"];
};
export type Fields = readonly string[];
export interface ConvShape {
    cin: number;
    k?: number;
    dil?: number;
    elu?: boolean;
    /** Transposed convolutions: the stride, one GEMM per output phase. */
    stride?: number;
}
export interface GemmOptions {
    K: number;
    N: number;
    act?: Act;
    bias?: boolean;
    res?: Residual;
    load?: "linear" | "conv" | "convT";
    conv?: ConvShape | null;
    splits?: number;
    /** int8 weights, else f16. */
    q8: boolean;
    gate: Gate;
}
export interface GemvOptions {
    K: number;
    N: number;
    act?: Act;
    bias?: boolean;
    res?: Residual;
    ln?: {
        affine: boolean;
        modulate: boolean;
        eps: string;
    } | null;
    combine?: number;
    /** 32-lane subgroup sums, else shared memory. */
    sg: boolean;
}
/** int8 weights per f16 scale. */
export declare const Q8_GROUP = 32;
/** LayerNorm over 512 channels per row (one workgroup each), with its affine weights. */
export declare function layernorm({ eps, gate }: {
    eps: string;
    gate: Gate;
}): string;
/** RoPE on q and k of `rows` rows, writing q to qb and k, v to the cache at their positions
 *  (row m at pos0 + m, past the cache's own position). */
export declare function ropeKV({ theta, cache }: {
    theta: string;
    cache: Cache;
}): string;
/** Causal (optionally windowed) attention for many query rows, flash style: a workgroup takes
 *  16 query rows of one head and walks the keys in tiles of 16 through shared memory with an
 *  online softmax, so each key is read once per 16 rows. Thread (r, c): row r, key c for the
 *  scores and dims 4c..4c+3 for the output. */
export declare function attention({ cache }: {
    cache: Cache;
}): string;
/** Token ids padded with BOS and PAD (lm.ts) to embedding rows: int8 token rows, the f16 BOS
 *  row, zero rows for PAD. */
export declare function embedRows({ bos, pad }: {
    bos: number;
    pad: number;
}): string;
/** Noise for the next latent: Box-Muller over a counter hash of the seed and frame. */
export declare function noise(): string;
/** Codec input: denormalize latents and run the folded latent_proj + depthwise upsample (int8),
 *  for the chunk starting at the state's codec position (200 Hz). */
export declare function upsample(): string;
/** Moves the last `ctx` rows of a stage buffer to its front, for the next chunk. */
export declare function shiftRows(): string;
/** GEMM: 64x64 tiles, K in steps of 32, optional split-K. With splits > 1 it writes raw
 *  partial sums, and gemmReduce finishes them. A convolution reads its input stage from row 0,
 *  where its (k - 1) * dil rows of left context begin; a transposed one has one row of context
 *  and runs one GEMM per output phase. Each thread loads one 16-byte word of weights per K step
 *  (8 f16, or 16 int8 for half the threads). */
export declare function gemm({ K, N, act, bias, res, load, conv, splits, q8, gate, }: GemmOptions): string;
/** Finishes a split-K GEMM: sums the partials, then bias, activation and residual. */
export declare function gemmReduce({ N, splits, act, bias, res, gate, }: {
    N: number;
    splits: number;
    act?: Act;
    bias?: boolean;
    res?: Residual;
    gate: Gate;
}): string;
/** A GEMV's layout for K inputs: 128 threads, 2 rows per 32 lanes (more when a row needs
 *  fewer than 32 lanes), 16-byte weight loads. */
export declare function gemvShape(K: number): {
    WG: number;
    R: number;
    U: number;
    LPR: number;
    RPP: number;
    RS: number;
    rowsPerGroup: number;
};
/** GEMV for one row of an LM step: y[n] = epilogue(sum_k x[k] w[n, k]) with int8 weights.
 *  32 lanes per output row, 2 rows per 32 lanes, 16-byte weight loads. Optional prologues:
 *  LayerNorm (affine and/or adaLN modulation) or the attention splits' merge. */
export declare function gemv({ K, N, act, bias, res, ln, combine, sg, }: GemvOptions): string;
/** The end of an LM step, with the final LayerNorm fused: the EOS logit from LN(x row 0), the
 *  latent kept, the position and frame moved on, and the end set once the logit passes
 *  `threshold` (keeping the state's `after` more frames). */
export declare function finishStep({ eps, threshold }: {
    eps: string;
    threshold: string;
}): string;
/** Split-K decode attention (flash-decoding) for an LM step: workgroup (split, head) of 128
 *  threads rotates q, and the split owning the current position writes its k and v (f16)
 *  first. 8 lanes per key for the scores, 16 lanes x 8 key groups for the values. It leaves
 *  each split's unnormalized output, max and sum for the next GEMV to merge. */
export declare function attnSplit({ splits, theta, sg, align, }: {
    splits: number;
    theta: string;
    sg: boolean;
    /** Keep head 0's scores over the text, for word times. */
    align?: Align;
}): string;
/** Frame 0's alignment row, in the prefill after the alignment layer's attention: head 0's
 *  scores of the BOS row (the last text row, at the state's position) over the text. */
export declare function alignRow({ lo }: Align): string;
