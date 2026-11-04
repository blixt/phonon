import * as K from "./kernels";
import { type Graph, type Op } from "./graph";
export declare const H = 512;
export declare const PROMPT = 125;
/** Positions the LM keeps: the voice prompt, the text and the frames. */
export declare const S_MAX = 1024;
/** Frames one synthesis may make (61 s). */
export declare const F_MAX = 768;
export declare const TEXT_MAX = 240;
/** Per-row activations of a transformer. */
export interface Rows {
    x: GPUBuffer;
    h: GPUBuffer;
    qkv: GPUBuffer;
    q: GPUBuffer;
    att: GPUBuffer;
    ff: GPUBuffer;
}
export declare function rowBuffers(g: Graph, rows: number): Rows;
/** One transformer layer over `rows` rows. Row m sits at position pos0 + m, past the cache's
 *  own position (kernels.ts Cache). `out` takes the layer's result in place of `B.x`. */
export declare function layerRows(g: Graph, pre: string, o: {
    B: Rows;
    rows: number;
    kc: GPUBuffer;
    vc: GPUBuffer;
    cache: K.Cache;
    pos0: number;
    out?: {
        buf: GPUBuffer;
        yo: number;
    };
}): Op[];
export declare class LanguageModel {
    /** The latents made so far, one row of 32 per frame. */
    readonly lats: GPUBuffer;
    /** One step: a latent, then the next position. */
    readonly step: Op[];
    /** The alignment head's scores over the text, one row of TEXT_MAX per frame. */
    readonly align: GPUBuffer;
    private readonly g;
    private readonly rows;
    private readonly kc;
    private readonly vc;
    private readonly ids;
    /** The noise's uniform (kernels.ts UNIFORMS.noise: seed, then scale), written per text. */
    private readonly noise;
    /** The BOS row's output on its way to row 0, where the step's tail reads. */
    private readonly bos;
    private readonly voiceIn;
    private readonly voiceOps;
    private readonly prefills;
    private readonly tail;
    private readonly cond;
    private readonly ada;
    private readonly fh;
    private readonly ft;
    private readonly z;
    private readonly lat;
    private readonly parts;
    private readonly depth;
    /** The head that follows the text as it speaks (files.ts), its scores kept for word times. */
    private readonly alignLayer;
    private readonly alignAt;
    constructor(g: Graph, depth: number, align: {
        layer: number;
        head: number;
    });
    /** The layers over `rows` rows from position `pos0`, in the prefill's way, and for a text's
     *  prefill frame 0's alignment row. */
    private layers;
    /** A voice's prompt rows through every layer, its keys and values kept in a new buffer. */
    voice(latents: Float32Array): GPUBuffer;
    /** Sets up a text in a voice (its keys and values from `voice`) and encodes its prefill, which
     *  makes frame 0. The state's position starts on BOS, and frame 0's step moves it past. */
    start(enc: GPUCommandEncoder, ids: number[], voice: GPUBuffer, seed: number, temperature: number, framesAfterEos: number): void;
    /** A step's layer: one row at the state's position, attention split over the keys. */
    private stepLayer;
    /** From the final hidden state (row 0 of x): the flow head samples the next latent, then its
     *  EOS logit, the next input row and the next position. */
    private tailOps;
}
