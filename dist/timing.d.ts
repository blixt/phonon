export declare const SAMPLE_RATE = 24000;
export interface Span {
    text: string;
    /** Seconds from the start of the audio. */
    start: number;
    end: number;
}
/** Where the head reads in each frame: the mean position near its peak, never moving back. */
export declare function alignPositions(scores: Float32Array, frames: number, n: number): Float32Array;
/** Token and word spans of one synthesis. `pieces` are its tokens' pieces, a word's first
 *  piece marked with ▁. */
export declare function spans(scores: Float32Array, pieces: string[], pcm: Float32Array): {
    tokens: Span[];
    words: Span[];
};
