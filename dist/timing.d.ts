export declare const SAMPLE_RATE = 24000;
export interface Span {
    text: string;
    /** Seconds from the start of the audio. */
    start: number;
    end: number;
}
/** A word, with the tokens it is made of. */
export interface Word extends Span {
    tokens: Span[];
}
/** The words of a text's tokens: each word's text and its tokens [from, to). A token that starts
 *  with ▁ starts a word, unless it is only punctuation. */
export declare function wordRanges(pieces: string[]): {
    text: string;
    from: number;
    to: number;
}[];
/** One text's word and token times, from its alignment rows and audio as they arrive. */
export declare class WordTimes {
    private readonly pieces;
    private readonly ranges;
    private readonly onWord?;
    /** Each frame's position before the never-back rule, and after it (pos[0] waits for frame 1). */
    private readonly raw;
    private readonly pos;
    /** Loudness per complete 10 ms window, and the samples not yet in a window. */
    private readonly loud;
    private samples;
    private partial;
    private partialAt;
    /** The next time a token can start at, and each token's start before and after the silence step. */
    private t;
    private readonly starts;
    private readonly at;
    private emitted;
    private ended;
    constructor(pieces: string[], onWord?: (word: Word) => void);
    /** Alignment rows of the next frames, `pieces.length` scores each. */
    addRows(rows: Float32Array): void;
    /** The next samples of the text's audio. */
    addAudio(pcm: Float32Array): void;
    /** The text has ended: every remaining word, and all the times. */
    finish(): {
        tokens: Span[];
        words: Word[];
    };
    private get end();
    /** Frame 0's row comes from the end-of-text query, which in some models reads the last token:
     *  it counts no further than frame 1's. Positions never move back. */
    private settle;
    /** The head's position heard at time t, or null while frames still to come could change it. */
    private posAt;
    /** Takes every time step and token start that can no longer change, then reports words. */
    private advance;
    private token;
    private word;
}
