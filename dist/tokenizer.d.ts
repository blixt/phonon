interface TokenizerJson {
    model: {
        vocab: [string, number][];
        unk_id?: number;
    };
    normalizer?: Normalizer | {
        type: "Sequence";
        normalizers: Normalizer[];
    };
}
type Normalizer = {
    type: "Replace";
    pattern: {
        Regex?: string;
        String?: string;
    };
    content: string;
} | {
    type: "Strip";
    strip_left: boolean;
    strip_right: boolean;
} | {
    type: "Prepend";
    prepend: string;
};
export declare class Tokenizer {
    readonly pieces: string[];
    private readonly score;
    private readonly byte;
    private readonly maxLen;
    private readonly unk;
    private readonly norm;
    constructor(json: TokenizerJson);
    normalize(text: string): string;
    /** Ids for `text`, as the reference tokenizer gives them (no special tokens). */
    encode(text: string): number[];
    /** Viterbi over one word: its best-scoring pieces, bytes for what no piece covers. */
    private word;
}
export {};
