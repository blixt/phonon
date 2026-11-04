export declare const MODEL_CACHE = "phonon-webgpu";
export type Language = "en" | "fr" | "de" | "es" | "pt";
export interface Align {
    layer: number;
    head: number;
    lag: number;
}
/** Each language's default voice. Outside English, the one UTMOS rated highest, with no word
 *  errors, in its 40M model; in English, the benchmark's voice. */
export declare const DEFAULT_VOICE: Record<Language, string>;
export interface ModelFiles {
    /** Gradium's checkpoint the weights come from. */
    source: {
        repo: string;
        revision: string;
    };
    /** The engine's weights, converted from the source (checkpoint.ts). */
    weights: string;
    /** Its tokenizer.5.0.0.json. */
    tokenizer: string;
    languages: Language[];
    license: string;
    /** The attention head whose scores follow the text as it speaks, and how many frames it reads
     *  ahead of the audio: the word times. bench/heads.mjs finds both. */
    align: Align;
}
export declare const MODELS: {
    "phonon-31m-en": {
        source: {
            repo: string;
            revision: string;
        };
        weights: string;
        tokenizer: string;
        languages: "en"[];
        license: string;
        align: {
            layer: number;
            head: number;
            lag: number;
        };
    };
    "phonon-40m-en": ModelFiles;
    "phonon-40m-fr": ModelFiles;
    "phonon-40m-de": ModelFiles;
    "phonon-40m-es": ModelFiles;
    "phonon-40m-pt": ModelFiles;
    "phonon-40m-multilingual": ModelFiles;
};
export type ModelId = keyof typeof MODELS;
/** The voice catalog: one JSON line per voice. */
export declare const CATALOG_URL: string;
export declare const voiceUrl: (id: string) => string;
