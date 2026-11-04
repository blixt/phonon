export declare const MODEL_CACHE = "phonon-webgpu";
export type Language = "en" | "fr" | "de" | "es" | "pt";
export interface ModelFiles {
    /** Gradium's checkpoint the weights come from. */
    source: {
        repo: string;
        revision: string;
    };
    /** The engine's weights, converted from the source (checkpoint.ts). */
    weights: string;
    tokenizer: string;
    languages: Language[];
    license: string;
    /** The attention head whose scores follow the text as it speaks: the word times. */
    align: {
        layer: number;
        head: number;
    };
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
/** The tokenizer's URL: in the source repository, or a copy beside the weights. */
export declare const tokenizerUrl: (m: ModelFiles) => string;
/** The voice catalog: one JSON line per voice. */
export declare const CATALOG_URL: string;
export declare const voiceUrl: (id: string) => string;
