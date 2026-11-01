export declare const PHONON_REPO = "gradium/phonon-31m-en";
export declare const PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f";
export declare const VOICES_REPO = "gradium/phonon-voices";
export declare const VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
export declare const WEIGHTS_REPO = "blixt/phonon-31m-en-webgpu";
export declare const WEIGHTS_REVISION = "f4635a6dddaa6072451e0155c03c97fbe889b6ec";
export declare const MODEL_CACHE = "phonon-webgpu";
/** The engine's weights: Gradium's checkpoint at PHONON_REVISION, converted (checkpoint.ts). */
export declare const WEIGHTS_URL: string;
export declare const TOKENIZER_URL: string;
/** The voice catalog: one JSON line per voice. */
export declare const CATALOG_URL: string;
export declare const voiceUrl: (id: string) => string;
