export declare const PHONON_REPO = "gradium/phonon-31m-en";
export declare const PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f";
export declare const VOICES_REPO = "gradium/phonon-voices";
export declare const VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
export declare const MODEL_CACHE = "phonon-webgpu";
/** The checkpoint: the flow LM in bf16 and the codec's decoder in f32. */
export declare const CHECKPOINT_URL: string;
/** The cache's key for the engine's form of the checkpoint (checkpoint.ts), kept in its place. */
export declare const MODEL_URL: string;
export declare const TOKENIZER_URL: string;
/** The voice catalog: one JSON line per voice. */
export declare const CATALOG_URL: string;
export declare const voiceUrl: (id: string) => string;
