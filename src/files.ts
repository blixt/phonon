// Where Phonon's files come from: Hugging Face repositories at pinned
// revisions, so a cached file never meets a changed one. The browser keeps
// them in one cache once downloaded (index.ts). The weights come converted
// (scripts/build-weights.ts) from Gradium's checkpoint.
export const PHONON_REPO = "gradium/phonon-31m-en";
export const PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f";
export const VOICES_REPO = "gradium/phonon-voices";
export const VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
const hubFile = (repo: string, revision: string, path: string) =>
  `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
export const WEIGHTS_REPO = "blixt/phonon-31m-en-webgpu";
export const WEIGHTS_REVISION = "f4635a6dddaa6072451e0155c03c97fbe889b6ec";
export const MODEL_CACHE = "phonon-webgpu";
/** The engine's weights: Gradium's checkpoint at PHONON_REVISION, converted (checkpoint.ts). */
export const WEIGHTS_URL = hubFile(WEIGHTS_REPO, WEIGHTS_REVISION, "phonon-31m-en.int8.bin");
export const TOKENIZER_URL = hubFile(PHONON_REPO, PHONON_REVISION, "tokenizer.5.0.0.json");
/** The voice catalog: one JSON line per voice. */
export const CATALOG_URL = hubFile(VOICES_REPO, VOICES_REVISION, "voices.jsonl");
export const voiceUrl = (id: string) =>
  hubFile(VOICES_REPO, VOICES_REVISION, `voices/${id}.safetensors`);
