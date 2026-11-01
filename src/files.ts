// Where Phonon's files come from: Gradium's Hugging Face repositories at
// pinned revisions, so a cached file never meets a changed one. The browser
// keeps them in one cache once downloaded (index.ts).
export const PHONON_REPO = "gradium/phonon-31m-en";
export const PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f";
export const VOICES_REPO = "gradium/phonon-voices";
export const VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
const hubFile = (repo: string, revision: string, path: string) =>
  `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
export const MODEL_CACHE = "phonon-webgpu";
/** The checkpoint: the flow LM in bf16 and the codec's decoder in f32. */
export const CHECKPOINT_URL = hubFile(PHONON_REPO, PHONON_REVISION, "model.safetensors");
/** The cache's key for the engine's form of the checkpoint (checkpoint.ts), kept in its place. */
export const MODEL_URL = `${CHECKPOINT_URL}?form=int8-1`;
export const TOKENIZER_URL = hubFile(PHONON_REPO, PHONON_REVISION, "tokenizer.5.0.0.json");
/** The voice catalog: one JSON line per voice. */
export const CATALOG_URL = hubFile(VOICES_REPO, VOICES_REVISION, "voices.jsonl");
export const voiceUrl = (id: string) =>
  hubFile(VOICES_REPO, VOICES_REVISION, `voices/${id}.safetensors`);
