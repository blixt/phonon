// Where Phonon's files come from: Hugging Face repositories at pinned
// revisions, so a cached file never meets a changed one. The browser keeps
// them in one cache once downloaded (index.ts). The weights come converted
// (scripts/build-weights.ts) from Gradium's checkpoints.
const hubFile = (repo: string, revision: string, path: string) =>
  `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
export const MODEL_CACHE = "phonon-webgpu";

export type Language = "en" | "fr" | "de" | "es" | "pt";
/** Each language's default voice. Outside English, the one UTMOS rated highest, with no word
 *  errors, in its 40M model; in English, the benchmark's voice. */
export const DEFAULT_VOICE: Record<Language, string> = { en: "Delia", fr: "Augustin", de: "Mats", es: "Marcos", pt: "Davi" };
export interface ModelFiles {
  /** Gradium's checkpoint the weights come from. */
  source: { repo: string; revision: string };
  /** The engine's weights, converted from the source (checkpoint.ts). */
  weights: string;
  tokenizer: string;
  languages: Language[];
  license: string;
  /** The attention head whose scores follow the text as it speaks: the word times. */
  align: { layer: number; head: number };
}

/** Gradium's 40M models: converted in one repository, each beside its tokenizer, under CC BY-NC 4.0. */
const FORTY_REPO = "blixt/phonon-40m-webgpu";
const FORTY_REVISION = "dde41aa27ed6bed7c874621d8b17e77dfece21bf";
const fortyM = (name: string, revision: string, languages: Language[], align: { layer: number; head: number }): ModelFiles => ({
  source: { repo: `gradium/${name}`, revision },
  weights: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.int8.bin`),
  tokenizer: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.tokenizer.5.0.0.json`),
  languages,
  license: "CC-BY-NC-4.0",
  align,
});

export const MODELS = {
  "phonon-31m-en": {
    source: { repo: "gradium/phonon-31m-en", revision: "92741c5b07d085ff840e26fea8ce43110b87852f" },
    weights: hubFile("blixt/phonon-31m-en-webgpu", "f4635a6dddaa6072451e0155c03c97fbe889b6ec", "phonon-31m-en.int8.bin"),
    tokenizer: "tokenizer.5.0.0.json",
    languages: ["en"],
    license: "Apache-2.0",
    align: { layer: 3, head: 0 },
  },
  "phonon-40m-en": fortyM("phonon-40m-en", "bc7e8ae7e543a7a3e4604753e42fbda76ddf0781", ["en"], { layer: 6, head: 0 }),
  "phonon-40m-fr": fortyM("phonon-40m-fr", "d91086970206b88efd2c88ca971d0628b5968f0f", ["fr"], { layer: 4, head: 1 }),
  "phonon-40m-de": fortyM("phonon-40m-de", "79cfc7b47fcfc0b666657b981eb366415023a862", ["de"], { layer: 3, head: 5 }),
  "phonon-40m-es": fortyM("phonon-40m-es", "e5e052b26e31abd5918ade71d3135e69d51a66f8", ["es"], { layer: 4, head: 6 }),
  "phonon-40m-pt": fortyM("phonon-40m-pt", "4cbfd742a1555d60eb5399c6d296e9f2e07389e9", ["pt"], { layer: 4, head: 2 }),
  "phonon-40m-multilingual": fortyM("phonon-40m-multilingual", "1537ce46c64f42903e0d608b132ccb073dac46d2", ["en", "fr", "de", "es", "pt"], { layer: 4, head: 1 }),
} satisfies Record<string, ModelFiles>;
export type ModelId = keyof typeof MODELS;

/** The tokenizer's URL: in the source repository, or a copy beside the weights. */
export const tokenizerUrl = (m: ModelFiles) =>
  m.tokenizer.startsWith("https:") ? m.tokenizer : hubFile(m.source.repo, m.source.revision, m.tokenizer);

const VOICES_REPO = "gradium/phonon-voices";
const VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
/** The voice catalog: one JSON line per voice. */
export const CATALOG_URL = hubFile(VOICES_REPO, VOICES_REVISION, "voices.jsonl");
export const voiceUrl = (id: string) =>
  hubFile(VOICES_REPO, VOICES_REVISION, `voices/${id}.safetensors`);
