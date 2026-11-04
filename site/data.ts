// What the page shows: the benchmark's results (docs/results.json, from bench/plots.py), its
// phrases, and the names and labels the page and its build share.
import results from "../docs/results.json";
import phrases from "../bench/phrases.json";
import type { ModelId } from "../src/files";

export type Stat = { median: number; p25: number; p75: number; n: number } | null;
export type Metric = "warmRealtime" | "warmFirstMs" | "coldStartMs" | "firstVisitMs";
type RuntimeResult = { errors: string[]; firstVisitMB?: Stat; firstVisitMeasuredMs?: Stat; firstVisitNetworkMBps?: Stat } & Record<Metric, Stat>;
type Results = {
  results: Record<string, Record<string, RuntimeResult>>;
  phrases: Record<string, Record<string, [number, number, string][]>>;
  timing: { errorsMs: number[]; words: number; medianMs: number; medianSignedMs: number; within100ms: number };
  quality: { wer: Record<string, number[]>; utmos: Record<string, number[]> };
};
const data = results as unknown as Results;
export const RESULTS = data.results;
export const PHRASES = data.phrases;
export const TIMING = data.timing;
export const QUALITY = data.quality;
export const TEXTS = phrases.phrases;
/** Megabytes a runtime stores after a first visit (the same files on every device). */
const mb = (rt: string) => {
  const s = Object.values(RESULTS).find((d) => d[rt]?.firstVisitMB)?.[rt]?.firstVisitMB;
  return s ? s.median.toFixed(1) : "?";
};

/** The demo's models, grouped by license, Phonon 31M first. */
export const MODEL_MENU: { license: string; models: { id: ModelId; label: string }[] }[] = [
  { license: "Apache 2.0", models: [{ id: "phonon-31m-en", label: "Phonon 31M, English" }] },
  {
    license: "CC BY-NC 4.0, non-commercial",
    models: [
      { id: "phonon-40m-en", label: "Phonon 40M, English" },
      { id: "phonon-40m-fr", label: "Phonon 40M, French" },
      { id: "phonon-40m-de", label: "Phonon 40M, German" },
      { id: "phonon-40m-es", label: "Phonon 40M, Spanish" },
      { id: "phonon-40m-pt", label: "Phonon 40M, Portuguese" },
      { id: "phonon-40m-multilingual", label: "Phonon 40M, 5 languages" },
    ],
  },
];
/** Megabytes a first visit downloads with a model. The benchmark measured Phonon 31M. The 40M
 *  models' files are the same size, and the figure is the multilingual model's shorter run
 *  (bench/results/40m, in the README). */
export const downloadMb = (id: ModelId) => (id === "phonon-31m-en" ? mb("ours") : "54.3");

export const RUNTIMES = [
  { id: "ours", name: "blixt/phonon", short: "blixt/phonon", note: "WebGPU" },
  { id: "gradiumWebgpu", name: "phonon-tts WebGPU", short: "phonon-tts", note: "WebGPU" },
  { id: "gradiumCpu", name: "phonon-tts CPU", short: "phonon-tts", note: "CPU" },
  { id: "kokoro", name: "Kokoro 82M", short: "Kokoro 82M", note: "different model" },
];
export const DEVICES = [
  { id: "mac", name: "M3 Max", short: "M3 Max", full: "MacBook Pro, M3 Max, Chrome" },
  { id: "pc", name: "RTX 2080 Ti", short: "2080 Ti", full: "PC, RTX 2080 Ti, Chrome" },
  { id: "iphone", name: "iPhone 17 Pro", short: "iPhone 17", full: "iPhone 17 Pro, Safari" },
];
export type Unit = "x" | "ms" | "s";
export const METRICS: { id: Metric; tab: string; caption: string; unit: Unit }[] = [
  { id: "warmRealtime", tab: "Speed", caption: "Seconds of speech per second of compute, warm.", unit: "x" },
  { id: "warmFirstMs", tab: "First audio", caption: "Milliseconds from the call to the first chunk of audio, warm.", unit: "ms" },
  { id: "coldStartMs", tab: "Cold start", caption: "Milliseconds for a fresh page to load the runtime and speak one sentence, with the files cached.", unit: "ms" },
  {
    id: "firstVisitMs",
    tab: "First visit",
    caption: `Seconds for a fresh page with nothing stored to speak, with its download at 50 Mbit/s: ${mb("ours")} MB, ${mb("gradiumWebgpu")} MB (phonon-tts) or ${mb("kokoro")} MB (Kokoro).`,
    unit: "s",
  },
];

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : v >= 10 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`);
/** A value with its unit, for tooltips. Every metric is stored in milliseconds or × real time. */
export const fmt = (v: number, unit: Unit) => (unit === "x" ? `${Math.round(v)}×` : ms(v));
/** A bar's label: the caption names the unit. */
export const short = (v: number, unit: Unit) =>
  unit === "x" ? `${Math.round(v)}×` : unit === "s" ? (v / 1000).toFixed(1) : v < 10 ? v.toFixed(1) : String(Math.round(v));
