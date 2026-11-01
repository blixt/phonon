// Word and token times from the alignment head (layer 3, head 0) of Phonon's
// LM. Each 80 ms frame's scores over the text give a position in it. The head
// reads about 3 frames (240 ms) ahead of what is heard, and a start that falls
// in silence moves to where the sound begins.
export const SAMPLE_RATE = 24000;
const FRAME_SECS = 0.08;
/** Frames the alignment head reads ahead of the audio. */
const LAG = 3;
/** Quieter than this (dB) is silence. */
const SILENT_DB = -45;
/** Pieces that join the word before them. */
const PUNCTUATION = new Set([".", ",", "!", "?", ";", ":", ")", '"', "'"]);

export interface Span {
  text: string;
  /** Seconds from the start of the audio. */
  start: number;
  end: number;
}

/** Where the head reads in each frame: the mean position near its peak, never moving back. */
export function alignPositions(scores: Float32Array, frames: number, n: number): Float32Array {
  const pos = new Float32Array(frames);
  let run = 0;
  for (let f = 0; f < frames; f++) {
    const row = scores.subarray(f * n, f * n + n);
    let k = 0;
    for (let i = 1; i < n; i++) if (row[i] > row[k]) k = i;
    let num = 0;
    let den = 0;
    for (let i = Math.max(0, k - 2); i <= Math.min(n - 1, k + 2); i++) {
      const p = Math.exp(row[i] - row[k]);
      num += p * i;
      den += p;
    }
    run = Math.max(run, num / den);
    pos[f] = run;
  }
  return pos;
}

/** Loudness per 10 ms, in dB. */
function envelope(pcm: Float32Array): Float32Array {
  const hop = SAMPLE_RATE / 100;
  const out = new Float32Array(Math.floor(pcm.length / hop));
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    for (let j = i * hop; j < (i + 1) * hop; j++) s += pcm[j] * pcm[j];
    out[i] = 20 * Math.log10(Math.sqrt(s / hop) + 1e-9);
  }
  return out;
}

/** Token and word spans of one synthesis. `pieces` are its tokens' pieces, a word's first
 *  piece marked with ▁. */
export function spans(
  scores: Float32Array,
  pieces: string[],
  pcm: Float32Array,
): { tokens: Span[]; words: Span[] } {
  const n = pieces.length;
  const frames = n ? scores.length / n : 0;
  const end = pcm.length / SAMPLE_RATE;
  if (!n || !frames) return { tokens: [], words: [] };
  const pos = alignPositions(scores, frames, n);
  const posAt = (t: number) => {
    const g = (t - FRAME_SECS / 2) / FRAME_SECS - LAG;
    if (g <= 0) return pos[0];
    if (g >= frames - 1) return pos[frames - 1];
    const i = Math.floor(g);
    return pos[i] * (1 - (g - i)) + pos[i + 1] * (g - i);
  };
  const starts: (number | null)[] = Array.from({ length: n }, () => null);
  for (let t = 0; t <= end; t += 0.005) {
    const k = Math.round(posAt(t));
    for (let j = 0; j <= k && j < n; j++) starts[j] ??= t;
  }
  const loud = envelope(pcm);
  const at: number[] = [];
  for (let j = 0; j < n; j++) {
    let i = Math.floor((starts[j] ?? end) * 100);
    while (i < loud.length && loud[i] < SILENT_DB) i++;
    at.push(Math.max(starts[j] ?? end, i / 100, j ? at[j - 1] : 0));
  }
  const tokens = pieces.map((p, j) => ({
    text: p.replaceAll("▁", ""),
    start: at[j],
    end: j + 1 < n ? at[j + 1] : end,
  }));
  const words: Span[] = [];
  pieces.forEach((p, j) => {
    const text = p.replaceAll("▁", "");
    const joins = !p.startsWith("▁") || (!!text && [...text].every((c) => PUNCTUATION.has(c)));
    if (words.length && joins) {
      const w = words.at(-1)!;
      w.text += text;
      w.end = tokens[j].end;
    } else words.push({ text, start: tokens[j].start, end: tokens[j].end });
  });
  return { tokens, words: words.filter((w) => w.text) };
}
