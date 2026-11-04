// The backends against the PyTorch reference and each other, in headless Chrome: speaks each clip
// of bench/golden.json (golden.py) on WebGPU and on the CPU with its seed, and checks that each take
// has the reference's length and stays near its log-mel frames, and that the two backends' word
// starts agree. Exits with 1 when a check fails. Usage: node bench/golden.mjs [PROFILE_DIR]
// Needs a build (dist). Playwright from PLAYWRIGHT, or the repo's.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { serve } from "./serve.mjs";
const root = join(new URL(".", import.meta.url).pathname, "..");
const golden = JSON.parse(readFileSync(join(root, "bench/golden.json"), "utf8"));
/** A take's mean log-mel distance from the reference's, over the shorter, as parity.py measures. */
const MAX_MEL_DB = 4;
/** The backends' word starts: the largest difference, in seconds (one frame). */
const MAX_WORD_GAP = 0.08;

// parity.py's log-mel: a 512-point STFT (Hann window of 480, hop 480, reflected edges), power
// in 40 mel bands from 60 Hz to 11 kHz, in dB.
const N = 512;
const HOP = 480;
const bank = (() => {
  const hz = (m) => 700 * (10 ** (m / 2595) - 1);
  const mel = (h) => 2595 * Math.log10(1 + h / 700);
  const lo = mel(60);
  const hi = mel(11000);
  const bins = Array.from({ length: 42 }, (_, i) => Math.floor(((N + 1) * hz(lo + ((hi - lo) * i) / 41)) / 24000));
  return Array.from({ length: 40 }, (_, i) => {
    const [l, c, r] = [bins[i], bins[i + 1], bins[i + 2]];
    const f = new Float64Array(N / 2 + 1);
    for (let k = l; k < c; k++) f[k] = (k - l) / Math.max(1, c - l);
    for (let k = c; k < r; k++) f[k] = (r - k) / Math.max(1, r - c);
    return f;
  });
})();
function fftPower(re) {
  const im = new Float64Array(N);
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) [re[i], re[j]] = [re[j], re[i]];
  }
  for (let len = 2; len <= N; len <<= 1) {
    const a = (-2 * Math.PI) / len;
    for (let i = 0; i < N; i += len)
      for (let k = 0; k < len / 2; k++) {
        const [c, s] = [Math.cos(a * k), Math.sin(a * k)];
        const [xr, xi] = [re[i + k + len / 2], im[i + k + len / 2]];
        const tr = xr * c - xi * s;
        const ti = xr * s + xi * c;
        re[i + k + len / 2] = re[i + k] - tr;
        im[i + k + len / 2] = im[i + k] - ti;
        re[i + k] += tr;
        im[i + k] += ti;
      }
  }
  return Float64Array.from({ length: N / 2 + 1 }, (_, k) => re[k] * re[k] + im[k] * im[k]);
}
function logmel(x) {
  const pad = N / 2;
  const at = (i) => x[i < 0 ? -i : i >= x.length ? 2 * (x.length - 1) - i : i];
  const frames = 1 + Math.floor(x.length / HOP);
  const out = [];
  for (let t = 0; t < frames; t++) {
    const buf = new Float64Array(N);
    for (let n = 0; n < 480; n++) buf[16 + n] = at(t * HOP - pad + 16 + n) * (0.5 - 0.5 * Math.cos((2 * Math.PI * n) / 480));
    const p = fftPower(buf);
    out.push(bank.map((f) => 10 * Math.log10(f.reduce((s, w, k) => s + w * p[k], 0) + 1e-8)));
  }
  return out;
}
const distance = (a, b) => {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let t = 0; t < n; t++) s += Math.sqrt(a[t].reduce((q, v, i) => q + (v - b[t][i]) ** 2, 0) / 40);
  return s / n;
};

const { server, url } = await serve({ root });
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const ctx = await pw.chromium.launchPersistentContext(process.argv[2] ?? join(tmpdir(), "phonon-golden"), {
  headless: true,
  channel: "chromium",
  args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"],
});
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`${url}/bench/blank.html`);
const takes = await page.evaluate(async (clips) => {
  const { Phonon } = await import("/dist/phonon.js");
  const out = {};
  for (const backend of ["webgpu", "cpu"]) {
    const tts = await Phonon.load({ backend });
    out[backend] = [];
    for (const c of clips) {
      const s = await tts.speak(c.text, { voice: "Delia", seed: c.seed });
      out[backend].push({ audio: Array.from(s.audio), words: s.words.map((w) => w.start) });
    }
  }
  return out;
}, golden.clips.map(({ text, seed }) => ({ text, seed })));
await ctx.close();
server.close();

let failed = 0;
const check = (ok, line) => {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${line}`);
};
for (const [i, c] of golden.clips.entries()) {
  const bytes = Buffer.from(c.mel, "base64");
  const q = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const ref = Array.from({ length: c.melFrames }, (_, t) => Array.from(q.subarray(t * 40, t * 40 + 40), (v) => v * golden.melStepDb));
  const mels = {};
  for (const backend of ["webgpu", "cpu"]) {
    const take = takes[backend][i];
    const frames = take.audio.length / 1920;
    mels[backend] = logmel(take.audio);
    const db = distance(mels[backend], ref);
    check(frames === c.frames && db <= MAX_MEL_DB, `${c.id} seed ${c.seed}, ${backend}: ${frames} frames (reference ${c.frames}), ${db.toFixed(2)} dB from the reference`);
  }
  const [a, b] = [takes.webgpu[i].words, takes.cpu[i].words];
  const gap = a.length === b.length ? Math.max(...a.map((s, j) => Math.abs(s - b[j]))) : Infinity;
  check(gap <= MAX_WORD_GAP, `${c.id} seed ${c.seed}, the backends: ${distance(mels.webgpu, mels.cpu).toFixed(2)} dB apart, word starts at most ${(gap * 1000).toFixed(0)} ms apart`);
}
console.log(failed ? `${failed} checks failed` : "All checks passed");
process.exit(failed ? 1 : 0);
