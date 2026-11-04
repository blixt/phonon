// The benchmark page. The server (run.mjs) plans the runs: it answers each POST /next with the
// next task. A cold run is a fresh page: it loads one runtime and speaks one phrase. A first visit
// is a cold run that first deletes the origin's storage and fetches past the HTTP cache. A warm run
// is a fresh page too: it loads one runtime, speaks one phrase untimed, then speaks every
// phrase with the task's seed. Nothing is played.
import { countDownloads, countWorkerDownloads, observeCode } from "./netcount.js";
const device = new URLSearchParams(location.search).get("device") ?? "unnamed";
const $ = (id) => document.getElementById(id);
const log = (s) => ($("log").textContent = `${s}\n${$("log").textContent}`.slice(0, 4000));
const post = (path, body) =>
  fetch(path, { method: "POST", body: JSON.stringify(body) }).then((r) => r.json());
const t0 = performance.now(); // milliseconds since the navigation began
const plainFetch = globalThis.fetch;
navigator.wakeLock?.request("screen").catch(() => {});
// A heartbeat, so the server can tell a running page from a crashed one.
setInterval(() => post("/beat", { device }).catch(() => {}), 3000);
post("/beat", { device }).catch(() => {});

const VOICE_REPO = "https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7";
const PHONON = "hf://gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f";
const PHONON_MULTILINGUAL = "hf://gradium/phonon-40m-multilingual@1537ce46c64f42903e0d608b132ccb073dac46d2";
/** Each runtime: load it, and speak a text, calling `first` on its first audio. */
const RUNTIMES = {
  async ours() {
    const { Phonon } = await import("../dist/phonon.js");
    const tts = await Phonon.load();
    return {
      info: "WebGPU, int8",
      speak: async (text, first, seed) => (await tts.speak(text, { voice: "Delia", seed, onAudio: first })).audio,
    };
  },
  async oursMultilingual() {
    const { Phonon } = await import("../dist/phonon.js");
    const tts = await Phonon.load({ model: "phonon-40m-multilingual" });
    return {
      info: "WebGPU, int8, 40M multilingual",
      speak: async (text, first, seed) => (await tts.speak(text, { voice: "Delia", seed, onAudio: first })).audio,
    };
  },
  async oursCpu() {
    return oursCpu();
  },
  async oursCpu1() {
    return oursCpu(1);
  },
  async gradiumWebgpu() {
    return gradium("webgpu");
  },
  async gradiumCpu() {
    return gradium("cpu");
  },
  async gradiumWebgpuMultilingual() {
    return gradium("webgpu", PHONON_MULTILINGUAL);
  },
  async kokoro() {
    const { KokoroTTS } = await import("../node_modules/kokoro-js/dist/kokoro.web.js");
    const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype: "fp32", device: "webgpu" });
    return {
      info: "WebGPU, fp32",
      // Kokoro speaks a sentence per call: its first audio is the first sentence's.
      async speak(text, first) {
        const parts = [];
        for (const sentence of text.match(/[^.!?]+[.!?]*/g).map((s) => s.trim()).filter(Boolean)) {
          parts.push((await tts.generate(sentence, { voice: "af_heart" })).audio);
          first();
        }
        return join(parts);
      },
    };
  },
};
async function oursCpu(threads) {
  const { Phonon } = await import("../dist/phonon.js");
  const tts = await Phonon.load({ backend: "cpu", threads });
  return {
    info: `CPU, ${tts.backend.threads} threads, ${tts.backend.simd}`,
    speak: async (text, first, seed) => (await tts.speak(text, { voice: "Delia", seed, onAudio: first })).audio,
  };
}
async function gradium(where, model = PHONON) {
  const { PhononTTS } = await import("../node_modules/phonon-tts/index.js");
  // No text normalization: this engine has none, so both do the same work.
  const tts = await PhononTTS.load({ lang: "none", model, device: where, quant: "q8" });
  await tts.addVoice("Delia", `${VOICE_REPO}/voices/nGzMHMPAWGaJFHOo.safetensors`);
  return {
    info: `${tts.device}, ${tts.threads} threads (${tts.threadsReason || tts.deviceReason})`,
    async speak(text, first, seed) {
      const parts = [];
      for await (const chunk of tts.stream(text, { voice: "Delia", seed })) {
        first();
        parts.push(chunk);
      }
      return join(parts);
    },
  };
}
function join(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}
/** Speaks `text`: the time to its first audio, the whole time, and the audio. */
async function measure(engine, text, seed = 1) {
  const start = performance.now();
  let firstAt = 0;
  const audio = await engine.speak(text, () => (firstAt ||= performance.now()), seed);
  const end = performance.now();
  return { firstMs: firstAt - start, totalMs: end - start, audioSecs: audio.length / 24000, audio };
}
function wavBase64(pcm) {
  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVEfmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, 24000, true); out.setUint32(28, 48000, true); out.setUint16(32, 2, true);
  out.setUint16(34, 16, true); str(36, "data"); out.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => out.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(v * 32767))), true));
  const bytes = new Uint8Array(out.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
async function adapterInfo() {
  const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  return a ? { vendor: a.info.vendor, architecture: a.info.architecture, description: a.info.description } : null;
}

/** Deletes everything this origin keeps: Cache Storage, IndexedDB and the origin private file
 *  system, where the runtimes keep their files. Answers what is left. */
async function clearStorage() {
  for (const k of await caches.keys()) await caches.delete(k);
  for (const db of (await indexedDB.databases?.()) ?? [])
    await new Promise((done) => {
      const q = indexedDB.deleteDatabase(db.name);
      q.onsuccess = q.onerror = q.onblocked = done;
    });
  const root = await navigator.storage?.getDirectory?.().catch(() => null);
  if (root) for await (const name of root.keys()) await root.removeEntry(name, { recursive: true }).catch(() => {});
  return { caches: (await caches.keys()).length, databases: ((await indexedDB.databases?.()) ?? []).length };
}
/** The connection's speed to Hugging Face's file CDN, in MB/s: 16 MB of a model file, fetched
 *  past the HTTP cache. A first visit's time is mostly its download, so each one notes this. */
async function probeNetwork() {
  const url = "https://huggingface.co/blixt/phonon-31m-en-webgpu/resolve/f4635a6dddaa6072451e0155c03c97fbe889b6ec/phonon-31m-en.int8.bin";
  const start = performance.now();
  const response = await plainFetch(url, { cache: "no-store", headers: { range: "bytes=0-15999999" } });
  const bytes = (await response.arrayBuffer()).byteLength;
  return bytes / 1e6 / ((performance.now() - start) / 1000);
}

/** A first visit's downloads, up to its first audio: the time at least one was running (code
 *  from this server included), the files from other origins with their bytes (as received, and on
 *  the network when compressed), and the paths of the code files. */
function summarize(downloads, from, to) {
  const all = downloads.filter((d) => d.end <= to && d.end >= from);
  const done = all.filter((d) => !d.code);
  const spans = all.map((d) => [Math.max(d.start, from), d.end]).filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  let ms = 0;
  let open = null;
  for (const [a, b] of spans) {
    if (open && a <= open[1]) open[1] = Math.max(open[1], b);
    else {
      if (open) ms += open[1] - open[0];
      open = [a, b];
    }
  }
  if (open) ms += open[1] - open[0];
  return {
    count: done.length,
    after: downloads.filter((d) => d.end > to).length,
    ms,
    bytes: done.reduce((n, d) => n + d.bytes, 0),
    wireBytes: done.reduce((n, d) => n + (d.wire ?? d.bytes), 0),
    compressed: done.filter((d) => d.wire && d.bytes > d.wire).length,
    files: done.map((d) => ({ file: new URL(d.url).pathname.split("/").pop(), bytes: d.bytes, wire: d.wire, ms: Math.round(d.end - d.start) })),
    code: [...new Set(all.filter((d) => d.code).map((d) => new URL(d.url).pathname))],
  };
}

let loaded = null; // { name, engine }
const hello = { device, ua: navigator.userAgent, cores: navigator.hardwareConcurrency, gpu: await adapterInfo(), isolated: crossOriginIsolated };
for (let task = await post("/next", hello); task.kind !== "done"; ) {
  $("task").textContent = `${task.runtime}: ${task.kind} run ${task.run} of ${task.runs}`;
  $("progress").textContent = `Task ${task.index + 1} of ${task.total}. Keep this page open and the screen on.`;
  let result;
  try {
    if (task.kind === "cold" || task.kind === "first") {
      const first = task.kind === "first";
      const left = first ? await clearStorage() : null;
      const downloads = [];
      if (first) {
        countDownloads((d) => downloads.push(d));
        countWorkerDownloads((d) => downloads.push(d));
        observeCode((d) => downloads.push(d));
      }
      const pageMs = performance.now() - t0;
      const loadStart = performance.now();
      const engine = await RUNTIMES[task.runtime]();
      const loadMs = performance.now() - loadStart;
      const speakStart = performance.now();
      const m = await measure(engine, task.phrases[0].text);
      result = { info: engine.info, sinceNavigationMs: performance.now(), pageMs, loadMs, phrases: [{ id: task.phrases[0].id, firstMs: m.firstMs, totalMs: m.totalMs, audioSecs: m.audioSecs }] };
      if (first)
        Object.assign(result, {
          storageLeft: left,
          downloads: summarize(downloads, performance.timeOrigin + loadStart, performance.timeOrigin + speakStart + m.firstMs),
          storedBytes: (await navigator.storage?.estimate?.())?.usage ?? null,
          // After the measured run, so it cannot change it.
          networkMBps: await probeNetwork().catch(() => null),
        });
    } else {
      if (loaded?.name !== task.runtime) {
        loaded = { name: task.runtime, engine: await RUNTIMES[task.runtime]() };
        await measure(loaded.engine, "This sentence warms the runtime up.");
      }
      const phrases = [];
      for (const p of task.phrases) {
        const m = await measure(loaded.engine, p.text, task.seed);
        phrases.push({ id: p.id, firstMs: m.firstMs, totalMs: m.totalMs, audioSecs: m.audioSecs });
        if (task.saveAudio) await post("/audio", { device, runtime: task.runtime, id: p.id, run: task.run, wav: wavBase64(m.audio) });
      }
      result = { info: loaded.engine.info, phrases };
    }
    log(`${task.runtime} ${task.kind} ${task.run}: ${result.phrases.map((p) => `${p.id} ${p.firstMs.toFixed(0)}/${p.totalMs.toFixed(0)} ms`).join(", ")}`);
  } catch (e) {
    result = { error: `${e?.name ?? "Error"}: ${e?.message ?? e}\n${e?.stack ?? ""}`.slice(0, 2000) };
    log(`${task.runtime} ${task.kind} ${task.run}: ${result.error.slice(0, 200)}`);
  }
  const next = await post("/next", { ...hello, done: { id: task.id, ...result } });
  // A cold run needs a fresh page, and so does a warm run of a runtime this page has not loaded:
  // a page holds one instance of one runtime.
  if (next.kind === "cold" || next.kind === "first" || (next.kind === "warm" && loaded?.name !== next.runtime)) {
    location.reload();
    await new Promise(() => {});
  }
  task = next;
}
$("task").textContent = "Done. Thank you!";
$("progress").textContent = "";
