// Finds a model's alignment head and lag, step 1 of 2: speaks texts once per attention head and
// records each take's alignment rows, audio and tokens, then replays the word times at each lag.
// Usage: node bench/heads.mjs MODEL TEXTS.json VOICES OUT [LAYERS] [HEADS]
//   TEXTS: [{ id, text }] in the model's language. VOICES: "Mats,Nele". LAYERS, HEADS: "0,1,2".
// Then: python bench/heads.py OUT LANG, which scores every head and lag against MMS forced
// alignment and prints the choice for src/files.ts. Run it under the GPU lock you use for benches.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { serve } from "./serve.mjs";

const [model, textsFile, voicesArg, out, layersArg, headsArg = "0,1,2,3,4,5,6,7"] = process.argv.slice(2);
const root = join(new URL(".", import.meta.url).pathname, "..");
const texts = JSON.parse(readFileSync(textsFile, "utf8"));
const voices = voicesArg.split(",");
const LAGS = [1.5, 2, 2.5, 3, 3.5, 4];
// The page loads the library and, for the replays, the timing code on its own.
execFileSync(join(root, "node_modules/.bin/esbuild"), ["src/timing.ts", "--bundle", "--format=esm", "--outfile=.test/timing.js", "--log-level=warning"], { cwd: root });
const { server, url } = await serve({ root });
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const browser = await pw.chromium.launch({ channel: "chromium", args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`${url}/docs/listen.html`);
const layers = layersArg ? layersArg.split(",").map(Number) : null;
const depth = await page.evaluate(async (model) => (await import("/dist/phonon.js")).MODELS[model].weights.includes("40m") ? 9 : 6, model);
for (const l of layers ?? Array.from({ length: depth }, (_, i) => i))
  for (const h of headsArg.split(",").map(Number)) {
    const takes = await page.evaluate(async ([model, l, h, texts, voices, lags]) => {
      const { Phonon, MODELS } = await import("/dist/phonon.js");
      const { WordTimes } = await import("/.test/timing.js");
      MODELS[model].align = { ...MODELS[model].align, layer: l, head: h };
      const tts = await Phonon.load({ model });
      const out = [];
      for (const voice of voices)
        for (const t of texts) {
          await tts.speak("Ok.", { voice }); // the voice's prefill, before the timed take
          const v = tts.voices.find((x) => x.name === voice);
          const ids = tts.tokenizer.encode(t.text);
          const pieces = ids.map((i) => tts.tokenizer.pieces[i]);
          const rows = [], chunks = [];
          await tts.engine.synthesize(ids, v.id, { seed: 1, words: t.text.split(/\s+/).length, onRows: (r) => rows.push(r), onChunk: (c) => chunks.push(c) });
          const pcm = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
          let o = 0;
          for (const c of chunks) (pcm.set(c, o), (o += c.length));
          const byLag = {};
          for (const lag of lags) {
            const times = new WordTimes(pieces, lag);
            for (const r of rows) times.addRows(r);
            times.addAudio(pcm);
            byLag[lag] = times.finish().words;
          }
          let bin = "";
          const u8 = new Uint8Array(pcm.buffer);
          for (let k = 0; k < u8.length; k += 0x8000) bin += String.fromCharCode(...u8.subarray(k, k + 0x8000));
          out.push({ id: `${voice}-${t.id}`, voice, text: t.text, byLag, pcm: btoa(bin) });
        }
      tts.engine.g.device.destroy();
      return out;
    }, [model, l, h, texts, voices, LAGS]);
    const dir = join(out, `L${l}H${h}`);
    mkdirSync(dir, { recursive: true });
    for (const take of takes) {
      writeFileSync(join(dir, `${take.id}.f32`), Buffer.from(take.pcm, "base64"));
      delete take.pcm;
      writeFileSync(join(dir, `${take.id}.json`), JSON.stringify(take));
    }
    console.log(`L${l}H${h}: ${takes.length} takes`);
  }
await browser.close();
server.close();
