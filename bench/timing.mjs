// Speaks every benchmark phrase in a few voices with this engine in headless Chrome, and keeps
// the audio and the engine's word times for align.py. Usage: node bench/timing.mjs OUT_DIR
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { serve } from "./serve.mjs";
const out = process.argv[2];
mkdirSync(out, { recursive: true });
const root = join(new URL(".", import.meta.url).pathname, "..");
const { phrases } = JSON.parse(readFileSync(join(root, "bench/phrases.json"), "utf8"));
const { server, url } = await serve({ root });
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const browser = await pw.chromium.launch({ channel: "chromium", args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"] });
const page = await browser.newPage();
await page.goto(`${url}/docs/index.html`);
await page.waitForFunction(() => globalThis.phonon, null, { timeout: 300000 });
for (const voice of ["Delia", "Miles", "Maeve", "Hugh"])
  for (const p of phrases) {
    const r = await page.evaluate(async ([text, voice]) => {
      const s = await phonon.speak(text, { voice });
      return { words: s.words, pcm: Array.from(s.audio) };
    }, [p.text, voice]);
    const id = `${voice}-${p.id}`;
    writeFileSync(join(out, `${id}.json`), JSON.stringify({ id, voice, text: p.text, words: r.words }));
    writeFileSync(join(out, `${id}.f32`), Buffer.from(new Float32Array(r.pcm).buffer));
  }
await browser.close();
server.close();
