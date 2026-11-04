// Engine side of the parity check: speaks each benchmark text in the voice Delia with each seed,
// on a backend, in headless Chrome, and keeps the audio for parity.py.
// Usage: node bench/parity.mjs OUT_DIR [webgpu|cpu] [SEEDS, e.g. 1,2,3]
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { serve } from "./serve.mjs";
const [out, backend = "webgpu", seeds = "1"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const root = join(new URL(".", import.meta.url).pathname, "..");
const { phrases } = JSON.parse(readFileSync(join(root, "bench/phrases.json"), "utf8"));
const { server, url } = await serve({ root });
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const browser = await pw.chromium.launch({ channel: "chromium", args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"] });
const page = await browser.newPage();
await page.goto(`${url}/bench/blank.html`);
await page.evaluate(async (backend) => {
  const { Phonon } = await import("/dist/phonon.js");
  globalThis.phonon = await Phonon.load({ backend });
}, backend);
for (const seed of seeds.split(",").map(Number))
  for (const p of phrases) {
    const pcm = await page.evaluate(async ([text, seed]) => Array.from((await phonon.speak(text, { voice: "Delia", seed })).audio), [p.text, seed]);
    const id = `${p.id}-s${seed}`;
    writeFileSync(join(out, `${id}.f32`), Buffer.from(new Float32Array(pcm).buffer));
    writeFileSync(join(out, `${id}.json`), JSON.stringify({ id, text: p.text, seed }));
  }
await browser.close();
server.close();
