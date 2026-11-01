// Engine side of the parity check: speaks each benchmark text with the voice Delia and seed 1
// in headless Chrome, and keeps the audio for parity.py. Usage: node bench/parity.mjs OUT_DIR
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
for (const p of phrases) {
  const pcm = await page.evaluate(async (text) => Array.from((await phonon.speak(text, { voice: "Delia", seed: 1 })).audio), p.text);
  writeFileSync(join(out, `${p.id}.f32`), Buffer.from(new Float32Array(pcm).buffer));
  writeFileSync(join(out, `${p.id}.json`), JSON.stringify({ id: p.id, text: p.text }));
}
await browser.close();
server.close();
