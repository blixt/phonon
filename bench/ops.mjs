// Runs bench/ops.ts in headless Chrome: every op of graph.ts on WebGPU and on the CPU, with
// Phonon 31M's weights, and prints how far apart the outputs are. Exits with 1 when an output
// differs by more than its tolerance. Usage: node bench/ops.mjs [MODEL] [PROFILE_DIR]
// Needs dist/cpu/phonon-cpu-*.wasm (npm run build-cpu). Playwright from PLAYWRIGHT, or the repo's.
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { build } from "esbuild";
import { serve } from "./serve.mjs";
const root = join(new URL(".", import.meta.url).pathname, "..");
const [model = "phonon-31m-en", profile = join(tmpdir(), "phonon-ops")] = process.argv.slice(2);
mkdirSync(join(root, ".test"), { recursive: true });
await build({ entryPoints: [join(root, "bench/ops.ts")], bundle: true, format: "esm", target: "es2022", outfile: join(root, ".test/ops.js"), logLevel: "warning" });
await build({ entryPoints: [join(root, "src/files.ts")], bundle: true, format: "esm", target: "es2022", outfile: join(root, ".test/files.js"), logLevel: "warning" });
const { server, url } = await serve({ root });
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const ctx = await pw.chromium.launchPersistentContext(profile, { headless: true, channel: "chromium", args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"] });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`${url}/bench/blank.html`);
const result = await page.evaluate(async (model) => {
  const { MODELS } = await import("/.test/files.js");
  const { compareBackends } = await import("/.test/ops.js");
  const cache = await caches.open("phonon-webgpu");
  const weights = MODELS[model].weights;
  let hit = await cache.match(weights);
  if (!hit) {
    await cache.put(weights, await fetch(weights));
    hit = await cache.match(weights);
  }
  const relaxed = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 0, 253, 15, 65, 0, 253, 15, 253, 128, 2, 11]));
  const packed = await hit.arrayBuffer();
  const out = [];
  for (const level of relaxed ? ["relaxed", "simd"] : ["simd"]) {
    const wasm = new Uint8Array(await (await fetch(`/dist/cpu/phonon-cpu-${level}.wasm`)).arrayBuffer());
    const r = await compareBackends(packed, wasm);
    out.push(...r.rows.map((row) => ({ ...row, build: level, subgroups: r.subgroups })));
  }
  return out;
}, model);
await ctx.close();
server.close();
let failed = 0;
for (const r of result) {
  if (!r.ok) failed++;
  console.log(`${r.ok ? "ok  " : "FAIL"} ${r.build}: ${r.case}${r.variant ? ` (${r.variant})` : ""}: ${r.output} (${r.values} values) differs by ${r.diff.toPrecision(3)}, ${(r.rel * 100).toPrecision(3)}% of its largest ${r.max.toPrecision(3)}`);
}
console.log(`${result.length - failed} of ${result.length} outputs agree (WebGPU subgroups: ${result[0]?.subgroups})`);
process.exit(failed ? 1 : 0);
