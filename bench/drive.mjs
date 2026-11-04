// Opens the benchmark page in headless Chrome with a profile on disk, and waits for "Done".
// Usage: node bench/drive.mjs URL [PROFILE_DIR]  (Playwright from PLAYWRIGHT, or the repo's node_modules)
// With CDP=http://host:port it drives a Chrome that is already running (another machine's).
import { createRequire } from "node:module";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
const [url, profile = mkdtempSync(join(tmpdir(), "phonon-bench-"))] = process.argv.slice(2);
const pw = createRequire(process.env.PLAYWRIGHT ?? import.meta.url)("playwright-core");
const remote = process.env.CDP && (await pw.chromium.connectOverCDP(process.env.CDP));
const ctx = remote
  ? remote.contexts()[0]
  : await pw.chromium.launchPersistentContext(profile, {
      headless: true,
      channel: "chromium",
      args: ["--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--mute-audio"],
    });
// HF_LOCAL="owner/repo=DIR,…" serves those Hugging Face repositories' files from local copies,
// and answers phonon-tts's lookup of them as for a public repository: phonon-tts loads only public
// repositories in a browser, and Gradium's 40M models are gated. Files go through a local server,
// as a route that fulfils a large body drops the page. It listens on loopback only, and serves
// only the files of a mapped repository, each under a random id the route hands out.
let files = null;
if (process.env.HF_LOCAL) {
  const { createServer } = await import("node:http");
  const { createReadStream, readdirSync, statSync } = await import("node:fs");
  const { randomUUID } = await import("node:crypto");
  const local = Object.fromEntries(process.env.HF_LOCAL.split(",").map((e) => e.split("=")));
  const served = new Map();
  files = createServer((q, r) => {
    const f = served.get(q.url.slice(1));
    r.setHeader("access-control-allow-origin", "*");
    r.setHeader("cross-origin-resource-policy", "cross-origin");
    if (!f) return r.writeHead(404).end();
    r.setHeader("content-length", statSync(f).size);
    createReadStream(f).pipe(r);
  });
  await new Promise((resolve) => files.listen(0, "127.0.0.1", resolve));
  const filesUrl = `http://127.0.0.1:${files.address().port}`;
  const cors = { "access-control-allow-origin": "*", "cross-origin-resource-policy": "cross-origin" };
  await ctx.route(/huggingface\.co\/(api\/models\/)?([^/]+\/[^/]+)\/(resolve|revision)\/([^/?]+)(\/[^?]+)?/, (route) => {
    const [, api, repo, , revision, file] = route.request().url().match(/huggingface\.co\/(api\/models\/)?([^/]+\/[^/]+)\/(resolve|revision)\/([^/?]+)(\/[^?]+)?/);
    if (!local[repo]) return route.continue();
    const names = readdirSync(local[repo]);
    if (api)
      return route.fulfill({ status: 200, contentType: "application/json", headers: cors, body: JSON.stringify({ sha: revision, gated: false, siblings: names.map((rfilename) => ({ rfilename })) }) });
    const name = decodeURIComponent(file ?? "").slice(1);
    if (!names.includes(name)) return route.fulfill({ status: 404, headers: cors });
    const id = randomUUID();
    served.set(id, join(local[repo], name));
    route.fulfill({ status: 302, headers: { ...cors, location: `${filesUrl}/${id}` } });
  });
}
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message.slice(0, 300)));
page.on("crash", () => console.log("[crash]"));
page.on("console", (m) => console.log(`[${m.type()}]`, m.text().slice(0, 300)));
page.on("requestfailed", (r) => console.log("[failed]", r.url().slice(0, 120), r.failure()?.errorText));
await page.goto(url);
// The page reloads itself between runs: every 2 s, read its status, and ask the server whether
// this device's plan is done (a page can finish between two reads of a remote browser).
const device = new URL(url).searchParams.get("device");
const serverDone = () =>
  fetch(new URL("/status", url.replace("localhost", "127.0.0.1")), { method: "POST", body: JSON.stringify({ device }) })
    .then((r) => r.json())
    .then((s) => s.done)
    .catch(() => false);
for (;;) {
  const status = await page.textContent("#task", { timeout: 5000 }).catch(() => "");
  if (status?.startsWith("Done") || (await serverDone())) break;
  await new Promise((r) => setTimeout(r, 2000));
}
await (remote ? page.close() : ctx.close());
files?.close();
