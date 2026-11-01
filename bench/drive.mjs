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
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message.slice(0, 300)));
page.on("crash", () => console.log("[crash]"));
page.on("console", (m) => console.log(`[${m.type()}]`, m.text().slice(0, 300)));
page.on("requestfailed", (r) => console.log("[failed]", r.url().slice(0, 120), r.failure()?.errorText));
await page.goto(url);
await page.waitForFunction(() => document.getElementById("task")?.textContent?.startsWith("Done"), null, { timeout: 0, polling: 2000 });
await (remote ? page.close() : ctx.close());
