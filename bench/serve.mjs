// Static server for the demo and benchmark pages, with cross-origin isolation (Wasm threads).
// `HF_HUB_CACHE` serves Hugging Face files from a local hub cache at /hub/ (tests offline).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, normalize } from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".mp3": "audio/mpeg", ".wav": "audio/wav" };

/** Serves `root` at `port` (0: any). `onPost(path, body)` takes POSTed JSON. */
export function serve({ root, port = 0, host = "127.0.0.1", hub = process.env.HF_HUB_CACHE, onPost }) {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      const headers = { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp", "cross-origin-resource-policy": "cross-origin", "access-control-allow-origin": "*", "cache-control": "no-store" };
      const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
      if (req.method === "POST") {
        let body = "";
        for await (const c of req) body += c;
        const answer = (await onPost?.(path, JSON.parse(body || "null"))) ?? {};
        res.writeHead(200, { ...headers, "content-type": "application/json" });
        res.end(JSON.stringify(answer));
        return;
      }
      let file;
      const m = hub && path.match(/^\/hub\/([^/]+)\/([^/]+)\/resolve\/([^/]+)\/(.+)$/);
      if (m) file = join(hub, `models--${m[1]}--${m[2]}`, "snapshots", m[3], m[4]);
      else file = normalize(join(root, path.endsWith("/") ? path + "index.html" : path));
      if (!m && !file.startsWith(root)) { res.writeHead(403); res.end(); return; }
      try {
        const s = await stat(file);
        if (!s.isFile()) throw new Error();
        // Pages revalidate; code and data stay in the browser's cache, as a deployed site's would.
        const cache = extname(file) === ".html" ? "no-cache" : "max-age=86400";
        res.writeHead(200, { ...headers, "cache-control": cache, "content-type": TYPES[extname(file)] ?? "application/octet-stream", "content-length": s.size });
        res.end(await readFile(file));
      } catch { res.writeHead(404, headers); res.end(); }
    });
    server.listen(port, host, () => resolve({ server, url: `http://${host === "0.0.0.0" ? "127.0.0.1" : host}:${server.address().port}` }));
  });
}
