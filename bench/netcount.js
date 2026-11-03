// Counts the downloads of a page or a worker during a first visit: each fetch to another origin
// goes past the HTTP cache, and reports its bytes and when it started and ended (its last byte),
// in milliseconds since the epoch, so page and worker times compare.
// `wire` is the response's Content-Length: the bytes on the network when it was compressed.
// The runtimes' own code comes from this page's server: `observeCode` reports each such file
// (`code: true`), whose size on a real site the analysis takes from the file itself.

/** Patches `fetch` in this scope. `report` hears each finished download. */
export function countDownloads(report) {
  const plain = globalThis.fetch;
  const now = () => performance.timeOrigin + performance.now();
  globalThis.fetch = async (input, init) => {
    const url = new URL(typeof input === "string" ? input : (input.url ?? String(input)), location.href);
    if (url.origin === location.origin || url.protocol === "blob:" || url.protocol === "data:") return plain(input, init);
    const start = now();
    const response = await plain(input, { ...init, cache: "reload" });
    const wire = Number(response.headers.get("content-length")) || null;
    if (!response.body) {
      report({ url: url.href, start, end: now(), bytes: 0, wire });
      return response;
    }
    let bytes = 0;
    const counted = response.body.pipeThrough(
      new TransformStream({
        transform(chunk, out) {
          bytes += chunk.byteLength;
          out.enqueue(chunk);
        },
        flush() {
          report({ url: url.href, start, end: now(), bytes, wire });
        },
      }),
    );
    const out = new Response(counted, { status: response.status, statusText: response.statusText, headers: response.headers });
    Object.defineProperties(out, { url: { value: response.url }, redirected: { value: response.redirected } });
    return out;
  };
}

/** Reports each file this scope loads from its own origin, other than the benchmark's own. */
export function observeCode(report) {
  const origin = new URL(location.href.replace(/^blob:/, "")).origin;
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      const url = new URL(e.name);
      if (url.origin !== origin || url.pathname.startsWith("/bench/")) continue;
      report({ url: e.name, start: performance.timeOrigin + e.startTime, end: performance.timeOrigin + e.responseEnd, bytes: e.decodedBodySize, wire: e.encodedBodySize || null, code: true });
    }
  }).observe({ type: "resource", buffered: true });
}

// In a worker started by `countWorkerDownloads`: count, and post each download to the page.
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  const post = (d) => self.postMessage({ benchDownload: d });
  countDownloads(post);
  observeCode(post);
}

/** Makes every module worker the page starts count its downloads too (phonon-tts downloads its
 *  model in one): each starts as a small module that imports this file, then the real one. */
export function countWorkerDownloads(report) {
  const Native = globalThis.Worker;
  const here = import.meta.url;
  globalThis.Worker = class extends Native {
    constructor(url, options) {
      if (options?.type !== "module") {
        super(url, options);
        return;
      }
      const source = `import ${JSON.stringify(here)};\nimport ${JSON.stringify(new URL(url, location.href).href)};\n`;
      super(URL.createObjectURL(new Blob([source], { type: "text/javascript" })), options);
      this.addEventListener("message", (e) => e.data?.benchDownload && report(e.data.benchDownload));
    }
  };
}
