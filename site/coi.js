// A service worker that makes the demo cross-origin isolated on a host that cannot set headers
// (GitHub Pages), so the CPU engine can run threads: it adds COOP and COEP to the page's own
// responses, and CORP to the files it loads. It is registered for the demo's URL with "?cpu",
// which a page without WebGPU moves to once, and for the CPU tier's folder cpu/, where its
// workers' scripts are (site.ts isolate), so it controls no other page.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
  e.respondWith(
    fetch(r).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      if (!headers.has("Cross-Origin-Resource-Policy")) headers.set("Cross-Origin-Resource-Policy", "cross-origin");
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
