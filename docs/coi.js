// A service worker that makes the demo cross-origin isolated on a host that cannot set headers
// (GitHub Pages), so the CPU engine can run threads: it adds COOP and COEP to the page's own
// responses, and CORP to the files it loads. The demo registers it only when it will speak on
// the CPU, then reloads once.
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
