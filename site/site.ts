// The web page: the demo with its live numbers, and the benchmark's charts drawn from
// docs/results.json (bench/plots.py).
import { Phonon, type Speech } from "../src/index";
import { MODEL_CACHE, WEIGHTS_URL } from "../src/files";
import results from "../docs/results.json";
import phrases from "../bench/phrases.json";

type Stat = { median: number; p25: number; p75: number; n: number } | null;
type Metric = "warmRealtime" | "warmFirstMs" | "coldStartMs" | "firstVisitMs";
type RuntimeResult = { errors: string[] } & Record<Metric, Stat>;
const RESULTS = results.results as unknown as Record<string, Record<string, RuntimeResult>>;
const PHRASES = (results as unknown as { phrases: Record<string, Record<string, [number, number, string][]>> }).phrases;
const TIMING = (results as unknown as { timing: { errorsMs: number[]; words: number; medianMs: number; medianSignedMs: number; within100ms: number } }).timing;
const QUALITY = (results as unknown as { quality: { clips: Record<string, number>; wer: Record<string, number[]>; utmos: Record<string, number[]> } }).quality;

const RUNTIMES = [
  { id: "ours", name: "blixt/phonon", mb: 43.7 },
  { id: "gradiumWebgpu", name: "phonon-tts WebGPU", mb: 64.5 },
  { id: "gradiumCpu", name: "phonon-tts CPU", mb: 0 },
  { id: "kokoro", name: "Kokoro 82M", mb: 326 },
];
const DEVICES = [
  { id: "mac", name: "M3 Max", full: "MacBook Pro, M3 Max, Chrome" },
  { id: "pc", name: "RTX 2080 Ti", full: "PC, RTX 2080 Ti, Chrome" },
  { id: "iphone", name: "iPhone 17 Pro", full: "iPhone 17 Pro, Safari" },
];
const METRICS: { id: Metric; tab: string; caption: string; unit: "x" | "ms" }[] = [
  { id: "warmRealtime", tab: "Speed", caption: "Seconds of speech per second of compute. Higher is faster.", unit: "x" },
  { id: "warmFirstMs", tab: "First audio", caption: "From the call to the first chunk of audio, warm. Lower is better.", unit: "ms" },
  { id: "coldStartMs", tab: "Cold start", caption: "A fresh page loads the runtime and speaks, files cached. Lower is better.", unit: "ms" },
  {
    id: "firstVisitMs",
    tab: "First visit",
    caption: "The same with nothing cached: a download of 43.7 MB, 64.5 MB (phonon-tts) or 326 MB (Kokoro).",
    unit: "ms",
  },
];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SVG = "http://www.w3.org/2000/svg";
function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.append(e);
  return e;
}
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : v >= 10 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`);
const fmt = (v: number, unit: "x" | "ms") => (unit === "x" ? `${Math.round(v)}×` : ms(v));

/** A tooltip for one chart, placed in the chart's own box. */
function tooltip(box: HTMLElement) {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  box.append(tip);
  return {
    show(html: string, x: number, y: number) {
      tip.innerHTML = html;
      tip.hidden = false;
      const half = tip.offsetWidth / 2;
      tip.style.left = `${Math.min(Math.max(x, half), box.clientWidth - half)}px`;
      tip.style.top = `${y}px`;
    },
    hide: () => (tip.hidden = true),
  };
}

/** Redraws `draw` into `box` at its width, now and whenever the width changes. */
function responsive(box: HTMLElement, draw: (width: number) => void) {
  let last = 0;
  const run = () => {
    const w = Math.round(box.clientWidth);
    if (w && w !== last) draw((last = w));
  };
  new ResizeObserver(run).observe(box);
  run();
  return () => {
    last = 0;
    run();
  };
}

function legend(list: HTMLElement, ids: string[], toggle?: (id: string, on: boolean) => void) {
  list.replaceChildren(
    ...RUNTIMES.filter((r) => ids.includes(r.id)).map((r) => {
      const li = document.createElement("li");
      const swatch = `<i class="s-${r.id}"></i>${r.name}`;
      if (!toggle) li.innerHTML = swatch;
      else {
        const b = document.createElement("button");
        b.innerHTML = swatch;
        b.setAttribute("aria-pressed", "true");
        b.onclick = () => {
          const on = b.getAttribute("aria-pressed") !== "true";
          b.setAttribute("aria-pressed", String(on));
          toggle(r.id, on);
        };
        li.append(b);
      }
      return li;
    }),
  );
}

// ---- The first screen's chart: one metric per tab, grouped by device. ----

/** What this page measured, shown as its own group once the demo has spoken. */
const mine: Partial<Record<Metric, number>> = {};

function barChart() {
  const box = $("bars");
  const tip = tooltip(box);
  let metric = METRICS[0];
  const tabs = $("metric");
  for (const m of METRICS) {
    const b = document.createElement("button");
    b.role = "tab";
    b.textContent = m.tab;
    b.onclick = () => {
      metric = m;
      for (const t of tabs.children) t.setAttribute("aria-selected", String(t === b));
      $("metric-caption").textContent = m.caption;
      redraw();
    };
    tabs.append(b);
  }
  tabs.children[0].setAttribute("aria-selected", "true");
  $("metric-caption").textContent = metric.caption;
  legend($("bars-legend"), RUNTIMES.map((r) => r.id));

  const draw = (W: number) => {
    const groups = [...DEVICES.map((d) => ({ id: d.id, name: d.name, full: d.full, you: false }))];
    const own = mine[metric.id];
    if (own) groups.push({ id: "you", name: "This device", full: "This device, this page", you: true });
    const H = Math.round(Math.min(310, Math.max(250, W * 0.62)));
    const top = 26, bottom = 36;
    const stat = (g: string, r: string): Stat =>
      g === "you" ? (r === "ours" && own ? { median: own, p25: own, p75: own, n: 1 } : null) : (RESULTS[g]?.[r]?.[metric.id] ?? null);
    const values = groups.flatMap((g) => RUNTIMES.map((r) => stat(g.id, r.id))).filter((s): s is NonNullable<Stat> => !!s);
    const lo = Math.min(...values.map((s) => s.p25)) / 2.2;
    const hi = Math.max(...values.map((s) => s.p75)) * 1.15;
    const y = (v: number) => top + (H - top - bottom) * (1 - Math.log(v / lo) / Math.log(hi / lo));
    const base = H - bottom;
    const gap = W < 480 ? 14 : 26;
    const slots = groups.reduce((n, g) => n + (g.you ? 1 : RUNTIMES.length), 0);
    const bw = Math.min(54, (W - gap * (groups.length - 1)) / slots - 4);
    const groupW = (g: (typeof groups)[0]) => (g.you ? 1 : RUNTIMES.length) * (bw + 4) - 4;
    const total = groups.reduce((n, g) => n + groupW(g), 0) + gap * (groups.length - 1);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${metric.tab} by device` });
    let x = (W - total) / 2;
    for (const g of groups) {
      const gw = groupW(g);
      const groupMax = Math.max(...RUNTIMES.map((r) => stat(g.id, r.id)?.median ?? 0));
      const runtimes = g.you ? RUNTIMES.slice(0, 1) : RUNTIMES;
      runtimes.forEach((r, i) => {
        const s = stat(g.id, r.id);
        const bx = x + i * (bw + 4);
        const cx = bx + bw / 2;
        const failed = !s && !!RESULTS[g.id]?.[r.id]?.errors?.length;
        const hit = el("g", {}, svg);
        if (s) {
          el("rect", { class: `bar s-${r.id}`, x: bx, y: y(s.median), width: bw, height: base - y(s.median), rx: 4 }, hit);
          if (s.p75 > s.p25) {
            el("line", { class: "whisker", x1: cx, x2: cx, y1: y(s.p25), y2: y(s.p75) }, hit);
            for (const v of [s.p25, s.p75]) el("line", { class: "whisker", x1: cx - 4, x2: cx + 4, y1: y(v), y2: y(v) }, hit);
          }
          if (bw >= 34 || r.id === "ours")
            el("text", { class: "val", x: cx, y: Math.min(y(s.median), y(s.p75)) - 7, "text-anchor": "middle" }, hit).textContent = fmt(s.median, metric.unit);
        } else if (failed) {
          el("rect", { class: "dnf", x: bx + 0.6, y: y(groupMax), width: bw - 1.2, height: base - y(groupMax), rx: 4 }, hit);
        } else {
          el("text", { x: cx, y: base - 8, "text-anchor": "middle" }, hit).textContent = "–";
        }
        el("rect", { x: bx - 2, y: top - 20, width: bw + 4, height: H - top - bottom + 20, fill: "transparent" }, hit);
        const label = s
          ? `<b>${fmt(s.median, metric.unit)}</b>${s.n > 1 ? ` · ${fmt(s.p25, metric.unit)} to ${fmt(s.p75, metric.unit)}` : ""}`
          : failed
            ? "did not finish"
            : "not measured";
        const extra = g.you
          ? `<br>${metric.id === "warmRealtime" ? "your last text" : "your last run"}`
          : metric.id === "firstVisitMs" && r.mb && s
            ? `<br>${r.mb} MB download`
            : "";
        hit.onpointerenter = hit.onpointerdown = () =>
          tip.show(`${r.name} · ${g.full}<br>${label}${extra}`, cx, s ? Math.min(y(s.median), y(s.p75)) - 18 : base - 30);
        hit.onpointerleave = () => tip.hide();
      });
      el("text", { class: g.you ? "group you" : "group", x: x + gw / 2, y: H - 10, "text-anchor": "middle" }, svg).textContent = g.name;
      x += gw + gap;
    }
    el("line", { class: "axis", x1: 0, x2: W, y1: base, y2: base }, svg);
    box.querySelector("svg")?.remove();
    box.prepend(svg);
  };
  const redraw = responsive(box, draw);
  return redraw;
}

// ---- Charts in the README's sections. ----

function lengthChart(box: HTMLElement) {
  box.innerHTML = `<h4>Speed against text length</h4><p class="caption">Every warm phrase of every round. Log scales.</p><div class="tabs" role="tablist"></div><div class="chart"></div><ul class="legend"></ul>`;
  const plot = box.querySelector<HTMLElement>(".chart")!;
  const tabs = box.querySelector<HTMLElement>(".tabs")!;
  const tip = tooltip(plot);
  const hidden = new Set<string>();
  let device = DEVICES[0];
  for (const d of DEVICES) {
    const b = document.createElement("button");
    b.role = "tab";
    b.textContent = d.name;
    b.setAttribute("aria-selected", String(d === device));
    b.onclick = () => {
      device = d;
      for (const t of tabs.children) t.setAttribute("aria-selected", String(t === b));
      redraw();
    };
    tabs.append(b);
  }
  const text = Object.fromEntries(phrases.phrases.map((p) => [p.id, p.text]));
  const draw = (W: number) => {
    const H = Math.round(Math.min(360, Math.max(240, W * 0.5)));
    const L = 40, R = 10, T = 8, B = 30;
    const x = (v: number) => L + ((W - L - R) * Math.log(v / 1.5)) / Math.log(45 / 1.5);
    const y = (v: number) => T + (H - T - B) * (1 - Math.log(v / 3) / Math.log(260 / 3));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Speed against text length, ${device.full}` });
    const grid = el("g", { class: "grid" }, svg);
    for (const v of [2, 5, 10, 20, 40]) {
      el("line", { x1: x(v), x2: x(v), y1: T, y2: H - B }, grid);
      el("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, svg).textContent = `${v} s`;
    }
    for (const v of [5, 10, 20, 50, 100, 200]) {
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid);
      el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = `${v}×`;
    }
    const points: { px: number; py: number; rt: string; secs: number; speed: number; id: string }[] = [];
    for (const r of [...RUNTIMES].reverse()) {
      if (hidden.has(r.id)) continue;
      const g = el("g", { class: `s-${r.id}`, "fill-opacity": 0.6 }, svg);
      for (const [secs, speed, id] of PHRASES[device.id]?.[r.id] ?? []) {
        const px = x(secs), py = y(speed);
        el("circle", { cx: px, cy: py, r: 3.2 }, g);
        points.push({ px, py, rt: r.name, secs, speed, id });
      }
    }
    const ring = el("circle", { class: "hot", r: 5.5, fill: "none", visibility: "hidden" }, svg);
    svg.onpointermove = svg.onpointerdown = (e) => {
      const b = svg.getBoundingClientRect();
      const mx = ((e.clientX - b.left) * W) / b.width, my = ((e.clientY - b.top) * H) / b.height;
      let best = points[0], d = Infinity;
      for (const p of points) {
        const dd = (p.px - mx) ** 2 + (p.py - my) ** 2;
        if (dd < d) (d = dd), (best = p);
      }
      if (!best || d > 900) return (tip.hide(), ring.setAttribute("visibility", "hidden"));
      ring.setAttribute("cx", String(best.px));
      ring.setAttribute("cy", String(best.py));
      ring.setAttribute("visibility", "visible");
      const words = text[best.id] ?? best.id;
      tip.show(`<b>${Math.round(best.speed)}×</b> · ${best.rt}<br>${best.secs.toFixed(1)} s: “${words.length > 44 ? `${words.slice(0, 42)}…` : words}”`, (best.px / W) * plot.clientWidth, (best.py / H) * plot.clientHeight - 8);
    };
    svg.onpointerleave = () => (tip.hide(), ring.setAttribute("visibility", "hidden"));
    plot.querySelector("svg")?.remove();
    plot.prepend(svg);
  };
  const redraw = responsive(plot, draw);
  legend(box.querySelector(".legend")!, RUNTIMES.map((r) => r.id), (id, on) => {
    if (on) hidden.delete(id);
    else hidden.add(id);
    redraw();
  });
}

function timingChart(box: HTMLElement) {
  const t = TIMING;
  box.innerHTML = `<h4>Word starts against forced alignment</h4><p class="caption">${t.words} words, 4 voices. Median error ${Math.round(t.medianMs)} ms, ${Math.round(t.within100ms * 100)}% within 100 ms.</p><div class="chart"></div>`;
  const plot = box.querySelector<HTMLElement>(".chart")!;
  const tip = tooltip(plot);
  const bins = new Map<number, number>();
  for (const e of t.errorsMs) {
    const b = Math.max(-300, Math.min(280, Math.floor(e / 20) * 20));
    bins.set(b, (bins.get(b) ?? 0) + 1);
  }
  const most = Math.max(...bins.values());
  responsive(plot, (W) => {
    const H = Math.round(Math.min(240, Math.max(180, W * 0.32)));
    const L = 8, R = 8, T = 30, B = 30;
    const x = (v: number) => L + ((W - L - R) * (v + 300)) / 600;
    const y = (n: number) => T + (H - T - B) * (1 - n / most);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Histogram of word start errors" });
    for (let b = -300; b < 300; b += 20) {
      const n = bins.get(b) ?? 0;
      const g = el("g", {}, svg);
      el("rect", { class: "bar s-ours", x: x(b) + 1, y: y(n), width: x(b + 20) - x(b) - 2, height: H - B - y(n), rx: 2 }, g);
      el("rect", { x: x(b), y: T, width: x(b + 20) - x(b), height: H - T - B, fill: "transparent" }, g);
      g.onpointerenter = g.onpointerdown = () =>
        tip.show(`<b>${n} words</b><br>${b > 0 ? "+" : ""}${b} to ${b + 20 > 0 ? "+" : ""}${b + 20} ms`, (x(b + 10) / W) * plot.clientWidth, (y(n) / H) * plot.clientHeight - 8);
      g.onpointerleave = () => tip.hide();
    }
    for (const v of [-300, -200, -100, 0, 100, 200, 300])
      el("text", { x: x(v), y: H - 10, "text-anchor": v === -300 ? "start" : v === 300 ? "end" : "middle" }, svg).textContent = `${v > 0 ? "+" : ""}${v} ms`;
    const m = t.medianSignedMs;
    el("line", { class: "median", x1: x(m), x2: x(m), y1: T - 12, y2: H - B }, svg);
    el("text", { x: x(m) + 6, y: T - 16 }, svg).textContent = `median ${m > 0 ? "+" : ""}${Math.round(m)} ms, blixt/phonon later`;
    el("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }, svg);
    plot.querySelector("svg")?.remove();
    plot.prepend(svg);
  });
}

function qualityChart(box: HTMLElement) {
  const q = QUALITY;
  const clips = Math.min(...Object.values(q.clips));
  box.innerHTML = `<h4>Quality</h4><p class="caption">${clips} warm clips per runtime on the M3 Max, with 95% intervals.</p><div class="chart"></div>`;
  const plot = box.querySelector<HTMLElement>(".chart")!;
  const tip = tooltip(plot);
  const panels = [
    { key: "wer" as const, title: "Word errors (Whisper)", max: 2, f: (v: number) => `${(v * 100).toFixed(2)}%`, scale: 100 },
    { key: "utmos" as const, title: "Predicted quality (UTMOS)", max: 5, f: (v: number) => v.toFixed(2), scale: 1 },
  ];
  responsive(plot, (W) => {
    const two = W >= 560;
    const PW = two ? (W - 24) / 2 : W;
    const row = 30, labelW = Math.min(150, PW * 0.42);
    const PH = 26 + RUNTIMES.length * row;
    const H = two ? PH : PH * 2 + 16;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Word error rate and UTMOS per runtime" });
    panels.forEach((p, i) => {
      const ox = two ? i * (PW + 24) : 0, oy = two ? 0 : i * (PH + 16);
      el("text", { class: "group", x: ox, y: oy + 14 }, svg).textContent = p.title;
      const x = (v: number) => ox + labelW + ((PW - labelW - 56) * v * p.scale) / p.max;
      RUNTIMES.forEach((r, j) => {
        const [mid, a, b] = q[p.key][r.id] ?? [];
        if (mid === undefined) return;
        const yy = oy + 26 + j * row;
        const g = el("g", {}, svg);
        el("text", { x: ox, y: yy + 15 }, g).textContent = r.name;
        el("rect", { class: `bar s-${r.id}`, x: x(0), y: yy + 4, width: Math.max(2, x(mid) - x(0)), height: row - 12, rx: 3 }, g);
        el("line", { class: "whisker", x1: x(a), x2: x(b), y1: yy + row / 2 - 2, y2: yy + row / 2 - 2 }, g);
        el("text", { class: "val", x: Math.max(x(mid), x(b)) + 6, y: yy + 15 }, g).textContent = p.f(mid);
        el("rect", { x: ox, y: yy, width: PW, height: row, fill: "transparent" }, g);
        g.onpointerenter = g.onpointerdown = () =>
          tip.show(`${r.name}<br><b>${p.f(mid)}</b> · 95% interval ${p.f(a)} to ${p.f(b)}`, ((x(mid) + 0) / W) * plot.clientWidth, (yy / H) * plot.clientHeight);
        g.onpointerleave = () => tip.hide();
      });
    });
    plot.querySelector("svg")?.remove();
    plot.prepend(svg);
  });
}

function listenPlayers(box: HTMLElement) {
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Phrase");
  for (const p of phrases.phrases) select.add(new Option(p.text.length > 90 ? `${p.text.slice(0, 88)}…` : p.text, p.id));
  const players = document.createElement("div");
  players.className = "players";
  const audio = new Audio();
  let playing: HTMLButtonElement | null = null;
  const stop = () => {
    audio.pause();
    playing?.setAttribute("aria-pressed", "false");
    playing = null;
  };
  audio.onended = stop;
  const notes: Record<string, string> = { ours: "WebGPU", gradiumWebgpu: "Gradium, WebGPU", gradiumCpu: "Gradium, CPU", kokoro: "another model" };
  for (const r of RUNTIMES) {
    const b = document.createElement("button");
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<i class="s-${r.id}"></i><span>${r.name.replace(/ (WebGPU|CPU)$/, "")}<small>${notes[r.id]}</small></span>`;
    b.onclick = () => {
      const again = playing === b;
      stop();
      if (again) return;
      audio.src = `audio/${r.id}/${select.value}.mp3`;
      void audio.play();
      playing = b;
      b.setAttribute("aria-pressed", "true");
    };
    players.append(b);
  }
  select.onchange = stop;
  box.append(select, players);
}

// ---- The demo. ----

async function demo(redrawBars: () => void) {
  const button = $<HTMLButtonElement>("speak");
  const words = $("words");
  const speed = $<HTMLInputElement>("speed");
  speed.oninput = () => ($("speed-out").textContent = `${Number(speed.value).toFixed(2).replace(/0$/, "")}×`);
  const voice = $<HTMLSelectElement>("voice");
  const set = (id: string, value: string, note?: string) => {
    const s = $(id);
    s.querySelector("b")!.textContent = value;
    if (note !== undefined) s.querySelector("span")!.textContent = note;
  };
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  if (!adapter) {
    button.disabled = true;
    set("s-load", "No WebGPU", "in this browser");
    return;
  }
  const info = adapter.info;
  $("device").textContent = `GPU: ${[info.vendor, info.architecture].filter(Boolean).join(" ") || "unknown"}`;

  let tts: Phonon | null = null;
  let loading: Promise<Phonon> | null = null;
  let downloaded = false;
  /** A short untimed text in the chosen voice, as the benchmark's warm runs speak first: it
   *  fetches the voice and compiles the pipelines, so later numbers are warm ones. */
  let warm: Promise<number> = Promise.resolve(0);
  const warmUp = (p: Phonon) =>
    (warm = warm.then(async () => {
      const t0 = performance.now();
      let first = 0;
      await p.speak("Hello there.", { voice: voice.value, onAudio: () => (first ||= performance.now() - t0) }).catch(() => {});
      return first;
    }));
  const load = () =>
    (loading ??= (async () => {
      const t0 = performance.now();
      set("s-load", "…", "loading");
      const p = await Phonon.load({
        progress: (percent) => {
          downloaded = true;
          set("s-load", `${percent}%`, "of 43.7 MB");
        },
      });
      const loaded = performance.now() - t0;
      voice.replaceChildren(...p.voices.map((v) => new Option(v.name, v.name)));
      voice.value = "Delia";
      // As the benchmark measures it: load the runtime, then the first audio of a text.
      const start = loaded + (await warmUp(p));
      mine[downloaded ? "firstVisitMs" : "coldStartMs"] = start;
      set("s-load", ms(start), downloaded ? "first visit" : "cold start");
      redrawBars();
      tts = p;
      Object.assign(globalThis, { phonon: p });
      return p;
    })().catch((error) => {
      loading = null;
      set("s-load", "Failed", String(error?.message ?? error));
      throw error;
    }));
  voice.onchange = () => tts && void warmUp(tts);
  // With the model in the cache, loading costs no download: start at once.
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  if (await cache?.match(WEIGHTS_URL)) void load().catch(() => {});

  let ctx: AudioContext | null = null;
  let sources: AudioBufferSourceNode[] = [];
  let abort: AbortController | null = null;
  let last: Speech | null = null;
  let frame = 0;
  const silence = () => {
    for (const s of sources) s.stop();
    sources = [];
    cancelAnimationFrame(frame);
    for (const s of words.children) s.classList.remove("now");
  };
  const play = (chunk: Float32Array, at: number) => {
    const buffer = ctx!.createBuffer(1, chunk.length, 24000);
    buffer.copyToChannel(chunk as Float32Array<ArrayBuffer>, 0);
    const s = ctx!.createBufferSource();
    s.buffer = buffer;
    s.connect(ctx!.destination);
    s.start(at);
    sources.push(s);
  };
  /** Lights each word while it is heard, from audio time `start` on the context's clock. */
  const follow = (speech: Speech, start: number) => {
    const spans = [...words.children];
    const tick = () => {
      const t = ctx!.currentTime - start;
      spans.forEach((s, i) => s.classList.toggle("now", t >= speech.words[i].start && t < speech.words[i].end));
      if (t < speech.audio.length / 24000) frame = requestAnimationFrame(tick);
      else done();
    };
    frame = requestAnimationFrame(tick);
  };
  const done = () => {
    button.textContent = "Speak";
    abort = null;
    sources = [];
  };
  const showWords = (speech: Speech) => {
    words.classList.remove("empty");
    words.replaceChildren(
      ...speech.words.flatMap((w) => {
        const s = document.createElement("span");
        s.textContent = w.text;
        s.onclick = () => {
          if (!ctx || last !== speech) return;
          silence();
          const offset = w.start;
          const at = ctx.currentTime + 0.03;
          const src = ctx.createBufferSource();
          const buffer = ctx.createBuffer(1, speech.audio.length, 24000);
          buffer.copyToChannel(speech.audio as Float32Array<ArrayBuffer>, 0);
          src.buffer = buffer;
          src.connect(ctx.destination);
          src.start(at, offset);
          sources = [src];
          button.textContent = "Stop";
          follow(speech, at - offset);
        };
        return [s, document.createTextNode(" ")];
      }),
    );
  };

  button.onclick = async () => {
    if (abort || sources.length) {
      abort?.abort();
      silence();
      done();
      return;
    }
    ctx ??= new AudioContext({ sampleRate: 24000 });
    void ctx.resume();
    abort = new AbortController();
    button.textContent = "Stop";
    const signal = abort.signal;
    try {
      const p = tts ?? (await load());
      await warm;
      if (signal.aborted) return;
      const t0 = performance.now();
      let first = 0;
      let made = 0;
      let at = 0;
      set("s-first", "…");
      const live = () => {
        const took = performance.now() - t0;
        set("s-speed", `${Math.round(made / (took / 1000))}×`);
        set("s-audio", `${made.toFixed(1)} s`, `in ${ms(took)}`);
      };
      $("s-speed").classList.add("live");
      const speech = await p.speak($<HTMLTextAreaElement>("text").value, {
        voice: voice.value,
        speed: Number(speed.value),
        signal,
        onAudio: (chunk) => {
          if (!first) {
            first = performance.now() - t0;
            set("s-first", ms(first));
            at = ctx!.currentTime + 0.05;
          }
          play(chunk, at);
          at += chunk.length / 24000;
          made += chunk.length / 24000;
          live();
        },
      });
      $("s-speed").classList.remove("live");
      live();
      last = speech;
      mine.warmRealtime = made / ((performance.now() - t0) / 1000);
      mine.warmFirstMs = first;
      redrawBars();
      showWords(speech);
      follow(speech, at - speech.audio.length / 24000);
    } catch (error) {
      if (!signal.aborted) set("s-first", "Failed", String((error as Error)?.message ?? error));
      silence();
      done();
    }
  };
}

const redrawBars = barChart();
for (const box of document.querySelectorAll<HTMLElement>("[data-chart]")) {
  const draw = { length: lengthChart, timing: timingChart, quality: qualityChart }[box.dataset.chart!];
  draw?.(box);
}
const listen = document.getElementById("listen-players");
if (listen) listenPlayers(listen);
void demo(redrawBars);
