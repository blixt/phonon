// The web page's script: it draws the charts into the boxes the build wrote (frames.ts), and
// runs the demo with its live numbers. Nothing here changes a box's size.
import { Phonon, type Speech, type Voice, type Word } from "../src/index";
import { MODEL_CACHE, MODELS, type Language, type ModelId } from "../src/files";
import { DEVICES, METRICS, PHRASES, QUALITY, RESULTS, RUNTIMES, TEXTS, TIMING, fmt, ms, short, type Metric, type Stat, downloadMb } from "./data";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SVG = "http://www.w3.org/2000/svg";
function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.append(e);
  return e;
}

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

/** Redraws `draw` into `box` at the size its CSS gives it, now and whenever that changes. */
function responsive(box: HTMLElement, draw: (width: number, height: number) => void) {
  let last = "";
  const run = (force = false) => {
    const w = Math.round(box.clientWidth), h = Math.round(box.clientHeight);
    if (w && h && (force || `${w}x${h}` !== last)) {
      last = `${w}x${h}`;
      box.querySelector("svg")?.remove();
      draw(w, h);
    }
  };
  new ResizeObserver(() => run()).observe(box);
  run();
  return () => run(true);
}

/** Tabs and their captions from frames.ts: selecting a tab shows its caption. */
function wireTabs(box: HTMLElement, select: (i: number) => void) {
  const buttons = [...box.querySelectorAll<HTMLButtonElement>(".tabs button")];
  const captions = box.querySelectorAll(".captions .caption");
  buttons.forEach((b, i) => {
    b.onclick = () => {
      buttons.forEach((t) => t.setAttribute("aria-selected", String(t === b)));
      if (captions.length > 1) captions.forEach((c, j) => c.classList.toggle("off", j !== i));
      select(i);
    };
  });
}

/** Value labels that fit: each stays inside the chart, and one that would touch another label
 *  or another bar is left out (its bar's tooltip still has it). blixt/phonon's labels always
 *  stay, on top of everything, with a halo where they cross a bar. */
function fitLabels(svg: SVGSVGElement, W: number) {
  const touches = (a: DOMRect, b: DOMRect) => a.x < b.x + b.width + 1 && b.x < a.x + a.width + 1 && a.y < b.y + b.height && b.y < a.y + a.height;
  const bars = [...svg.querySelectorAll<SVGRectElement>("rect.bar, rect.dnf")].map((r) => ({ g: r.parentElement, b: r.getBBox() }));
  const labels = [...svg.querySelectorAll<SVGTextElement>("text.val")];
  const ours = (t: SVGTextElement) => !!t.parentElement!.querySelector(".s-ours");
  labels.sort((a, b) => Number(ours(b)) - Number(ours(a)));
  const placed: DOMRect[] = [];
  for (const t of labels) {
    let b = t.getBBox();
    const shift = b.x < 0 ? -b.x : b.x + b.width > W ? W - b.x - b.width : 0;
    if (shift) {
      t.setAttribute("x", String(Number(t.getAttribute("x")) + shift));
      b = t.getBBox();
    }
    if (ours(t)) svg.append(t);
    else if (placed.some((p) => touches(b, p)) || bars.some((r) => r.g !== t.parentElement && touches(b, r.b))) {
      t.remove();
      continue;
    }
    placed.push(b);
  }
}

// ---- The first screen's chart: one metric per tab, grouped by device. ----

/** What this page measured, shown as its own group once the demo has spoken. */
const mine: Partial<Record<Metric, number>> = {};

function barChart(frame: HTMLElement) {
  const box = frame.querySelector<HTMLElement>("[data-plot]")!;
  const tip = tooltip(box);
  let metric = METRICS[0];
  wireTabs(frame, (i) => {
    metric = METRICS[i];
    redraw();
  });

  const draw = (W: number, H: number) => {
    const groups = [...DEVICES.map((d) => ({ id: d.id, name: d.name, short: d.short, full: d.full, you: false }))];
    const own = mine[metric.id];
    if (own) groups.push({ id: "you", name: "This device", short: "This device", full: "This device", you: true });
    const top = 26, bottom = 36;
    const stat = (g: string, r: string): Stat =>
      g === "you" ? (r === "ours" && own ? { median: own, p25: own, p75: own, n: 1 } : null) : (RESULTS[g]?.[r]?.[metric.id] ?? null);
    const values = groups.flatMap((g) => RUNTIMES.map((r) => stat(g.id, r.id))).filter((s): s is NonNullable<Stat> => !!s);
    // A linear scale from zero: each tab is one measure, so bar heights compare directly.
    const hi = Math.max(...values.map((s) => s.p75)) * 1.04;
    const base = H - bottom;
    const y = (v: number) => Math.min(base - 2, top + (H - top - bottom) * (1 - v / hi));
    const gap = W < 480 ? 14 : 26;
    // This device's group is one bar, but as wide as its label.
    const youW = own ? 80 : 0;
    const devices = groups.length - (own ? 1 : 0);
    const bw = Math.min(54, (W - youW - gap * (groups.length - 1)) / (devices * RUNTIMES.length) - 4);
    const groupW = (g: (typeof groups)[0]) => (g.you ? youW : RUNTIMES.length * (bw + 4) - 4);
    const total = groups.reduce((n, g) => n + groupW(g), 0) + gap * (groups.length - 1);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${metric.tab} by device` });
    let x = (W - total) / 2;
    for (const g of groups) {
      const gw = groupW(g);
      const groupMax = Math.max(...RUNTIMES.map((r) => stat(g.id, r.id)?.median ?? 0));
      const runtimes = g.you ? RUNTIMES.slice(0, 1) : RUNTIMES;
      runtimes.forEach((r, i) => {
        const s = stat(g.id, r.id);
        const bx = g.you ? x + youW - bw : x + i * (bw + 4);
        const cx = bx + bw / 2;
        const failed = !s && !!RESULTS[g.id]?.[r.id]?.errors?.length;
        const hit = el("g", {}, svg);
        if (s) {
          el("rect", { class: `bar s-${r.id}`, x: bx, y: y(s.median), width: bw, height: base - y(s.median), rx: 4 }, hit);
          if (s.p75 > s.p25) {
            el("line", { class: "whisker", x1: cx, x2: cx, y1: y(s.p25), y2: y(s.p75) }, hit);
            for (const v of [s.p25, s.p75]) el("line", { class: "whisker", x1: cx - 4, x2: cx + 4, y1: y(v), y2: y(v) }, hit);
          }
          el("text", { class: "val", x: cx, y: Math.min(y(s.median), y(s.p75)) - 7, "text-anchor": "middle" }, hit).textContent = short(s.median, metric.unit);
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
        const r0 = RESULTS[g.id]?.[r.id];
        const extra = g.you
          ? "<br>your last run"
          : metric.id === "firstVisitMs" && r0?.firstVisitMeasuredMs && r0.firstVisitNetworkMBps
            ? `<br>measured ${ms(r0.firstVisitMeasuredMs.median)} at ${r0.firstVisitNetworkMBps.median.toFixed(1)} MB/s`
            : "";
        hit.onpointerenter = hit.onpointerdown = () =>
          tip.show(`${r.name} · ${g.full}<br>${label}${extra}`, cx, s ? Math.min(y(s.median), y(s.p75)) - 18 : base - 30);
        hit.onpointerleave = () => tip.hide();
      });
      el("text", { class: g.you ? "group you" : "group", x: g.you ? x + gw : x + gw / 2, y: H - 10, "text-anchor": g.you ? "end" : "middle" }, svg).textContent = gw < 92 ? g.short : g.name;
      x += gw + gap;
    }
    el("line", { class: "axis", x1: 0, x2: W, y1: base, y2: base }, svg);
    box.prepend(svg);
    fitLabels(svg, W);
  };
  const redraw = responsive(box, draw);
  return redraw;
}

// ---- Charts in the README's sections. ----

function lengthChart(frame: HTMLElement) {
  const plot = frame.querySelector<HTMLElement>("[data-plot]")!;
  const tip = tooltip(plot);
  const hidden = new Set<string>();
  let device = DEVICES[0];
  wireTabs(frame, (i) => {
    device = DEVICES[i];
    redraw();
  });
  for (const b of frame.querySelectorAll<HTMLButtonElement>(".legend button")) {
    b.onclick = () => {
      const on = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", String(on));
      if (on) hidden.delete(b.dataset.runtime!);
      else hidden.add(b.dataset.runtime!);
      redraw();
    };
  }
  const text = Object.fromEntries(TEXTS.map((p) => [p.id, p.text]));
  const draw = (W: number, H: number) => {
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
    plot.prepend(svg);
  };
  const redraw = responsive(plot, draw);
}

function timingChart(frame: HTMLElement) {
  const t = TIMING;
  const plot = frame.querySelector<HTMLElement>("[data-plot]")!;
  const tip = tooltip(plot);
  const bins = new Map<number, number>();
  for (const e of t.errorsMs) {
    const b = Math.max(-300, Math.min(280, Math.floor(e / 20) * 20));
    bins.set(b, (bins.get(b) ?? 0) + 1);
  }
  const most = Math.max(...bins.values());
  responsive(plot, (W, H) => {
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
    for (const v of W < 520 ? [-200, 0, 200] : [-300, -200, -100, 0, 100, 200, 300])
      el("text", { x: x(v), y: H - 10, "text-anchor": v === -300 ? "start" : v === 300 ? "end" : "middle" }, svg).textContent = `${v > 0 ? "+" : ""}${v} ms`;
    const m = t.medianSignedMs;
    el("line", { class: "median", x1: x(m), x2: x(m), y1: T - 12, y2: H - B }, svg);
    el("text", { x: x(m) + 6, y: T - 16 }, svg).textContent = `median ${m > 0 ? "+" : ""}${Math.round(m)} ms (later)`;
    el("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }, svg);
    plot.prepend(svg);
  });
}

function qualityChart(frame: HTMLElement) {
  const q = QUALITY;
  const plot = frame.querySelector<HTMLElement>("[data-plot]")!;
  const tip = tooltip(plot);
  const panels = [
    { key: "wer" as const, title: "Word errors (Whisper)", max: 2, f: (v: number) => `${(v * 100).toFixed(2)}%`, scale: 100 },
    { key: "utmos" as const, title: "Predicted quality (UTMOS)", max: 5, f: (v: number) => v.toFixed(2), scale: 1 },
  ];
  responsive(plot, (W, H) => {
    // Two panels side by side from 560 px, as the CSS sizes the box.
    const two = W >= 560;
    const PW = two ? (W - 24) / 2 : W;
    const row = 30, labelW = Math.min(150, PW * 0.42);
    const PH = 26 + RUNTIMES.length * row;
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
    plot.prepend(svg);
  });
}

function listenPlayers(box: HTMLElement) {
  const select = box.querySelector("select")!;
  const audio = new Audio();
  let playing: HTMLButtonElement | null = null;
  const stop = () => {
    audio.pause();
    playing?.setAttribute("aria-pressed", "false");
    playing = null;
  };
  audio.onended = stop;
  for (const b of box.querySelectorAll<HTMLButtonElement>(".players button")) {
    b.onclick = () => {
      const again = playing === b;
      stop();
      if (again) return;
      audio.src = `audio/${b.dataset.runtime}/${select.value}.mp3`;
      void audio.play();
      playing = b;
      b.setAttribute("aria-pressed", "true");
    };
  }
  select.onchange = stop;
}

// ---- The demo. ----

/** What a bug report needs: the page, the browser, the GPU and its limits, the model and the last error. */
function details(adapter: GPUAdapter | null, model: string, error: string | null): string {
  const info = adapter?.info as (GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number }) | undefined;
  const limits = adapter
    ? (["maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup"] as const)
        .map((k) => `${k} ${adapter.limits[k]}`)
        .join(", ")
    : "–";
  const stat = (id: string) => document.getElementById(id)?.innerText.replace(/\s+/g, " ") ?? "";
  return [
    "blixt/phonon details",
    `page: ${location.href}`,
    `browser: ${navigator.userAgent}`,
    `WebGPU: ${"gpu" in navigator ? "yes" : "no"}`,
    `adapter: ${info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ") || "unnamed" : "none"}`,
    `subgroups: ${adapter?.features.has("subgroups") ? `${info?.subgroupMinSize}-${info?.subgroupMaxSize}` : "no"}`,
    `features: ${adapter ? [...adapter.features].sort().join(", ") : "–"}`,
    `limits: ${limits}`,
    `model: ${model}`,
    `stats: ${["s-load", "s-first", "s-speed", "s-audio"].map(stat).join(" | ")}`,
    `error: ${error ?? "none"}`,
  ].join("\n");
}

/** Each language's first voice (outside English, the one UTMOS rated highest with no word errors
 *  in its 40M model) and sample text: a sample still in the box follows the voice's language. */
const LANGUAGES: Record<Language, { name: string; voice: string; sample: string }> = {
  en: { name: "English", voice: "Delia", sample: "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background." },
  fr: { name: "French", voice: "Augustin", sample: "Le petit chat dort au soleil près de la fenêtre. Demain, nous irons au marché acheter du pain frais." },
  de: { name: "German", voice: "Mats", sample: "Der kleine Hund spielt im Garten mit einem roten Ball. Morgen fahren wir mit dem Zug nach Berlin." },
  es: { name: "Spanish", voice: "Marcos", sample: "El niño camina por la playa al atardecer. Mañana vamos a visitar a mis abuelos en el pueblo." },
  pt: { name: "Portuguese", voice: "Davi", sample: "O menino corre pelo parque com o seu cachorro. Amanhã vamos viajar para o Rio de Janeiro." },
};
/** The model the benchmark measured: only its numbers join the charts. */
const BENCHMARKED: ModelId = "phonon-31m-en";

/** The voice menu: one group per language when the model has several. */
function voiceOptions(voices: readonly Voice[], languages: readonly Language[]): HTMLElement[] {
  const option = (v: Voice) => new Option(v.name, v.name);
  if (languages.length === 1) return voices.map(option);
  return languages.map((l) => {
    const group = document.createElement("optgroup");
    group.label = LANGUAGES[l].name;
    group.append(...voices.filter((v) => v.language === l && v.locale !== "characters").map(option));
    return group;
  });
}

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
  const model = $<HTMLSelectElement>("model");
  const id = () => model.value as ModelId;
  /** The GPU line: what runs here, or why nothing does. */
  const note = (text: string) => ($("device-text").textContent = text);
  /** Why the demo cannot speak, over the text box, until the text is edited. */
  const explain = (why: string) => {
    const box = $("words");
    box.replaceChildren(Object.assign(document.createElement("span"), { className: "why", textContent: why }));
    box.classList.add("on");
  };
  let lastError: string | null = null;
  const copy = $<HTMLButtonElement>("copy-details");
  copy.onclick = async () => {
    const ok = await navigator.clipboard.writeText(details(adapter ?? null, id(), lastError)).then(() => true, () => false);
    copy.textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => (copy.textContent = "Copy details"), 2000);
  };
  if (!adapter) {
    button.disabled = true;
    set("s-load", "–", "no WebGPU here");
    note("No WebGPU");
    explain(
      gpu
        ? "This browser has WebGPU, but it gave no GPU to use."
        : /Android/.test(navigator.userAgent) && /Chrome\//.test(navigator.userAgent)
          ? "This browser has no WebGPU. Chrome on Android turns WebGPU off under Advanced Protection and on some GPUs. Firefox with WebGPU on can work."
          : "This browser has no WebGPU, which the demo needs.",
    );
    return;
  }
  const info = adapter.info;
  const gpuName = `GPU: ${[info.vendor, info.architecture].filter(Boolean).join(" ") || "unknown"}`;
  const text = $<HTMLTextAreaElement>("text");
  /** Stops what is speaking (set once the player below exists). */
  let stopAll = () => {};

  /** Each model once loaded (or loading), and the one the controls speak with. */
  const loads = new Map<ModelId, Promise<Phonon>>();
  let tts: Phonon | null = null;
  const language = () => tts?.voices.find((v) => v.name === voice.value)?.language ?? "en";
  /** Shows the chosen model's state: its download before it loads, then its load time. */
  const describe = (start?: { ms: number; downloaded: boolean }) => {
    const m = MODELS[id()];
    note(m.license === "Apache-2.0" ? gpuName : `${gpuName} · CC BY-NC 4.0, non-commercial`);
    if (start) set("s-load", ms(start.ms), start.downloaded ? "first visit" : "cold start");
    else set("s-load", `${downloadMb(id())} MB`, "download");
  };
  describe();
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
  /** Puts a model's voices in the menu, keeping the chosen voice when the model has it. */
  /** A sample text still in the box follows the chosen voice's language. */
  const followLanguage = () => {
    const samples = Object.values(LANGUAGES).map((l) => l.sample);
    if (!samples.includes(text.value.trim())) return;
    text.value = LANGUAGES[language()].sample;
    words.classList.remove("on");
  };
  const showVoices = (p: Phonon) => {
    const keep = voice.value;
    voice.replaceChildren(...voiceOptions(p.voices, MODELS[id()].languages));
    // The chosen voice when the model has it, else the model's first language's voice.
    voice.value = p.voices.some((v) => v.name === keep) ? keep : LANGUAGES[MODELS[id()].languages[0]].voice;
    followLanguage();
  };
  const load = (which = id()) => {
    const known = loads.get(which);
    if (known) return known;
    const loading = (async () => {
      const t0 = performance.now();
      let downloaded = false;
      if (which === id()) set("s-load", "…", "loading");
      const p = await Phonon.load({
        model: which,
        progress: (percent) => {
          downloaded = true;
          if (which === id()) set("s-load", `${percent}%`, "of the model");
        },
      });
      const loaded = performance.now() - t0;
      if (which !== id()) return p;
      tts = p;
      showVoices(p);
      // As the benchmark measures it: load the runtime, then the first audio of a text.
      const start = loaded + (await warmUp(p));
      // A first visit here depends on this connection, and the chart's are at 50 Mbit/s: only a
      // cold start of the benchmarked model joins the chart.
      if (!downloaded && which === BENCHMARKED) mine.coldStartMs = start;
      if (which === id()) describe({ ms: start, downloaded });
      redrawBars();
      Object.assign(globalThis, { phonon: p });
      return p;
    })().catch((error) => {
      loads.delete(which);
      lastError = `load ${which}: ${error?.stack ?? error}`;
      if (which === id()) {
        set("s-load", "Failed", "to load");
        note("Loading failed");
        explain(`Loading failed: ${error?.message ?? error}`);
      }
      throw error;
    });
    loads.set(which, loading);
    return loading;
  };
  voice.onchange = () => {
    followLanguage();
    if (tts) void warmUp(tts);
  };
  model.onchange = async () => {
    const chosen = id();
    stopAll();
    tts = null;
    describe();
    const p = await load().catch(() => null);
    if (p && chosen === id()) {
      tts = p;
      showVoices(p);
      void warmUp(p);
    }
  };
  // With the model in the cache, loading costs no download: start at once.
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  if (await cache?.match(MODELS[id()].weights)) void load().catch(() => {});

  let ctx: AudioContext | null = null;
  let sources: AudioBufferSourceNode[] = [];
  let abort: AbortController | null = null;
  let last: Speech | null = null;
  let frame = 0;
  /** The words shown, the times known so far (they arrive while the audio is made), and when
   *  the audio ends (known once it is all made). */
  let timed: Word[] = [];
  let audioEnd: number | null = null;
  const wordSpans = () => [...words.children] as HTMLElement[];
  const silence = () => {
    for (const s of sources) s.stop();
    sources = [];
    cancelAnimationFrame(frame);
    for (const s of words.querySelectorAll(".lit")) s.classList.remove("lit");
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
  /** Lights each word while it is heard, token by token, from audio time `start` on the
   *  context's clock, and keeps it in view when the text is longer than the box. */
  const follow = (start: number) => {
    cancelAnimationFrame(frame);
    const spans = wordSpans();
    const tick = () => {
      const t = ctx!.currentTime - start;
      timed.forEach((w, i) => {
        const s = spans[i];
        const now = t >= w.start && t < w.end;
        if (now && !s.firstElementChild!.classList.contains("lit") && (s.offsetTop < words.scrollTop || s.offsetTop + s.offsetHeight > words.scrollTop + words.clientHeight))
          words.scrollTop = s.offsetTop - 10;
        w.tokens.forEach((tok, k) => s.children[k]?.classList.toggle("lit", now && t >= tok.start));
      });
      if (audioEnd === null || t < audioEnd) frame = requestAnimationFrame(tick);
      else done();
    };
    frame = requestAnimationFrame(tick);
  };
  const done = () => {
    button.textContent = "Speak";
    abort = null;
    sources = [];
  };
  stopAll = () => {
    abort?.abort();
    silence();
    done();
    words.classList.remove("on");
  };
  // Editing the text puts the words away: they belong to the text that was spoken.
  text.onfocus = () => words.classList.remove("on");
  /** The words over the text, each made of its tokens, before their times are known. */
  const showWords = (list: { text: string; tokens: string[] }[]) => {
    words.classList.add("on");
    words.scrollTop = 0;
    words.replaceChildren(
      ...list.flatMap((w, i) => {
        const s = document.createElement("span");
        s.className = "w";
        s.append(...w.tokens.map((t) => Object.assign(document.createElement("span"), { textContent: t })));
        s.onclick = () => {
          const speech = last;
          if (!ctx || !speech || speech.words.length !== list.length) return;
          silence();
          const offset = speech.words[i].start;
          const at = ctx.currentTime + 0.03;
          const src = ctx.createBufferSource();
          const buffer = ctx.createBuffer(1, speech.audio.length, 24000);
          buffer.copyToChannel(speech.audio as Float32Array<ArrayBuffer>, 0);
          src.buffer = buffer;
          src.connect(ctx.destination);
          src.start(at, offset);
          sources = [src];
          button.textContent = "Stop";
          timed = speech.words;
          audioEnd = speech.audio.length / 24000;
          follow(at - offset);
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
      last = null;
      timed = [];
      audioEnd = null;
      showWords(p.words(text.value));
      const t0 = performance.now();
      let first = 0;
      let made = 0;
      let at = 0;
      set("s-first", "…");
      const live = () => {
        const took = performance.now() - t0;
        set("s-speed", `${Math.round(made / (took / 1000))}×`);
        set("s-audio", `${made.toFixed(1)} s`);
      };
      $("s-speed").classList.add("live");
      const speech = await p.speak(text.value, {
        voice: voice.value,
        speed: Number(speed.value),
        signal,
        // Each word lights from its first token as soon as its times arrive, before it is heard.
        onWord: (w) => timed.push(w),
        onAudio: (chunk) => {
          if (!first) {
            first = performance.now() - t0;
            set("s-first", ms(first));
            at = ctx!.currentTime + 0.05;
            follow(at);
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
      audioEnd = speech.audio.length / 24000;
      if (id() === BENCHMARKED) {
        mine.warmRealtime = made / ((performance.now() - t0) / 1000);
        mine.warmFirstMs = first;
      }
      redrawBars();
    } catch (error) {
      if (!signal.aborted) {
        lastError = `speak ${id()}: ${(error as Error)?.stack ?? error}`;
        set("s-first", "Failed", "to speak");
        note("Speaking failed");
        explain(`Speaking failed: ${(error as Error)?.message ?? error}`);
      }
      silence();
      done();
    }
  };
}

const redrawBars = barChart($("bars"));
const charts: Record<string, (frame: HTMLElement) => void> = { length: lengthChart, timing: timingChart, quality: qualityChart };
for (const frame of document.querySelectorAll<HTMLElement>("[data-chart]")) charts[frame.dataset.chart!]?.(frame);
const listen = document.getElementById("listen-players");
if (listen) listenPlayers(listen);
void demo(redrawBars);
