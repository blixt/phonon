// The charts' and players' markup without their drawings: tabs, captions, an empty chart box
// and legends. The build writes it into the page (scripts/build-site.ts), so every size is set
// before the script runs, and the script (site.ts) only draws into the boxes and wires them up.
import { DEVICES, METRICS, RUNTIMES, TEXTS, TIMING } from "./data";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

/** Tabs whose selected one is bold: each keeps the width of its bold text. */
function tabs(labels: string[]): string {
  return `<div class="tabs" role="tablist">${labels
    .map((l, i) => `<button role="tab" aria-selected="${i === 0}" data-label="${esc(l)}">${esc(l)}</button>`)
    .join("")}</div>`;
}

/** Captions in one cell, so it is as tall as the longest at any width. */
function captions(texts: string[]): string {
  return `<div class="captions">${texts.map((t, i) => `<p class="caption${i ? " off" : ""}">${esc(t)}</p>`).join("")}</div>`;
}

function legend(toggles: boolean): string {
  return `<ul class="legend">${RUNTIMES.map((r) => {
    const swatch = `<i class="s-${r.id}"></i>${esc(r.name)}`;
    return `<li>${toggles ? `<button aria-pressed="true" data-runtime="${r.id}">${swatch}</button>` : swatch}</li>`;
  }).join("")}</ul>`;
}

export const frames = {
  bars: () =>
    `${tabs(METRICS.map((m) => m.tab))}${captions(METRICS.map((m) => m.caption))}<div class="chart" data-plot></div>${legend(false)}` +
    `<p class="note">Medians of 10 rounds per device. Whiskers: 25th to 75th percentile. <a href="#results">All results</a></p>`,
  length: () =>
    `${captions(["Each dot is one warm phrase. Log scales."])}${tabs(DEVICES.map((d) => d.name))}<div class="chart" data-plot></div>${legend(true)}`,
  timing: () =>
    captions([
      `${TIMING.words} words, 4 voices. Median absolute error ${Math.round(TIMING.medianMs)} ms, ${Math.round(TIMING.within100ms * 100)}% within 100 ms.`,
    ]) + `<div class="chart" data-plot></div>`,
  quality: () => `${captions(["Lines show 95% intervals."])}<div class="chart" data-plot></div>`,
  listen: () =>
    `<select aria-label="Phrase">${TEXTS.map((p) => `<option value="${p.id}">${esc(p.text.length > 90 ? `${p.text.slice(0, 88)}…` : p.text)}</option>`).join("")}</select>` +
    `<div class="players">${RUNTIMES.map(
      (r) => `<button aria-pressed="false" data-runtime="${r.id}"><i class="s-${r.id}"></i><span>${esc(r.short)}<small>${esc(r.note)}</small></span></button>`,
    ).join("")}</div>`,
};
