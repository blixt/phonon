# Figures and a summary from the benchmark results: bench/results/*.jsonl -> docs/figures/*.svg
# and docs/results.json. Usage: python bench/plots.py   (needs matplotlib, numpy)
import glob, json, os, numpy as np
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)
RUNTIMES = [("ours", "blixt/phonon (WebGPU)"), ("gradiumWebgpu", "phonon-tts (WebGPU)"), ("gradiumCpu", "phonon-tts (CPU, Wasm)"), ("kokoro", "Kokoro 82M (WebGPU)")]
COLORS = {"ours": "#d55e00", "gradiumWebgpu": "#0072b2", "gradiumCpu": "#56b4e9", "kokoro": "#999999"}
DEVICES = {"mac": "MacBook Pro, M3 Max", "pc": "PC, RTX 2080 Ti", "iphone": "iPhone 17 Pro, Safari 27.2"}
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True, "svg.fonttype": "none", "figure.dpi": 100})

data, hellos = {}, {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "*.jsonl"))):
    dev = os.path.basename(f)[:-6]
    # A runtime run on its own on a device (DEVICE-RUNTIME.jsonl) joins that device's results.
    only = None
    if "-" in dev: dev, only = dev.split("-", 1)
    if dev not in DEVICES: continue
    for line in open(f):
        r = json.loads(line)
        if "hello" in r:
            if not only: hellos[dev] = r["hello"]
            continue
        if "result" not in r: continue
        t, res = r["task"], r["result"]
        if only and t["runtime"].lower() != only: continue
        d = data.setdefault(dev, {}).setdefault(t["runtime"], {"cold": [], "warm": [], "first": [], "errors": []})
        if res.get("error"): d["errors"].append(res["error"][:300]); continue
        # First visits are their own tasks. A cold run 0 only fills the caches after them.
        if t["kind"] == "first": d["first"].append(res)
        elif t["run"] > 0: d[t["kind"]].append(res)
devices = [d for d in DEVICES if d in data]
# A runtime with no warm run on a device did not finish there: its few other runs are no median.
for dev in devices:
    for rt, d in data[dev].items():
        if not d["warm"]:
            data[dev][rt] = {"cold": [], "warm": [], "first": [], "errors": d["errors"] or ["did not finish"]}

# A first visit is mostly its download, so its time depends on the connection more than on the
# runtime. Each one is compared at one reference speed: the time any download was running is
# replaced by its bytes at REFERENCE_MBIT. Its bytes are what came from other origins (as sent on
# the network) and each runtime's own code at its gzip size, as a static host would serve it.
REFERENCE_MBIT = 50
_gzip = {}
def code_bytes(paths):
    import gzip
    for p in paths:
        if p not in _gzip:
            f = os.path.join(HERE, "..", p.lstrip("/"))
            _gzip[p] = len(gzip.compress(open(f, "rb").read(), 9)) if os.path.exists(f) else 0
    return sum(_gzip[p] for p in paths)
def first_visit(r):
    d = r["downloads"]
    total = d["wireBytes"] + code_bytes(d["code"])
    return {"ms": r["loadMs"] + r["phrases"][0]["firstMs"] - d["ms"] + total * 8 / (REFERENCE_MBIT * 1e6) * 1000, "measuredMs": r["loadMs"] + r["phrases"][0]["firstMs"], "bytes": total}

def warm_phrases(dev, rt):
    return [p for r in data[dev].get(rt, {}).get("warm", []) for p in r["phrases"]]
def stat(values):
    v = np.array(values, float)
    return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "n": int(len(v))} if len(v) else None

summary = {"devices": {d: {"label": DEVICES[d], "hello": hellos.get(d)} for d in devices}, "results": {}}
for dev in devices:
    for rt, _ in RUNTIMES:
        d = data[dev].get(rt)
        if not d: continue
        wp = warm_phrases(dev, rt)
        summary["results"].setdefault(dev, {})[rt] = {
            "info": (d["warm"] or d["cold"] or [{}])[0].get("info"),
            "errors": d["errors"],
            "warmFirstMs": stat([p["firstMs"] for p in wp]),
            "warmRealtime": stat([p["audioSecs"] / (p["totalMs"] / 1000) for p in wp]),
            "warmTotalSecs": stat([sum(p["totalMs"] for p in r["phrases"]) / 1000 for r in d["warm"]]),
            "coldLoadMs": stat([r["loadMs"] for r in d["cold"]]),
            "coldFirstMs": stat([r["phrases"][0]["firstMs"] for r in d["cold"]]),
            "coldStartMs": stat([r["loadMs"] + r["phrases"][0]["firstMs"] for r in d["cold"]]),
            "coldPageMs": stat([r["pageMs"] for r in d["cold"]]),
            "firstVisitMs": stat([first_visit(r)["ms"] for r in d["first"] if r.get("downloads")]),
            "firstVisitMeasuredMs": stat([first_visit(r)["measuredMs"] for r in d["first"] if r.get("downloads")]),
            "firstVisitMB": stat([first_visit(r)["bytes"] / 1e6 for r in d["first"] if r.get("downloads")]),
            "firstVisitNetworkMBps": stat([r["networkMBps"] for r in d["first"] if r.get("networkMBps")]),
        }
# What the web page's charts draw beyond the medians: each warm phrase (seconds of speech,
# × real time), the word timing errors and the quality scores.
summary["phrases"] = {dev: {rt: [[round(p["audioSecs"], 2), round(p["audioSecs"] / (p["totalMs"] / 1000), 1), p["id"]] for p in warm_phrases(dev, rt)] for rt, _ in RUNTIMES if warm_phrases(dev, rt)} for dev in devices}
for name, key in [("timing-summary.json", "timing"), ("quality-mac.json", "quality")]:
    f = os.path.join(HERE, "results", name)
    if os.path.exists(f): summary[key] = {k: v for k, v in json.load(open(f)).items() if k != "heard"}
json.dump(summary, open(os.path.join(HERE, "..", "docs", "results.json"), "w"), separators=(",", ":"))

def grouped(metric, title, ylabel, fname, log=True, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.8 / len(RUNTIMES)
    top = 0
    missing = []
    for i, (rt, label) in enumerate(RUNTIMES):
        xs, ys, lo, hi = [], [], [], []
        for j, dev in enumerate(devices):
            r = summary["results"].get(dev, {}).get(rt, {})
            s = r.get(metric)
            x = j + (i - (len(RUNTIMES) - 1) / 2) * width
            if not s:
                if r.get("errors"): missing.append((x, rt, dev))
                continue
            xs.append(x); ys.append(s["median"]); lo.append(s["median"] - s["p25"]); hi.append(s["p75"] - s["median"])
            top = max(top, s["p75"])
        ax.bar(xs, ys, width * 0.92, color=COLORS[rt], label=label, yerr=[lo, hi], error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": "#333"})
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, (y + h) * (1.06 if log else 1.01), fmt.format(y), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(devices)), [DEVICES[d] for d in devices])
    if log:
        ax.set_yscale("log")
        ax.set_ylim(top=top * 2.2)
    # A runtime that did not finish on a device keeps its place: an empty hatched bar as tall as
    # that device's tallest bar.
    bottom = ax.get_ylim()[0]
    for x, rt, dev in missing:
        h = max((v[metric]["median"] for v in summary["results"].get(dev, {}).values() if v.get(metric)), default=top)
        ax.bar([x], [h - bottom], width * 0.92, bottom=bottom, fill=False, hatch="////", edgecolor=COLORS[rt], linewidth=0.8)
        ax.text(x, bottom * 1.15, "did not finish", rotation=90, ha="center", va="bottom", fontsize=7, color="#444",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

grouped("warmRealtime", "Speed: seconds of speech per second of compute (warm)", "× real time (log)", "speed.svg")
grouped("warmFirstMs", "Time to first audio (warm)", "milliseconds (log)", "first-audio.svg", fmt="{:.1f}")
grouped("coldStartMs", "Cold start: load the runtime, then first audio (fresh page, files cached)", "milliseconds (log)", "cold-start.svg")

# Speed against length, per device: each warm phrase.
fig, axes = plt.subplots(1, len(devices), figsize=(3.2 * len(devices) + 0.6, 3.6), sharey=True, squeeze=False)
for ax, dev in zip(axes[0], devices):
    for rt, label in RUNTIMES:
        wp = warm_phrases(dev, rt)
        if wp: ax.scatter([p["audioSecs"] for p in wp], [p["audioSecs"] / (p["totalMs"] / 1000) for p in wp], s=7, alpha=0.5, color=COLORS[rt], label=label, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(DEVICES[dev], fontsize=10); ax.set_xlabel("seconds of speech")
    ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"]); ax.minorticks_off()
    ax.set_yticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
axes[0][0].set_ylabel("× real time")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, ncols=4, loc="lower center", markerscale=2)
fig.suptitle("Speed against text length (warm, every phrase and run)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "speed-by-length.svg")); plt.close(fig)

# Word times against a forced aligner.
t = os.path.join(HERE, "results", "timing-summary.json")
if os.path.exists(t):
    s = json.load(open(t)); e = np.array(s["errorsMs"])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.hist(e, bins=np.arange(-300, 301, 20), color=COLORS["ours"], alpha=0.85)
    ax.axvline(np.median(e), color="#333", lw=0.8, ls="--")
    ax.text(np.median(e) + 6, ax.get_ylim()[1] * 0.92, f"median {np.median(e):+.0f} ms (blixt/phonon later)", fontsize=8)
    ax.set_xlabel("word start: blixt/phonon minus wav2vec2 forced alignment (ms)"); ax.set_ylabel("words")
    ax.set_title(f"Word timestamps: {len(e)} words, 4 voices, median error {np.median(np.abs(e)):.0f} ms", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "word-timing.svg")); plt.close(fig)
print(json.dumps({d: {rt: {k: (v["median"] if isinstance(v, dict) and "median" in v else v) for k, v in r.items() if k != "errors"} for rt, r in rs.items()} for d, rs in summary["results"].items()}, indent=1)[:3000])

# Quality on one device's warm clips (bench/quality.py): word errors and predicted MOS.
q = os.path.join(HERE, "results", "quality-mac.json")
if os.path.exists(q):
    s = json.load(open(q))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, scale, label, title in [
        (axes[0], "wer", 100, "word error rate (%)", "Word errors (Whisper large-v3-turbo)"),
        (axes[1], "utmos", 1, "predicted MOS", "UTMOS"),
    ]:
        rts = [(rt, lab) for rt, lab in RUNTIMES if rt in s[key]]
        vals = [s[key][rt] for rt, _ in rts]
        ys = [v[0] * scale for v in vals]
        err = [[(v[0] - v[1]) * scale for v in vals], [(v[2] - v[0]) * scale for v in vals]]
        ax.bar(range(len(rts)), ys, 0.7, color=[COLORS[rt] for rt, _ in rts], yerr=err, error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": "#333"})
        for i, y in enumerate(ys): ax.text(i, y + err[1][i] + (0.02 if key == "wer" else 0.02), f"{y:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(rts)), [lab.replace(" (", "\n(") for _, lab in rts], fontsize=7)
        ax.set_ylabel(label); ax.set_title(title, loc="left", fontsize=10, fontweight="bold"); ax.grid(axis="x", visible=False)
        if key == "utmos": ax.set_ylim(1, 5)
    fig.suptitle(f"Quality: {min(s['clips'].values())} warm clips per runtime on the M3 Max, 95% intervals", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "quality.svg")); plt.close(fig)
