# The README's results table, from docs/results.json, written between its results markers.
# Usage: python bench/table.py
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "..", "docs", "results.json")))
NAMES = {"ours": "**blixt/phonon**", "gradiumWebgpu": "phonon-tts, WebGPU", "gradiumCpu": "phonon-tts, CPU", "kokoro": "Kokoro 82M, WebGPU"}
def m(s, f="{:.0f}"):
    return f.format(s["median"]) if s else "did not finish"
# A runtime with no first visit and no errors has none to measure (plots.py says why).
def first_visit(x):
    s = x.get("firstVisitMs")
    if not s: return "did not finish" if x["errors"] else "–"
    return f"{s['median'] / 1000:.1f}"
lines = ["| Device | Runtime | Speed (× real time) | First audio, warm (ms) | Cold start (ms) | First visit (s) | All 11 phrases, warm (s) |", "|---|---|--:|--:|--:|--:|--:|"]
for dev, d in r["devices"].items():
    for rt, name in NAMES.items():
        x = r["results"].get(dev, {}).get(rt)
        if x is None: continue
        lines.append(f"| {d['label']} | {name} | {m(x['warmRealtime'])} | {m(x['warmFirstMs'], '{:.1f}')} | {m(x['coldStartMs'])} | {first_visit(x)} | {m(x['warmTotalSecs'], '{:.1f}')} |")
readme = os.path.join(HERE, "..", "README.md")
text = open(readme).read()
start, end = "<!-- results:start -->", "<!-- results:end -->"
before, rest = text.split(start)
open(readme, "w").write(before + start + "\n" + "\n".join(lines) + "\n" + end + rest.split(end)[1])
print("\n".join(lines))
