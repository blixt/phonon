# The listening page: every benchmark phrase in each runtime, from one device's first warm round
# (bench/results/audio/DEVICE), as 96 kbps MP3 in docs/audio and a table in docs/listen.html.
# Usage: python bench/listen.py [DEVICE]   (needs ffmpeg)
import html, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
device = sys.argv[1] if len(sys.argv) > 1 else "mac"
RUNTIMES = [("ours", "blixt/phonon", "Phonon 31M, voice Delia, WebGPU"), ("gradiumWebgpu", "phonon-tts WebGPU", "Phonon 31M, voice Delia"), ("gradiumCpu", "phonon-tts CPU", "Phonon 31M, voice Delia"), ("kokoro", "Kokoro 82M", "kokoro-js, voice af_heart")]
phrases = json.load(open(os.path.join(HERE, "phrases.json")))["phrases"]
rows = []
for p in phrases:
    cells = []
    for rt, _, _ in RUNTIMES:
        src = os.path.join(HERE, "results", "audio", device, rt, f"{p['id']}~1.wav")
        if not os.path.exists(src):
            cells.append("<td>–</td>"); continue
        rel = f"audio/{rt}/{p['id']}.mp3"
        os.makedirs(os.path.join(DOCS, "audio", rt), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-b:a", "96k", os.path.join(DOCS, rel)], check=True)
        cells.append(f'<td><audio controls preload="none" src="{rel}"></audio></td>')
    rows.append(f"<tr><th>{html.escape(p['text'])}</th>{''.join(cells)}</tr>")
heads = "".join(f"<th>{n}<br><small>{d}</small></th>" for _, n, d in RUNTIMES)
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>blixt/phonon: listen to the benchmark phrases</title>
<link rel="stylesheet" href="style.css"></head>
<body>
<header class="top wrap"><a class="mark" href="index.html">blixt/phonon</a>
<nav><a href="index.html#demo">Try it</a><a href="index.html#results">Results</a><a class="gh" href="https://github.com/blixt/phonon">GitHub</a></nav></header>
<main class="wrap lp">
<h1>Listen to the benchmark phrases</h1>
<p>Each clip is the first warm run on the MacBook Pro (M3 Max) in Chrome. The three Phonon runtimes use the same model and voice. They sample with different random numbers, so each clip is a different take of the same text. Kokoro is a different model, for reference.</p>
<div class="table"><table><thead><tr><th>Text</th>{heads}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></div>
</main>
</body></html>
"""
open(os.path.join(DOCS, "listen.html"), "w").write(page)
print("wrote", len(rows), "rows")
