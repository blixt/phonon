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
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phonon on WebGPU: listen</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; margin: 1.5rem; color: #1d1d1f; }}
table {{ border-collapse: collapse; }} td, th {{ padding: 0.4rem; border-bottom: 1px solid #ddd; vertical-align: middle; text-align: left; }}
tbody th {{ font-weight: normal; max-width: 22rem; }} audio {{ width: 13rem; }} small {{ font-weight: normal; color: #6e6e73; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #111; color: #eee; }} td, th {{ border-color: #333; }} }}
</style>
<h1>Listen: the benchmark phrases</h1>
<p>Each clip is the first warm run on the MacBook Pro (M3 Max) in Chrome. The three Phonon runtimes use the same model and voice. They sample with different random numbers, so each clip is a different take of the same text. Kokoro is a different model, for reference.</p>
<p><a href="index.html">Try it with your own text</a> · <a href="https://github.com/blixt/phonon">Source</a></p>
<table><thead><tr><th>Text</th>{heads}</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</html>
"""
open(os.path.join(DOCS, "listen.html"), "w").write(page)
print("wrote", len(rows), "rows")
