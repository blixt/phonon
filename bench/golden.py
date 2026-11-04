# Writes bench/golden.json: the PyTorch reference's take (parity.py, with the engine's noise) of a
# few short texts in the voice Delia, as its length in frames and its log-mel frames (parity.py's
# 40 bands, in steps of 0.25 dB). bench/golden.mjs checks each backend against it, with no Python.
# Usage: python bench/golden.py   (needs what parity.py needs)
import base64, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from parity import reference, logmel

phrases = {p["id"]: p["text"] for p in json.load(open(os.path.join(os.path.dirname(__file__), "phrases.json")))["phrases"]}
clips = []
for id, seed in [("h04", 1), ("h05", 1), ("h06", 1), ("h07", 2), ("h09", 3)]:
    a = reference(phrases[id], seed)
    mel = np.round(logmel(a) * 4).astype(np.int16)
    clips.append({"id": id, "seed": seed, "text": phrases[id], "frames": len(a) / 1920, "melFrames": mel.shape[0],
                  "mel": base64.b64encode(mel.tobytes()).decode()})
    print(id, seed, clips[-1]["frames"], flush=True)
json.dump({"model": "phonon-31m-en", "voice": "Delia", "melStepDb": 0.25, "clips": clips}, open(os.path.join(os.path.dirname(__file__), "golden.json"), "w"), indent=1)
