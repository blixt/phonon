# Checks the engine's word times (timing.mjs) against an independent forced aligner:
# wav2vec2-base-960h CTC with Viterbi alignment of the known words. Prints start-time errors.
# Usage: python bench/align.py DIR   (needs torch, transformers, numpy)
import glob, json, re, sys, numpy as np, torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
mid = "facebook/wav2vec2-base-960h"
proc = Wav2Vec2Processor.from_pretrained(mid); model = Wav2Vec2ForCTC.from_pretrained(mid).eval()
vocab = proc.tokenizer.get_vocab(); blank = vocab["<pad>"]
STEP = 320 / 16000  # seconds per wav2vec2 frame

def align(audio16, words):
    with torch.no_grad():
        x = (audio16 - audio16.mean()) / (audio16.std() + 1e-7)
        lp = torch.log_softmax(model(torch.tensor(x, dtype=torch.float32)[None]).logits[0], -1).numpy()
    S, owner = [blank], [-1]
    for wi, w in enumerate(words):
        letters = re.sub(r"[^A-Z']", "", w.upper()) or "A"
        if wi: S += [vocab["|"], blank]; owner += [-1, -1]
        for c in letters: S += [vocab[c], blank]; owner += [wi, -1]
    T, n = lp.shape[0], len(S)
    S = np.array(S); dp = np.full((T, n), -1e9); bp = np.zeros((T, n), int)
    dp[0, :2] = lp[0, S[:2]]
    skip = np.zeros(n, bool); skip[2:] = (S[2:] != blank) & (S[2:] != S[:-2])
    for t in range(1, T):
        stay, one = dp[t - 1], np.concatenate([[-1e9], dp[t - 1, :-1]])
        two = np.where(skip, np.concatenate([[-1e9, -1e9], dp[t - 1, :-2]]), -1e9)
        best = np.argmax(np.stack([stay, one, two]), 0)
        dp[t] = np.choose(best, [stay, one, two]) + lp[t, S]; bp[t] = best
    s = n - 1 if dp[-1, n - 1] > dp[-1, n - 2] else n - 2
    starts = {}
    for t in range(T - 1, -1, -1):
        if owner[s] >= 0: starts[owner[s]] = t * STEP
        s -= bp[t, s]
    return [starts.get(i) for i in range(len(words))]

errs = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    meta = json.load(open(f))
    pcm = np.fromfile(f[:-5] + ".f32", dtype=np.float32)
    a16 = np.interp(np.arange(0, len(pcm) / 24000, 1 / 16000), np.arange(len(pcm)) / 24000, pcm).astype(np.float32)
    words = [w["text"] for w in meta["words"]]
    ref = align(a16, words)
    for w, r in zip(meta["words"], ref):
        if r is not None: errs.append((w["start"] - r) * 1000)
signed = np.array(errs); errs = np.abs(signed)
summary = {"words": len(errs), "meanMs": float(errs.mean()), "medianMs": float(np.median(errs)), "p90Ms": float(np.percentile(errs, 90)), "within100ms": float((errs <= 100).mean()), "meanSignedMs": float(signed.mean()), "medianSignedMs": float(np.median(signed)), "errorsMs": signed.round(1).tolist()}
json.dump(summary, open(f"{sys.argv[1]}/../timing-summary.json", "w"))
print({k: v for k, v in summary.items() if k != "errorsMs"})
