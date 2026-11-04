# Finds a model's alignment head and lag, step 2 of 2: scores the takes bench/heads.mjs recorded
# against MMS forced alignment (torchaudio MMS_FA on romanized text, so any of the five languages).
# For each head and lag: word starts and token starts inside words, as mean, median, p90 and share
# within 80 ms. The head is the one with the lowest word-start mean at lag 3. Another lag replaces 3
# only when it is no worse on every one of those measures, over all takes, and better on the mean.
# The script also prints each voice's measures: check them on enough text before you change a lag.
# Usage: python bench/heads.py OUT LANG   (needs torch, torchaudio, unidecode)
import glob, json, os, sys, numpy as np, torch, torchaudio
from unidecode import unidecode

out, lang = sys.argv[1], sys.argv[2]
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
dic = bundle.get_dict(star=None)
rom = lambda s: "".join(c for c in unidecode(s).lower() if dic.get(c, 0) != 0)
refs = {}

def reference(f, take):
    """Each token's start in the audio, by MMS's per-character alignment (cached per take)."""
    key = os.path.basename(f)
    if key in refs: return refs[key]
    pcm = np.fromfile(f[:-5] + ".f32", dtype=np.float32)
    wav = torchaudio.functional.resample(torch.tensor(pcm)[None], 24000, 16000)
    toks = [(wi, ti, t["text"]) for wi, w in enumerate(take["byLag"]["3"]) for ti, t in enumerate(w["tokens"]) if rom(t["text"])]
    chars = [dic[c] for _, _, s in toks for c in rom(s)]
    with torch.inference_mode():
        em, _ = model(wav)
        ali, sc = torchaudio.functional.forced_align(em, torch.tensor(chars, dtype=torch.int32)[None], blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], sc[0].exp())
    ratio = wav.shape[1] / em.shape[1] / 16000
    starts, k = {}, 0
    for wi, ti, s in toks:
        starts[(wi, ti)] = spans[k].start * ratio
        k += len(rom(s))
    refs[key] = starts
    return starts

def stats(errs):
    a = np.abs(np.array(errs)) * 1000
    return {"mean": a.mean(), "median": np.median(a), "p90": np.percentile(a, 90), "within80": (a <= 80).mean()}

def score(dirname, lag):
    """Per voice and over all: word starts and token starts inside words."""
    per = {}
    for f in sorted(glob.glob(f"{dirname}/*.json")):
        take = json.load(open(f))
        ref = reference(f, take)
        for wi, w in enumerate(take["byLag"][lag]):
            for ti, t in enumerate(w["tokens"]):
                if (wi, ti) in ref:
                    for v in (take["voice"], "all"):
                        per.setdefault(v, {"word": [], "inner": []})["inner" if ti else "word"].append(t["start"] - ref[(wi, ti)])
    return {v: {k: stats(e) for k, e in d.items()} for v, d in per.items()}

def no_worse(a, b):
    """Every measure of a, over all takes, is at least as good as b's."""
    for k in ("word", "inner"):
        x, y = a["all"][k], b["all"][k]
        if x["mean"] > y["mean"] or x["median"] > y["median"] or x["p90"] > y["p90"] or x["within80"] < y["within80"]: return False
    return True

heads = sorted(d for d in glob.glob(f"{out}/L*H*") if os.path.isdir(d))
at3 = {os.path.basename(d): score(d, "3") for d in heads}
best = min(at3, key=lambda h: at3[h]["all"]["word"]["mean"])
lags = sorted(json.load(open(glob.glob(f"{out}/{best}/*.json")[0]))["byLag"], key=float)
choice, base = "3", at3[best]
for lag in lags:
    s = score(f"{out}/{best}", lag)
    if lag != "3" and no_worse(s, base) and s["all"]["word"]["mean"] < base["all"]["word"]["mean"]:
        choice, base = lag, s
for h in sorted(at3, key=lambda h: at3[h]["all"]["word"]["mean"])[:3]:
    print(f"{h} at lag 3: word starts mean {at3[h]['all']['word']['mean']:.0f} ms")
for v, d in sorted(base.items()):
    print(f"  {v}: word starts mean {d['word']['mean']:.0f} ms, p90 {d['word']['p90']:.0f} ms; tokens inside words mean {d['inner']['mean']:.0f} ms")
b = base["all"]
print(f"{lang}: layer {best[1:best.index('H')]}, head {best[best.index('H') + 1:]}, lag {choice}: word starts mean {b['word']['mean']:.0f} ms, median {b['word']['median']:.0f} ms; tokens inside words mean {b['inner']['mean']:.0f} ms")
