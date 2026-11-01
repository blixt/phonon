# Quality scores for the warm clips of one device (bench/results/audio/DEVICE/RUNTIME/ID~ROUND.wav):
# - word error rate from Whisper large-v3-turbo, with a 95% bootstrap interval, and a paired
#   bootstrap of blixt/phonon against phonon-tts on WebGPU (same text and round),
# - speaker similarity: cosine similarity of WavLM x-vectors between takes of the same text, for
#   blixt/phonon against phonon-tts and for phonon-tts against itself (other rounds and seeds),
# - UTMOS (utmos22_strong), a predicted mean opinion score.
# Usage: python bench/quality.py DEVICE   (needs torch, transformers, soundfile, numpy, and the
# internet for the models). Writes bench/results/quality-DEVICE.json.
import glob, json, os, re, sys, numpy as np, soundfile as sf, torch
from transformers import pipeline, AutoFeatureExtractor, WavLMForXVector

HERE = os.path.dirname(os.path.abspath(__file__))
device = sys.argv[1]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
texts = {p["id"]: p["text"] for p in json.load(open(os.path.join(HERE, "phrases.json")))["phrases"]}
clips = {}
for f in glob.glob(os.path.join(HERE, "results", "audio", device, "*", "*.wav")):
    rt = os.path.basename(os.path.dirname(f))
    pid, rnd = os.path.basename(f)[:-4].split("~")
    clips[(rt, pid, int(rnd))] = f
runtimes = sorted({k[0] for k in clips})
rng = np.random.default_rng(0)

def load16(f):
    a, sr = sf.read(f, dtype="float32")
    return np.interp(np.arange(0, len(a) / sr, 1 / 16000), np.arange(len(a)) / sr, a).astype(np.float32)

def words(t):
    return re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " ")).split()

def edits(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]

def interval(values, n=2000):
    v = np.asarray(values, float)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return [float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

# Word errors.
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=dev, torch_dtype=torch.float16 if dev == "mps" else torch.float32)
errors = {}
for k, f in sorted(clips.items()):
    heard = asr(load16(f), return_timestamps=True, generate_kwargs={"language": "en"})["text"]
    ref = words(texts[k[1]])
    errors[k] = (edits(ref, words(heard)), len(ref), heard)
    print("asr", k, errors[k][0], flush=True)
wer = {}
for rt in runtimes:
    ks = [k for k in errors if k[0] == rt]
    # Interval over clips, each weighted by its words: resample (errors, words) pairs.
    pairs = np.array([errors[k][:2] for k in ks], float)
    boot = [pairs[i][:, 0].sum() / pairs[i][:, 1].sum() for i in (rng.integers(0, len(pairs), len(pairs)) for _ in range(2000))]
    wer[rt] = [float(pairs[:, 0].sum() / pairs[:, 1].sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
paired = None
if "ours" in runtimes and "gradiumWebgpu" in runtimes:
    common = [(p, r) for (rt, p, r) in errors if rt == "ours" and ("gradiumWebgpu", p, r) in errors]
    d = np.array([[errors[("ours", p, r)][0] - errors[("gradiumWebgpu", p, r)][0], errors[("ours", p, r)][1]] for p, r in common], float)
    boot = [d[i][:, 0].sum() / d[i][:, 1].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
    paired = [float(d[:, 0].sum() / d[:, 1].sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

# Speaker similarity.
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
emb = {}
for k, f in clips.items():
    if k[0] == "kokoro": continue
    with torch.no_grad():
        e = sv(**fe(load16(f), sampling_rate=16000, return_tensors="pt")).embeddings[0]
    emb[k] = torch.nn.functional.normalize(e, dim=-1).numpy()
def sims(a, b):
    out = []
    for (rt, p, r), e in emb.items():
        if rt != a: continue
        for (rt2, p2, r2), e2 in emb.items():
            if rt2 == b and p2 == p and r2 != r: out.append(float(e @ e2))
    return out
speaker = {f"{a}~{b}": interval(sims(a, b)) for a, b in [("ours", "gradiumWebgpu"), ("gradiumWebgpu", "gradiumWebgpu"), ("ours", "ours"), ("gradiumCpu", "gradiumWebgpu")] if a in runtimes and b in runtimes}

# Predicted MOS. SpeechMOS imports torchaudio only to resample, and these clips are 16 kHz
# already: a stand-in module lets it run where no torchaudio matches the installed torch.
if "torchaudio" not in sys.modules:
    import types
    def resample(wave, orig_freq, new_freq):
        if orig_freq != new_freq: raise ValueError("only 16 kHz input")
        return wave
    import importlib.machinery
    stand_in = types.ModuleType("torchaudio"); stand_in.functional = types.SimpleNamespace(resample=resample)
    stand_in.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = stand_in
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
mos = {}
for rt in runtimes:
    scores = []
    for k, f in clips.items():
        if k[0] == rt:
            with torch.no_grad():
                scores.append(float(utmos(torch.from_numpy(load16(f))[None], 16000)))
    mos[rt] = interval(scores)

summary = {"device": device, "clips": {rt: sum(1 for k in clips if k[0] == rt) for rt in runtimes},
           "wer": wer, "werOursMinusGradiumWebgpu": paired, "speakerSimilarity": speaker, "utmos": mos,
           "heard": {f"{k[0]}/{k[1]}~{k[2]}": v[2] for k, v in errors.items()}}
json.dump(summary, open(os.path.join(HERE, "results", f"quality-{device}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "heard"}, indent=1))
