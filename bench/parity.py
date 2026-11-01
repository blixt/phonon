# Reference side of the parity check: runs the PyTorch model in Transformers (f32) on each text
# that parity.mjs spoke, with the engine's own noise (its PCG hash and Box-Muller) in place of
# torch.randn. With seed 1 it is the same take as the engine's, and with seed 2 a different take,
# for scale. It compares lengths in frames, log-mel distance and waveform correlation.
# Usage: python bench/parity.py DIR   (needs torch, transformers from source, safetensors, numpy)
import glob, json, sys, numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

REV = "92741c5b07d085ff840e26fea8ce43110b87852f"
mid = "gradium/phonon-31m-en"
proc = AutoProcessor.from_pretrained(mid, revision=REV)
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(mid, revision=REV, dtype=torch.float32).eval()
voice = load_file(hf_hub_download("gradium/phonon-voices", "voices/nGzMHMPAWGaJFHOo.safetensors", revision="0ee17205f254b5abc43b266c8a0f215565c352f7"))["speaker_wavs"][0].T.float()
M = 0xFFFFFFFF

def pcg(v):
    s = (v * 747796405 + 2891336453) & M
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M
    return (w >> 22) ^ w

def noise(seed, frame):
    i = np.arange(32, dtype=np.uint64)
    a = pcg(np.uint64(seed) ^ pcg(np.uint64(frame) * 64 + i * 2))
    b = pcg(a ^ np.uint64(0x9E3779B9))
    u1 = ((a >> 8).astype(np.float32) + 1) / np.float32(16777217)
    u2 = (b >> 8).astype(np.float32) / np.float32(16777216)
    return (np.sqrt(-2 * np.log(u1)) * np.cos(np.float32(6.283185307) * u2)).astype(np.float32)

def reference(text, seed):
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True, processor_kwargs={"audio_prompt_latents": [voice]})
    frame = {"n": 0}
    real_randn = torch.randn
    def fake_randn(*shape, **kw):
        if tuple(shape) == (1, 32):
            v = torch.from_numpy(noise(seed, frame["n"])[None]); frame["n"] += 1
            return v.to(kw.get("dtype") or torch.float32)
        return real_randn(*shape, **kw)
    torch.randn = fake_randn
    try:
        with torch.no_grad():
            g = model.generate(**inp)
    finally:
        torch.randn = real_randn
    return proc.decode(g.audio_latents, g.audio_latents_mask)[0].float().numpy().reshape(-1)

def mel_bank(n_fft=512, sr=24000, n=40):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1); mel = lambda h: 2595 * np.log10(1 + h / 700)
    pts = hz(np.linspace(mel(60), mel(11000), n + 2)); bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n, n_fft // 2 + 1))
    for i in range(n):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(1, c - l); fb[i, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return torch.tensor(fb, dtype=torch.float32)
FB = mel_bank()
def logmel(a):
    spec = torch.stft(torch.tensor(a), 512, hop_length=480, win_length=480, window=torch.hann_window(480), return_complex=True).abs() ** 2
    return (10 * torch.log10(FB @ spec + 1e-8)).T.numpy()
def mel_distance(a, b):
    """Mean RMS distance (dB) of 40-band log-mel frames, over the shorter clip, frame by frame."""
    x, y = logmel(a), logmel(b); n = min(len(x), len(y))
    return float(np.sqrt(((x[:n] - y[:n]) ** 2).mean(-1)).mean())
def correlation(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    x = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    near = np.r_[x[:480], x[-480:]]
    return float(near.max() / np.sqrt((a * a).sum() * (b * b).sum()))

rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    meta = json.load(open(f))
    ours = np.fromfile(f[:-5] + ".f32", dtype=np.float32)
    ref = reference(meta["text"], 1)
    other = reference(meta["text"], 2)
    rows.append({"id": meta["id"], "refFrames": len(ref) / 1920, "oursFrames": len(ours) / 1920, "otherSeedFrames": len(other) / 1920,
                 "melDbOursVsRef": mel_distance(ours, ref), "melDbOtherSeedVsRef": mel_distance(other, ref),
                 "corrOursVsRef": correlation(ours, ref), "corrOtherSeedVsRef": correlation(other, ref)})
    print(rows[-1], flush=True)
summary = {"texts": rows,
           "sameLength": sum(r["refFrames"] == r["oursFrames"] for r in rows),
           "melDbOursVsRef": float(np.mean([r["melDbOursVsRef"] for r in rows])),
           "melDbOtherSeedVsRef": float(np.mean([r["melDbOtherSeedVsRef"] for r in rows])),
           "corrOursVsRef": float(np.mean([r["corrOursVsRef"] for r in rows])),
           "corrOtherSeedVsRef": float(np.mean([r["corrOtherSeedVsRef"] for r in rows]))}
json.dump(summary, open(f"{sys.argv[1]}/../parity-summary.json", "w"), indent=1)
print({k: v for k, v in summary.items() if k != "texts"})
