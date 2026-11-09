import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from foley.generate import load_ckpt, generate
from foley.audio import save_wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-id", type=int, required=True, choices=range(7))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    # optional (defaults come from the checkpoint, so the 4-arg brief command uses the best settings)
    ap.add_argument("--guidance", type=float); ap.add_argument("--steps", type=int)
    ap.add_argument("--brightness", type=float, help="[bonus] 0..1 within-class percentile")
    ap.add_argument("--onset", type=float, help="[bonus] seconds, 0..3.5")
    ap.add_argument("--density", type=float, help="[bonus] events per 4 s clip")
    ap.add_argument("--device", default="auto"); ap.add_argument("--format", default="pcm16", choices=["pcm16", "float32"])
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if a.device not in ("auto", "cpu", "cuda"):
        raise ValueError("--device must be auto, cpu, or cuda")
    model, diff, ck = load_ckpt(a.weights, dev)
    w = ck["gen_defaults"]["w"] if a.guidance is None else a.guidance
    steps = ck["gen_defaults"]["steps"] if a.steps is None else a.steps
    controls = {"brightness": a.brightness, "onset_s": a.onset, "count": a.density}
    wavs, _ = generate(model, diff, ck, [a.class_id], [a.seed], w, steps, controls, dev)
    save_wav(a.out, wavs[0], "PCM_16" if a.format == "pcm16" else "FLOAT")

if __name__ == "__main__": main()
