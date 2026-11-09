#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.audio import save_wav
from foley.generate import generate, load_ckpt


def main():
    """Generate a small controlled demonstration set."""
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=ROOT / "weights/ctrl.pt")
    p.add_argument("--out", type=Path, default=ROOT / "bonus_demo")
    p.add_argument("--class-id", type=int, default=1)
    p.add_argument("--control", choices=("brightness", "onset", "density"), default="brightness")
    p.add_argument("--values", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model, diffusion, ck = load_ckpt(a.weights, a.device)
    for value in a.values:
        key = {"brightness": "brightness", "onset": "onset_s", "density": "count"}[a.control]
        argument = value * 4.0 if a.control == "onset" else value
        wavs, _ = generate(model, diffusion, ck, [a.class_id], [7],
                           ck["gen_defaults"]["w"], ck["gen_defaults"]["steps"],
                           {key: argument}, a.device)
        path = a.out / a.control / f"class_{a.class_id}_seed_7_{a.control}_{value:g}.wav"
        save_wav(path, wavs[0])


if __name__ == "__main__":
    main()
