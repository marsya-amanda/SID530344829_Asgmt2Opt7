#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.audio import save_wav
from foley.generate import generate, load_ckpt


def main():
    """Generate the exactly nine required demonstration WAV files."""
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=ROOT / "weights/core.pt")
    p.add_argument("--out", type=Path, default=ROOT / "demo_wavs")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model, diffusion, ck = load_ckpt(a.weights, a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [(c, 0) for c in range(7)] + [(1, 1), (1, 2)]
    for class_id, seed in seeds:
        wavs, _ = generate(model, diffusion, ck, [class_id], [seed],
                           ck["gen_defaults"]["w"], ck["gen_defaults"]["steps"], device=a.device)
        save_wav(a.out / f"class_{class_id}_seed_{seed}.wav", wavs[0])


if __name__ == "__main__":
    main()
