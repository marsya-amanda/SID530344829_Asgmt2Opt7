#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.audio import load_wav, normalise, wav_to_logmel
from foley.metrics import aligned_rmse


def main():
    """Compare one generated WAV against cached training mels."""
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--norm", type=Path, default=None)
    a = p.parse_args()
    import json
    norm = json.loads((a.norm or a.cache / "norm.json").read_text())
    query = normalise(wav_to_logmel(load_wav(a.wav)), **norm)
    train = normalise(np.load(a.cache / "train_logmel.npy").astype(np.float32), **norm)
    distances = [aligned_rmse(query, item) for item in train]
    print(sorted(distances)[:3])


if __name__ == "__main__":
    main()
