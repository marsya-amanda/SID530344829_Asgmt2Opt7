#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley import audio, descriptors
from foley.constants import N_CLASSES, N_FRAMES
from foley.data import list_clips


def read_split(path):
    """Read ``relative_path<TAB>class_id`` records."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            relative, label = line.split("\t")
            rows.append((relative, int(label)))
    return rows


def cache_records(records, root, cache, name):
    """Cache float16 log-mels and integer labels for records."""
    mels = np.stack([audio.wav_to_logmel(audio.load_wav(root / rel)) for rel, _ in records])
    if mels.shape[1:] != (80, N_FRAMES):
        raise ValueError(f"unexpected mel cache shape: {mels.shape}")
    np.save(cache / f"{name}_logmel.npy", mels.astype(np.float16))
    np.save(cache / f"{name}_labels.npy", np.asarray([label for _, label in records], np.int64))
    return mels


def main():
    """Precompute train/val/eval log-mels and train-only statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--splits", type=Path, default=ROOT / "splits")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    train = read_split(args.splits / "train.txt")
    val = read_split(args.splits / "val.txt")
    train_mels = cache_records(train, args.root / "dev", args.cache, "train")
    cache_records(val, args.root / "dev", args.cache, "val")
    eval_records = list_clips(args.root / "eval")
    cache_records(eval_records, args.root / "eval", args.cache, "eval")
    norm = audio.fit_norm(train_mels)
    (args.cache / "norm.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    quantiles = [[] for _ in range(N_CLASSES)]
    for mel, (_, label) in zip(train_mels, train):
        quantiles[label].append(descriptors.centroid(mel))
    stats = {"centroid_q": [np.quantile(v, np.linspace(0, 1, 101)).tolist()
                            for v in quantiles]}
    (args.cache / "ctrl_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
