#!/usr/bin/env python
import argparse
import csv
import shutil
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    """Select the latest EMA checkpoint and rewrite generation defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=Path, default=ROOT / "ckpts")
    p.add_argument("--out", type=Path, default=ROOT / "weights/core.pt")
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=50)
    a = p.parse_args()
    candidates = sorted(a.ckpts.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no step checkpoints in {a.ckpts}")
    ck = torch.load(candidates[-1], map_location="cpu", weights_only=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    raw = a.out.with_name("core_raw.pt")
    torch.save(ck, raw)
    ck["gen_defaults"] = {"w": float(a.guidance), "steps": int(a.steps)}
    torch.save(ck, a.out)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/checkpoint_selection.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["checkpoint", "selected"])
        writer.writerow([str(candidates[-1]), True])


if __name__ == "__main__":
    main()
