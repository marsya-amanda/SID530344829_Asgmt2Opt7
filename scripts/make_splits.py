#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.data import list_clips, make_split, write_split


def main():
    """Create a deterministic stratified split of the labelled dev directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    clips = list_clips(args.root / "dev")
    train, val = make_split(clips, args.val_frac, args.seed)
    write_split(clips, train, val, args.out, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
