import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import audio, descriptors
from .constants import N_CLASSES, folder_to_id


def list_clips(split_root: Path) -> list[tuple[str, int]]:
    """Recursively list WAVs below class folders as ``(relative_path,class_id)``."""
    root = Path(split_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    keys = {"dogbark", "footstep", "gunshot", "keyboard", "movingmotorvehicle",
            "rain", "sneezecough"}
    clips = []
    for wav in sorted(root.rglob("*.wav")):
        class_dir = next((part for part in wav.relative_to(root).parts[:-1]
                          if part.lower().replace("_", "").replace("-", "").replace(" ", "").replace("/", "") in keys), None)
        if class_dir is None:
            raise ValueError(f"cannot infer class folder for {wav}")
        clips.append((str(wav.relative_to(root)), folder_to_id(class_dir)))
    return clips


def make_split(clips, val_frac=0.1, seed=42):
    """Return deterministic sorted stratified train and validation indices."""
    if not clips or not 0 < val_frac < 1:
        raise ValueError("clips must be non-empty and val_frac must be in (0,1)")
    labels = [label for _, label in clips]
    train, val = train_test_split(list(range(len(clips))), test_size=val_frac,
                                  random_state=seed, stratify=labels)
    return sorted(train), sorted(val)


def write_split(clips, train, val, out_dir, val_frac, seed):
    """Write split lists and JSON metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, indices in (("train.txt", train), ("val.txt", val)):
        (out / name).write_text(
            "".join(f"{clips[i][0]}\t{clips[i][1]}\n" for i in indices), encoding="utf-8")
    counts = {str(c): {"train": sum(clips[i][1] == c for i in train),
                       "val": sum(clips[i][1] == c for i in val)} for c in range(N_CLASSES)}
    (out / "split_info.json").write_text(json.dumps(
        {"val_frac": val_frac, "seed": seed, "n_train": len(train), "n_val": len(val),
         "per_class": counts}, indent=2), encoding="utf-8")


class MelDataset(Dataset):
    """Read cached mels and return x ``(1,80,352)``, class, controls, and masks."""

    def __init__(self, cache, split, norm, ctrl_stats=None, train=False,
                 shift_prob=0.0, ctrl_mask_p=0.5):
        self.lm = np.load(Path(cache) / f"{split}_logmel.npy", mmap_mode="r")
        self.y = np.load(Path(cache) / f"{split}_labels.npy")
        self.norm, self.stats, self.train = norm, ctrl_stats, train
        self.shift_prob, self.mask_p = (shift_prob, ctrl_mask_p) if train else (0.0, 1.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        lm, label = np.asarray(self.lm[index], dtype=np.float32), int(self.y[index])
        u, mask = np.zeros(3, np.float32), np.zeros(3, np.float32)
        if self.stats is not None:
            rng = np.random.default_rng() if self.train else None
            mask = ((rng.random(3) < self.mask_p).astype(np.float32)
                    if self.train else np.ones(3, np.float32))
            if self.train and label in descriptors.TRANSIENT and rng.random() < self.shift_prob:
                lm = descriptors.shift_right(lm, int(rng.integers(1, 215)))
                mask[descriptors.ONSET] = 1.0
            u = descriptors.to_unit(descriptors.compute_all(lm), label, self.stats)
        x = audio.normalise(lm, **self.norm)
        return {"x": torch.from_numpy(x)[None], "y": torch.tensor(label, dtype=torch.long),
                "u": torch.from_numpy(u), "mask": torch.from_numpy(mask)}


def load_normalised_cache(cache, split, norm):
    """Load and normalise a cache as ``(N,80,352)`` float32 plus ``(N,)`` labels."""
    mels = np.load(Path(cache) / f"{split}_logmel.npy").astype(np.float32)
    labels = np.load(Path(cache) / f"{split}_labels.npy").astype(np.int64)
    return audio.normalise(mels, **norm), labels
