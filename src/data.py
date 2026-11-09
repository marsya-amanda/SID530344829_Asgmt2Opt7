import json, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foley.constants import * # type: ignore
from foley import audio, descriptors # type: ignore

def list_clips(split_root: Path) -> list[tuple[str, int]]:
    """Recursively list WAVs below class folders as ``(relative_path,class_id)``."""
    root = Path(split_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = []
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root)
        class_dir = next((p for p in rel.parts[:-1] if p.lower().replace("_", "") in
                          {"dogbark", "footstep", "gunshot", "keyboard",
                           "movingmotorvehicle", "rain", "sneezecough"}), None)
        if class_dir is None:
            raise ValueError(f"cannot infer class folder for {wav}")
        out.append((str(rel), folder_to_id(class_dir)))
    return out

def make_split(clips, val_frac=0.1, seed=42):
    """Return deterministic sorted stratified train and validation indices."""
    if not clips or not 0 < val_frac < 1:
        raise ValueError("clips must be non-empty and val_frac must be in (0,1)")
    labels = [c for _, c in clips]
    tr, va = train_test_split(list(range(len(clips))), test_size=val_frac,
                              random_state=seed, stratify=labels)
    return sorted(tr), sorted(va)

def write_split(clips, tr, va, out_dir, val_frac, seed):
    """Write split lists and JSON metadata without including evaluation paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, indices in (("train.txt", tr), ("val.txt", va)):
        lines = [f"{clips[i][0]}\t{clips[i][1]}\n" for i in indices]
        (out / name).write_text("".join(lines), encoding="utf-8")
    counts = {str(c): {"train": sum(clips[i][1] == c for i in tr),
                       "val": sum(clips[i][1] == c for i in va)} for c in range(N_CLASSES)}
    (out / "split_info.json").write_text(json.dumps(
        {"val_frac": val_frac, "seed": seed, "n_train": len(tr), "n_val": len(va),
         "per_class": counts}, indent=2), encoding="utf-8")

class MelDataset(Dataset):
    """Reads cache/{split}_logmel.npy (N,80,345 float16) + {split}_labels.npy.
    Returns dict(x=(1,80,352) float, y=int, u=(3,) float, mask=(3,) float)."""
    def __init__(self, cache, split, norm, ctrl_stats=None, train=False,
                 shift_prob=0.0, ctrl_mask_p=0.5):
        self.lm = np.load(f"{cache}/{split}_logmel.npy", mmap_mode="r")
        self.y = np.load(f"{cache}/{split}_labels.npy")
        self.norm, self.stats, self.train = norm, ctrl_stats, train
        self.shift_prob, self.mask_p = (shift_prob, ctrl_mask_p) if train else (0.0, 1.0)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        lm, y = np.asarray(self.lm[i], dtype=np.float32), int(self.y[i])
        u, mask = np.zeros(3, np.float32), np.zeros(3, np.float32)
        if self.stats is not None:                                   # bonus fine-tune only
            rng = np.random.default_rng() if self.train else None
            mask = (rng.random(3) < self.mask_p).astype(np.float32) if self.train else np.ones(3, np.float32)
            if self.train and y in descriptors.TRANSIENT and rng.random() < self.shift_prob:
                lm = descriptors.shift_right(lm, int(rng.integers(1, 215)))   # up to ~2.5 s delay
                mask[descriptors.ONSET] = 1.0      # a delayed clip MUST be labelled with its onset
            u = descriptors.to_unit(descriptors.compute_all(lm), y, self.stats)
        x = audio.normalise(lm, **self.norm)
        return {"x": torch.from_numpy(x)[None], "y": torch.tensor(y, dtype=torch.long),
                "u": torch.from_numpy(u), "mask": torch.from_numpy(mask)}
