#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley import audio
from foley.classifier import SmallCNN
from foley.generate import generate, load_ckpt
from foley.metrics import accuracy_confusion, classify, pairwise_rmse


def synth_set(model, diffusion, checkpoint, guidance, seeds, steps, device, out_dir):
    """Generate WAV paths for every class and requested seed."""
    from foley.audio import save_wav
    paths = {c: [] for c in range(7)}
    for c in range(7):
        for seed in seeds:
            wavs, _ = generate(model, diffusion, checkpoint, [c], [seed],
                               guidance, steps, device=device)
            path = Path(out_dir) / f"class_{c}_seed_{seed}.wav"
            save_wav(path, wavs[0]); paths[c].append(path)
    return paths


def analyse(paths_by_class, classifier, norm, device):
    """Reanalyse generated WAVs and return classifier and diversity metrics."""
    mels, labels = [], []
    for label, paths in paths_by_class.items():
        for path in paths:
            mels.append(audio.normalise(audio.wav_to_logmel(audio.load_wav(path)), **norm))
            labels.append(label)
    probs, _ = classify(classifier, np.stack(mels), device)
    accuracy, per_class, confusion = accuracy_confusion(probs, labels)
    return {"accuracy": accuracy, "per_class_accuracy": per_class.tolist(),
            "confusion": confusion.tolist(), "diversity": pairwise_rmse(np.stack(mels))}


def main():
    """Run a bounded generated-set evaluation from a saved classifier and model."""
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=ROOT / "weights/core.pt")
    p.add_argument("--classifier", type=Path, default=ROOT / "weights/classifier.pt")
    p.add_argument("--norm", type=Path, default=ROOT / "cache/norm.json")
    p.add_argument("--out", type=Path, default=ROOT / "logs/eval_generated")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=2)
    a = p.parse_args()
    model, diffusion, checkpoint = load_ckpt(a.weights, a.device)
    classifier = SmallCNN().to(a.device)
    classifier.load_state_dict(torch.load(a.classifier, map_location=a.device, weights_only=True)["model"])
    norm = json.loads(a.norm.read_text())
    paths = synth_set(model, diffusion, checkpoint, checkpoint["gen_defaults"]["w"],
                      range(a.seeds), checkpoint["gen_defaults"]["steps"], a.device, a.out)
    result = analyse(paths, classifier, norm, a.device)
    (a.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
