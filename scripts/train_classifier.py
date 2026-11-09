#!/usr/bin/env python
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.classifier import SmallCNN  # type: ignore[reportMissingImports]
from foley.audio import normalise  # type: ignore[reportMissingImports]

LOGGER = logging.getLogger("foley.classifier")


def _sync(device):
    """Synchronise CUDA for accurate optional profiling timestamps."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def preload_arrays(train, val, labels, device):
    """Move normalised arrays to GPU as fp16; return train/val tensors and labels."""
    train_gpu = torch.as_tensor(train, dtype=torch.float16, device=device)
    val_gpu = torch.as_tensor(val, dtype=torch.float16, device=device)
    train_labels = torch.as_tensor(labels[0], dtype=torch.long, device=device)
    val_labels = torch.as_tensor(labels[1], dtype=torch.long, device=device)
    LOGGER.info("GPU preload: train %.1f MB, val %.1f MB",
                train_gpu.numel() * train_gpu.element_size() / 2**20,
                val_gpu.numel() * val_gpu.element_size() / 2**20)
    return train_gpu, val_gpu, train_labels, val_labels


def preloaded_batches(x, y, batch_size, seed, augment=True):
    """Yield deterministic GPU batches: x ``(B,80,352)`` fp32 and y ``(B,)`` long."""
    if x.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != 352:
        raise ValueError(f"expected x=(N,80,352), y=(N,), got {x.shape}, {y.shape}")
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    order = torch.randperm(x.shape[0], generator=generator, device=x.device)
    for start in range(0, x.shape[0], batch_size):
        indices = order[start:start + batch_size]
        batch = x.index_select(0, indices).float()
        if augment:
            shifts = torch.randint(-20, 21, (batch.shape[0],), generator=generator,
                                   device=x.device)
            positions = torch.arange(batch.shape[-1], device=x.device)
            gather = (positions[None, :] - shifts[:, None]) % batch.shape[-1]
            batch = batch.gather(2, gather[:, None, :].expand(-1, batch.shape[1], -1))
        yield batch[:, None], y.index_select(0, indices)


def _profile_start(device):
    """Return a synchronised profiling timestamp."""
    _sync(device)
    return time.perf_counter()


def main():
    """Train the evaluation-only classifier on cached train data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "weights/classifier.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-interval", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(0)
    np.random.seed(0)

    norm = json.loads((args.cache / "norm.json").read_text())
    train = normalise(np.load(args.cache / "train_logmel.npy").astype(np.float32), **norm)
    val = normalise(np.load(args.cache / "val_logmel.npy").astype(np.float32), **norm)
    labels = [np.load(args.cache / f"{split}_labels.npy") for split in ("train", "val")]
    model = SmallCNN().to(device)
    LOGGER.info("model device: %s", next(model.parameters()).device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_preload = device.type == "cuda" and not args.no_preload
    if args.no_preload and device.type == "cuda":
        LOGGER.warning("GPU preload disabled; using DataLoader and host-to-device copies")
    if device.type == "cpu" and not args.no_preload:
        LOGGER.warning("CUDA unavailable; using CPU DataLoader path")

    if use_preload:
        train_x, val_x, train_y, val_y = preload_arrays(train, val, labels, device)
        loader = None
    else:
        dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(labels[0]).long())
        loader = DataLoader(dataset, 64, shuffle=True, num_workers=2)
        val_x = torch.from_numpy(val)
        val_y = torch.from_numpy(labels[1]).long()
        train_x = train_y = None

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        if use_preload:
            batches = preloaded_batches(train_x, train_y, 64, epoch, augment=True)
        else:
            batches = (
                (x[:, None].float(), y)
                for x, y in loader
            )
        interval_start = _profile_start(device) if args.profile else None
        batch_iter = iter(batches)
        iteration = 0
        while True:
            batch_start = _profile_start(device) if args.profile and iteration == 0 else None
            try:
                x, y = next(batch_iter)
            except StopIteration:
                break
            iteration += 1
            if not use_preload:
                shifts = torch.randint(-20, 21, (x.shape[0],))
                positions = torch.arange(x.shape[-1])
                gather = (positions[None, :] - shifts[:, None]) % x.shape[-1]
                x = x.gather(3, gather[:, None, None, :].expand(-1, 1, x.shape[2], -1))
            x, y = x.to(device), y.to(device)
            if args.profile and iteration == 1:
                _sync(device)
                LOGGER.info("first batch: shape=%s device=%s production=%.3fs",
                            tuple(x.shape), x.device, time.perf_counter() - batch_start)
            loss = torch.nn.functional.cross_entropy(model(x)[0], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.profile and iteration % args.profile_interval == 0:
                _sync(device)
                LOGGER.info("epoch=%d iteration=%d interval_seconds=%.3f peak_gpu_mb=%.1f",
                            epoch + 1, iteration, time.perf_counter() - interval_start,
                            torch.cuda.max_memory_allocated(device) / 2**20
                            if device.type == "cuda" else 0.0)
                interval_start = _profile_start(device)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x[:, None].float().to(device))[0]
            pred = val_logits.argmax(1)
        acc = float((pred == val_y.to(device)).float().mean().item())
        if acc > best:
            best = acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "acc": acc}, args.out)
    LOGGER.info("best_val_accuracy=%.4f", best)


if __name__ == "__main__":
    main()
