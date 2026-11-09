#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foley.constants import NULL_ID
from foley.data import MelDataset, load_normalised_cache
from foley.diffusion import Diffusion
from foley.ema import EMA
from foley.model import UNet

LOGGER = logging.getLogger("foley.train")


def save_ckpt(path, ema_model, cfg, norm, stats, step):
    """Atomically save EMA model metadata in a weights-only-compatible checkpoint."""
    payload = {"model": ema_model.state_dict(), "cfg": cfg, "norm": norm,
               "ctrl_stats": stats, "step": int(step), "gen_defaults": cfg["gen_defaults"]}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def val_loss(model, loader, diffusion, device, use_controls=False, repeats=2):
    """Evaluate fixed-noise validation epsilon loss on ``device``."""
    generator = torch.Generator(device=device).manual_seed(123)
    total = count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                else:
                    x, y = batch["x"].to(device), batch["y"].to(device)
                ctrl = None if isinstance(batch, (tuple, list)) else (
                    batch["u"].to(device) if use_controls else None)
                mask = None if isinstance(batch, (tuple, list)) else (
                    batch["mask"].to(device) if use_controls else None)
                value = diffusion.loss(model, x, y, ctrl, mask, generator)
                total += float(value) * len(y)
                count += len(y)
    model.train(was_training)
    return total / max(count, 1)


def _device(value):
    """Resolve the requested device without silently falling back."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def _gpu_cache(cache, split, norm, device):
    """Load one normalised cache split as fp16 tensors on the selected GPU."""
    x, y = load_normalised_cache(cache, split, norm)
    return (torch.as_tensor(x, dtype=torch.float16, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device))


def _gpu_batches(x, y, batch_size, seed):
    """Yield shuffled core batches as ``(B,1,80,352)`` tensors on one GPU."""
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    order = torch.randperm(len(y), generator=generator, device=x.device)
    for start in range(0, len(y), batch_size):
        indices = order[start:start + batch_size]
        yield x.index_select(0, indices).float()[:, None], y.index_select(0, indices)


def main():
    """Train core or control-conditioned diffusion with resumable checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "ckpts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--preload-gpu", dest="preload_gpu", action="store_true")
    parser.add_argument("--no-preload", dest="preload_gpu", action="store_false")
    parser.set_defaults(preload_gpu=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    device = _device(args.device)
    torch.manual_seed(int(train_cfg["seed"]))
    norm = json.loads((Path(data_cfg["cache"]) / "norm.json").read_text())
    use_controls = bool(cfg["model"]["use_controls"])
    stats = (json.loads((Path(data_cfg["cache"]) / "ctrl_stats.json").read_text())
             if use_controls else None)
    use_preload = args.preload_gpu if args.preload_gpu is not None else (
        device == "cuda" and bool(train_cfg.get("preload_gpu", True)))
    use_preload = bool(use_preload and device == "cuda" and not use_controls
                       and float(train_cfg.get("shift_prob", 0.0)) == 0.0)
    if args.preload_gpu and (use_controls or float(train_cfg.get("shift_prob", 0.0)) != 0.0):
        LOGGER.warning("GPU preload disabled for control/shift training; using DataLoader path")
    if args.preload_gpu is False and device == "cuda":
        LOGGER.warning("GPU preload disabled; using DataLoader and host-to-device copies")
    train_ds = MelDataset(data_cfg["cache"], "train", norm, stats, True,
                          train_cfg["shift_prob"], train_cfg["ctrl_mask_p"])
    val_ds = MelDataset(data_cfg["cache"], "val", norm, stats, False)
    workers = int(train_cfg["workers"])
    train_dl = None
    val_dl = None
    if use_preload:
        train_x, train_y = _gpu_cache(data_cfg["cache"], "train", norm, device)
        val_x, val_y = _gpu_cache(data_cfg["cache"], "val", norm, device)
        required = (train_x.numel() + val_x.numel()) * 2
        free, _ = torch.cuda.mem_get_info(device)
        if required > int(free * 0.8):
            LOGGER.warning("GPU cache requires %.1f MB but only %.1f MB headroom is available; using DataLoader",
                           required / 2**20, free / 2**20)
            use_preload = False
            del train_x, train_y, val_x, val_y
            torch.cuda.empty_cache()
        else:
            LOGGER.info("GPU preload: train+val %.1f MB", required / 2**20)
            val_dl = DataLoader(TensorDataset(val_x.float()[:, None], val_y),
                                64, shuffle=False, num_workers=0)
    if not use_preload:
        train_dl = DataLoader(train_ds, int(train_cfg["batch"]), shuffle=True, drop_last=True,
                              num_workers=workers, pin_memory=device == "cuda",
                              persistent_workers=workers > 0)
        val_dl = DataLoader(val_ds, 64, num_workers=workers, pin_memory=device == "cuda")
    model = UNet(**cfg["model"]).to(device)
    LOGGER.info("params(M): %.3f", sum(p.numel() for p in model.parameters()) / 1e6)
    ema = EMA(model, float(train_cfg["ema"]))
    diffusion = Diffusion(**cfg["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, int(train_cfg["warmup"]))))
    scaler = torch.amp.GradScaler('cuda', enabled=device == "cuda")
    step = 0
    if train_cfg.get("init_from"):
        initial = torch.load(train_cfg["init_from"], map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        if unexpected or (use_controls and any(not name.startswith("ctrl.") for name in missing)):
            raise RuntimeError(f"unexpected init checkpoint keys: missing={missing}, unexpected={unexpected}")
    resume_path = None if args.resume == "none" else (
        args.out / "last.pt" if args.resume == "auto" else Path(args.resume))
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        ema.shadow.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg["steps"])
    while step < max_steps:
        batches = (_gpu_batches(train_x, train_y, int(train_cfg["batch"]), step)
                   if use_preload else train_dl)
        for batch in batches:
            if use_preload:
                x, y = batch
                ctrl = mask = None
            else:
                x, y = batch["x"].to(device), batch["y"].to(device)
                ctrl = batch["u"].to(device) if use_controls else None
                mask = batch["mask"].to(device) if use_controls else None
            dropped = torch.rand(len(y), device=device) < float(train_cfg["p_uncond"])
            y = torch.where(dropped, torch.full_like(y, NULL_ID), y)
            if use_controls:
                mask = mask * (~dropped)[:, None]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = diffusion.loss(model, x, y, ctrl, mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step = (not scaler.is_enabled()
                              or scaler.get_scale() >= scale_before)
            if optimizer_step:
                scheduler.step()
                ema.update(model)
            step += 1
            if step % 100 == 0:
                LOGGER.info("step=%d loss=%.5f lr=%.3g", step, loss.detach().item(),
                            scheduler.get_last_lr()[0])
            if step % int(train_cfg["val_every"]) == 0:
                LOGGER.info("val_loss=%.5f", val_loss(ema.shadow, val_dl, diffusion, device, use_controls))
            if step % int(train_cfg["ckpt_every"]) == 0:
                save_ckpt(args.out / f"step_{step:06d}.pt", ema.shadow, cfg, norm, stats, step)
            if step % 1000 == 0:
                last = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "step": step}
                temporary = args.out / "last.pt.tmp"
                args.out.mkdir(parents=True, exist_ok=True)
                torch.save(last, temporary)
                os.replace(temporary, args.out / "last.pt")
            if step >= max_steps:
                break
    final_state = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "step": step}
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out / "last.pt.tmp"
    torch.save(final_state, temporary)
    os.replace(temporary, args.out / "last.pt")


if __name__ == "__main__":
    main()
