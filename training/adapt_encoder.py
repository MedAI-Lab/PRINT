#!/usr/bin/env python3
"""Adapt the URFM encoder to thyroid ultrasound with masked autoencoding."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "print_model/runtime"
sys.path[:0] = list(map(str, (RUNTIME/"scripts", RUNTIME/"third_party/URFM", RUNTIME)))
sys.path.insert(0, str(ROOT))
from masked_encoder_data import Images, encoder_state, manifest, seed_all, sha256
from models_mae import MaskedAutoencoderViT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--schedule-epochs", type=int, default=300)
    ap.add_argument("--capture-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.875e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available() or not a.device.startswith("cuda"):
        raise RuntimeError("Stage 1 training requires a CUDA device.")
    if not 1 <= a.capture_epoch <= a.schedule_epochs:
        raise ValueError("capture-epoch must be within the selected schedule")
    if not 0 < a.mask_ratio < 1:
        raise ValueError("mask-ratio must be in (0,1)")

    seed_all(a.seed)
    device = torch.device(a.device)
    nodule_count, images = manifest(a.data.resolve())
    if len(images) < a.batch_size:
        raise ValueError("Stage 1 requires enough images for one full batch; reduce --batch-size")
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(
        Images(images),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    model = MaskedAutoencoderViT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=12,
                                 mlp_ratio=4, norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6),
                                 norm_pix_loss=True)
    checkpoint = torch.load(a.init, map_location="cpu", weights_only=True)
    state = checkpoint.get("ema_state_dict", checkpoint.get("model"))
    load_status = str(model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True))
    del checkpoint, state
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    history = []
    start = time.time()
    mask_count = int(round(196 * a.mask_ratio))

    for epoch in range(1, a.capture_epoch + 1):
        if epoch <= a.warmup:
            scale = epoch / a.warmup
        else:
            scale = 0.01 + 0.99 * 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (epoch - a.warmup)
                    / max(1, a.schedule_epochs - a.warmup)
                )
            )
        for group in optimizer.param_groups:
            group["lr"] = a.lr * scale

        model.train()
        total = loss_sum = steps = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            mask = torch.zeros(x.shape[0], 196, dtype=torch.bool, device=device)
            for i in range(x.shape[0]):
                mask[i, torch.randperm(196, device=device)[:mask_count]] = True
            with torch.autocast("cuda"):
                output = model(x, mask)
                loss = model.forward_loss(
                    x,
                    output["pix_pred"][:, -output["mask_num"] :],
                    output["mask"],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += len(x)
            loss_sum += float(loss.detach()) * len(x)
            steps += 1
            if a.smoke_steps and steps >= a.smoke_steps:
                break
        row = {"epoch": epoch, "loss": loss_sum / total, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if a.smoke_steps:
            break

    official_sha = sha256(a.init)
    checkpoint = {
        "model": model.state_dict(),
        "encoder": encoder_state(model),
        "epoch": history[-1]["epoch"],
        "seed": a.seed,
        "lr": a.lr,
        "schedule_epochs": a.schedule_epochs,
        "capture_epoch": a.capture_epoch,
        "mask_ratio": a.mask_ratio,
        "official_sha256": official_sha,
        "load_status": load_status,
    }
    torch.save(checkpoint, out / "full_mae_checkpoint.pt")
    audit = {
        "status": "complete" if not a.smoke_steps else "smoke_complete",
        "parameters": vars(a),
        "nodules": nodule_count,
        "images": len(images),
        "official_sha256": official_sha,
        "load_status": load_status,
        "cuda": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "elapsed_seconds": time.time() - start,
    }
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
