#!/usr/bin/env python3
"""Factor guided encoder for PRINT."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "third_party/URFM")]
from models_mae import MaskedAutoencoderViT
ENCODER_PREFIXES = ("cls_token", "pos_embed", "patch_embed.", "blocks.", "norm.")
DECODER_PREFIXES = (
    "mask_token",
    "decoder_embed.",
    "decoder_pos_embed",
    "decoder_blocks.",
    "decoder_norm.",
    "decoder_pred.",
)
REPORT_METRICS = (
    "factor_spearman_mean",
    "profile_pcc_mean",
    "variance_ratio_mean",
    "mse_z",
    "mae_z",
)
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_encoder_name(name):
    return name.startswith(ENCODER_PREFIXES)


def is_decoder_name(name):
    return name.startswith(DECODER_PREFIXES)


class FactorGuidedURFM(MaskedAutoencoderViT):
    def __init__(self, full_mae_init):
        super().__init__(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=768,
            decoder_depth=8,
            decoder_num_heads=12,
            mlp_ratio=4,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
            norm_pix_loss=True,
        )
        if full_mae_init is not None:
            checkpoint = torch.load(full_mae_init, map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint["model"]
            status = self.load_state_dict(state, strict=True)
            self.mae_init_status = f"strict:{full_mae_init}:{status}"
        self.factor_layer_logits = nn.Parameter(torch.zeros(4))
        self.factor_simple = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 13),
        )


def set_encoder_depth(model, depth):
    for name, parameter in model.named_parameters():
        if not is_encoder_name(name):
            continue
        if name.startswith("pos_embed"):
            parameter.requires_grad = False
        elif depth >= 12:
            parameter.requires_grad = True
        elif name.startswith("norm."):
            parameter.requires_grad = depth > 0
        elif name.startswith("blocks."):
            block = int(name.split(".")[1])
            parameter.requires_grad = block >= 12 - depth
        else:
            parameter.requires_grad = False


def replace_none(grads, parameters):
    return [torch.zeros_like(parameter) if grad is None else grad for grad, parameter in zip(grads, parameters)]
