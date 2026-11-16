#!/usr/bin/env python3
"""Factor spatial modules for PRINT."""
from __future__ import annotations

import math
import copy
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialParameterInitializer(nn.Module):
    """Initialize scorer and residual tensors in the order used for training."""
    def __init__(
        self,
        token_dim: int = 768,
        *, num_factors: int = 13,
        score_hidden_dim: int = 128,
        residual_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.score_norm = nn.LayerNorm(token_dim)
        self.score_hidden = nn.Linear(token_dim, score_hidden_dim)
        self.score_output = nn.Linear(score_hidden_dim, num_factors)
        self.context_projection = nn.Linear(token_dim, residual_hidden_dim)
        self.residual_weight = nn.Parameter(torch.empty(num_factors, residual_hidden_dim))
        self.residual_bias = nn.Parameter(torch.zeros(num_factors))
        nn.init.trunc_normal_(self.score_output.weight, std=0.02)
        nn.init.normal_(self.residual_weight, std=1e-3)


def attention_flip_consistency_loss(
    attention: torch.Tensor,
    flipped_attention: torch.Tensor,
    *,
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """Mean squared difference between aligned per-factor soft attention maps."""
    if attention.shape != flipped_attention.shape or attention.ndim != 3:
        raise ValueError("Attention maps must match [I,F,P]")
    height, width = map(int, patch_grid)
    if height * width != attention.shape[-1]:
        raise ValueError("patch_grid mismatch")
    mapped = torch.flip(
        flipped_attention.reshape(*flipped_attention.shape[:2], height, width), dims=(-1,)
    ).flatten(2)
    per_factor = (attention.float() - mapped.float()).square().mean((0, 2))
    return per_factor.mean()


class PatientFactorSpatialGate(nn.Module):
    """Apply a shared context MLP to each patient and factor."""
    def __init__(
        self, context_dim: int = 768, *, num_factors: int = 13,
        hidden_dim: int = 32, initial: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0 < initial < 1 or min(context_dim, num_factors, hidden_dim) < 1:
            raise ValueError("invalid spatial gate parameters")
        self.context_dim, self.num_factors = int(context_dim), int(num_factors)
        self.net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(initial / (1 - initial)))



def aggregate_patient_factor_context(
    image_factor_context: torch.Tensor,
    patient_index: torch.Tensor,
    *, patient_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean aggregate TOPK image contexts to patient×factor contexts."""
    if image_factor_context.ndim != 3 or image_factor_context.shape[0] < 1:
        raise ValueError("image_factor_context must be [images,F,D]")
    if patient_index.ndim != 1 or patient_index.numel() != image_factor_context.shape[0]:
        raise ValueError("patient_index must have one entry per image")
    if patient_index.dtype not in (torch.int32, torch.int64) or patient_index.min() < 0:
        raise ValueError("patient_index must be nonnegative integer")
    inferred = int(patient_index.max().item()) + 1
    count = inferred if patient_count is None else int(patient_count)
    if count != inferred:
        raise ValueError("patient_index must densely cover exactly patient_count patients")
    contexts = []
    counts = []
    for patient in range(count):
        values = image_factor_context[patient_index == patient].float()
        if values.shape[0] == 0:
            raise ValueError("every patient must have at least one image")
        contexts.append(values.mean(0))
        counts.append(values.shape[0])
    return torch.stack(contexts), torch.tensor(counts, device=image_factor_context.device)


def randomized_spatial_copy(
    module: nn.Module,
    *, seed: int,
) -> nn.Module:
    """Deep-copy a trained module and rerandomize only its query/scorer path."""
    randomized = copy.deepcopy(module)
    devices = [p.device for p in randomized.parameters() if p.is_cuda]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        randomized.score_norm.reset_parameters()
        randomized.score_hidden.reset_parameters()
        randomized.score_output.reset_parameters()
        nn.init.trunc_normal_(randomized.score_output.weight, std=0.02)
    return randomized


def topk_flip_audit(
    hard_identity: torch.Tensor,
    hard_hflip: torch.Tensor,
    score_identity: torch.Tensor,
    score_hflip: torch.Tensor,
    *, patch_grid: tuple[int, int],
) -> dict[str, Any]:
    if hard_identity.shape != hard_hflip.shape or score_identity.shape != score_hflip.shape:
        raise ValueError("identity/hflip tensors must match")
    h, w = patch_grid
    if h * w != hard_identity.shape[-1]:
        raise ValueError("patch_grid mismatch")
    mapped_hard = torch.flip(hard_hflip.reshape(*hard_hflip.shape[:2], h, w), (-1,)).flatten(2)
    mapped_score = torch.flip(score_hflip.reshape(*score_hflip.shape[:2], h, w), (-1,)).flatten(2)
    intersection = (hard_identity.bool() & mapped_hard.bool()).sum(-1).float()
    union = (hard_identity.bool() | mapped_hard.bool()).sum(-1).float().clamp_min(1)
    cosine = F.cosine_similarity(score_identity.float(), mapped_score.float(), dim=-1)
    return {
        "flip_topk_jaccard": float((intersection / union).detach().mean().cpu()),
        "flip_score_cosine": float(cosine.detach().mean().cpu()),
        "finite": bool(torch.isfinite(cosine).all().item()),
    }


def spatial_randomization_audit(
    trained_map: torch.Tensor, randomized_map: torch.Tensor,
    *, reference_similarity: float = 1.0,
) -> dict[str, Any]:
    if trained_map.shape != randomized_map.shape or trained_map.ndim != 3:
        raise ValueError("spatial maps must match [I,F,P]")
    cosine = F.cosine_similarity(trained_map.float(), randomized_map.float(), dim=-1)
    similarity = float(cosine.detach().mean().cpu())
    return {
        "randomized_similarity": similarity,
        "randomized_similarity_drop": float(reference_similarity) - similarity,
        "finite": bool(torch.isfinite(cosine).all().item()),
    }


def spatial_parameter_displacement(
    module: nn.Module, initial_state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    current = dict(module.named_parameters())
    rows = {}
    total_sq = 0.0
    initial_sq = 0.0
    for name, initial in initial_state.items():
        if name not in current or current[name].shape != initial.shape:
            raise ValueError(f"initial parameter mismatch: {name}")
        difference = current[name].detach().float().cpu() - initial.detach().float().cpu()
        rows[name] = float(difference.norm())
        total_sq += float(difference.square().sum())
        initial_sq += float(initial.detach().float().square().sum())
    return {
        "per_parameter_l2": rows,
        "total_l2": total_sq**0.5,
        "relative_l2": total_sq**0.5 / max(initial_sq**0.5, 1e-12),
    }
