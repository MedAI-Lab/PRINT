#!/usr/bin/env python3
"""Patch query modules for PRINT."""
from __future__ import annotations

from typing import Any, Sequence

import torch


def aggregate_patient_patch_residual(
    image_delta: torch.Tensor,
    patient_index: torch.Tensor,
    *,
    patient_count: int | None = None,
) -> torch.Tensor:
    """Mean aggregate per-image patch evidence without assuming other branches."""

    if image_delta.ndim != 2 or image_delta.shape[0] < 1:
        raise ValueError("image_delta must have shape [images,F]")
    if patient_index.ndim != 1 or patient_index.numel() != image_delta.shape[0]:
        raise ValueError("patient_index must have one entry per image")
    if patient_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("patient_index must be integer")
    inferred = int(patient_index.max().item()) + 1
    count = inferred if patient_count is None else int(patient_count)
    if count < inferred or patient_index.min().item() < 0:
        raise ValueError("patient indices are outside patient_count")
    result = image_delta.new_zeros((count, image_delta.shape[1]))
    denominator = image_delta.new_zeros((count, 1))
    result.index_add_(0, patient_index.long(), image_delta)
    denominator.index_add_(
        0, patient_index.long(), image_delta.new_ones((image_delta.shape[0], 1))
    )
    if (denominator == 0).any():
        raise ValueError("patient_count includes a patient with no image")
    return result / denominator


def patch_residual_activation_audit(
    patient_patch_delta: torch.Tensor,
    train_factor_std: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    """Audit patch residual RMS as a fraction of train-fold target SD."""

    if patient_patch_delta.ndim != 2 or patient_patch_delta.shape[0] < 1:
        raise ValueError("patient_patch_delta must have shape [B,F]")
    factor_std = torch.as_tensor(
        train_factor_std,
        device=patient_patch_delta.device,
        dtype=torch.float32,
    )
    if factor_std.shape != (patient_patch_delta.shape[1],):
        raise ValueError("train_factor_std must have shape [F]")
    if not torch.isfinite(factor_std).all() or (factor_std <= 0).any():
        raise ValueError("train_factor_std must be finite and positive")
    rms = torch.sqrt(patient_patch_delta.float().square().mean(0))
    fraction = rms / factor_std
    return {
        "residual_rms_fraction_mean": float(fraction.detach().mean().cpu()),
        "residual_rms_fraction_per_factor": fraction.detach().cpu().tolist(),
        "residual_rms_fraction_minimum": float(fraction.detach().min().cpu()),
        "residual_rms_fraction_maximum": float(fraction.detach().max().cpu()),
        "finite": bool(torch.isfinite(fraction).all().item()),
    }
