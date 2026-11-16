#!/usr/bin/env python3
"""Residual gradient objectives for PRINT."""
from __future__ import annotations

from collections.abc import Sequence

import torch



RESIDUAL_CAP_TO_BASE = 0.15


def residual_gradient_scale(epoch: int) -> float:
    """The residual gradient scale increases linearly from 0 at epoch 31 to 1 at epoch 50."""

    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 or epoch > 150:
        raise ValueError("Residual gradient epoch must be an integer in [1,150]")
    if epoch <= 31:
        return 0.0
    if epoch >= 50:
        return 1.0
    return float(epoch - 31) / 19.0


def _checked_gradient_lists(
    uniform_gradients: Sequence[torch.Tensor],
    famo_gradients: Sequence[torch.Tensor],
    profile_gradients: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    uniform, famo, profile = list(uniform_gradients), list(famo_gradients), list(profile_gradients)
    if not uniform or len(uniform) != len(famo) or len(uniform) != len(profile):
        raise ValueError("Residual gradient gradient lists must be non-empty and have equal length")
    for values in zip(uniform, famo, profile):
        reference = values[0]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("Residual gradient gradients must be tensors")
        if any(value.shape != reference.shape or value.device != reference.device for value in values[1:]):
            raise ValueError("Residual gradient gradient tensor metadata mismatch")
        if any(not torch.isfinite(value.detach()).all() for value in values):
            raise FloatingPointError("Residual gradient gradient contains NaN or infinity")
    return uniform, famo, profile
