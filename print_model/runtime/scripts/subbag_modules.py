#!/usr/bin/env python3
"""Subbag modules for PRINT."""
from __future__ import annotations

import torch


def split_patient_subbags(
    image_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two disjoint, non-empty masks for every eligible patient.

    The returned eligibility mask is false for zero/one-image patients.  Those
    rows are all-false in both sub-bag masks, which makes accidental use easy
    to detect and keeps the single-image auxiliary path exact-zero.
    """
    if image_mask.ndim != 2:
        raise ValueError("image_mask must have shape [patients, images]")
    valid = image_mask.bool()
    a = torch.zeros_like(valid)
    b = torch.zeros_like(valid)
    eligible = valid.sum(dim=1) >= 2
    for row in eligible.nonzero(as_tuple=False).flatten().tolist():
        indices = valid[row].nonzero(as_tuple=False).flatten()
        generator_device = None if generator is None else torch.device(generator.device)
        permutation_device = indices.device if generator_device is None else generator_device
        order = torch.randperm(indices.numel(), generator=generator, device=permutation_device)
        if order.device != indices.device:
            order = order.to(indices.device)
        indices = indices[order]
        a[row, indices[0::2]] = True
        b[row, indices[1::2]] = True
    return a, b, eligible
