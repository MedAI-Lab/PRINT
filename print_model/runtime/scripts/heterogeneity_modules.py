#!/usr/bin/env python3
"""Image-driven multiview pooling for PRINT."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from multiview_aggregation import HCALPooling




class MultiviewResidualGate(nn.Module):
    """Aggregate valid views using a learned nodule-level gate."""

    def __init__(
        self,
        *,
        feature_dim: int = 768,
        num_factors: int = 13,
        heterogeneity_hidden_dim: int = 128,
        heterogeneity_gate_hidden_dim: int = 32,
        feature_maximum_relative_deviation: float = 0.025,
        agate_initial_alpha: float = 0.025,
        agate_maximum_alpha: float = 0.15,
        agate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_factors < 1 or agate_hidden_dim < 1:
            raise ValueError("feature, factor, and AGATE hidden dimensions must be positive")
        if not 0.0 < agate_initial_alpha < agate_maximum_alpha <= 1.0:
            raise ValueError("AGATE requires 0 < initial_alpha < maximum_alpha <= 1")
        self.feature_dim = int(feature_dim)
        self.num_factors = int(num_factors)
        self.agate_initial_alpha = float(agate_initial_alpha)
        self.agate_maximum_alpha = float(agate_maximum_alpha)
        self.register_buffer("orthogonality_axis", None, persistent=True)
        pooling = dict(feature_dim=self.feature_dim, hidden_dim=heterogeneity_hidden_dim,
                       gate_hidden_dim=heterogeneity_gate_hidden_dim,
                       maximum_relative_deviation=feature_maximum_relative_deviation,
                       activation_threshold=0.005)
        self.heterogeneity_pool = HCALPooling(**pooling)
        gate_input_dim = 2 * self.feature_dim + 1
        self.patient_gate = nn.Sequential(
            nn.Linear(gate_input_dim, agate_hidden_dim),
            nn.GELU(),
            nn.Linear(agate_hidden_dim, 1),
        )
        ratio = self.agate_initial_alpha / self.agate_maximum_alpha
        initial_logit = torch.log(torch.tensor(ratio / (1.0 - ratio))).item()
        nn.init.zeros_(self.patient_gate[-1].weight)
        nn.init.constant_(self.patient_gate[-1].bias, initial_logit)

    def _validate_forward(
        self,
        image_features: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        if image_features.ndim != 3 or image_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"image_features must have shape [patients, images, {self.feature_dim}]"
            )
        if image_mask.shape != image_features.shape[:2]:
            raise ValueError("image_mask shape mismatch")
        if not image_features.is_floating_point():
            raise TypeError("image_features must be floating point")
        valid = image_mask.bool()
        counts = valid.sum(dim=1)
        if (counts < 1).any():
            raise ValueError("every patient must have at least one valid image")
        return counts

    def forward(
        self,
        image_features: torch.Tensor,
        image_mask: torch.Tensor,
        factor_head: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        counts = self._validate_forward(image_features, image_mask)
        valid = image_mask.bool()
        clean = torch.where(
            valid.unsqueeze(-1), image_features, torch.zeros_like(image_features)
        )
        mean_feature = clean.sum(dim=1) / counts.to(image_features.dtype).unsqueeze(-1)
        heterogeneity_feature, pool_diagnostics = self.heterogeneity_pool(
            image_features, image_mask
        )
        z_subbag = factor_head(mean_feature)
        z_heterogeneity = factor_head(heterogeneity_feature)
        expected = (image_features.shape[0], self.num_factors)
        if z_subbag.shape != expected or z_heterogeneity.shape != expected:
            raise ValueError(f"factor_head must return shape {expected}")
        if not torch.isfinite(z_subbag).all() or not torch.isfinite(z_heterogeneity).all():
            raise FloatingPointError("factor prediction contains NaN or Inf")

        multi = counts > 1
        raw_delta = z_heterogeneity.float() - z_subbag.float()
        gate_input = torch.cat([pool_diagnostics["heterogeneity"].float(), counts.float().unsqueeze(-1)], dim=-1)
        with torch.autocast(device_type=image_features.device.type, enabled=False):
            alpha = self.agate_maximum_alpha * self.patient_gate(gate_input).sigmoid()
        alpha = torch.where(multi.unsqueeze(-1), alpha, torch.zeros_like(alpha))
        correction = alpha * raw_delta
        return z_subbag.float() + correction, dict(
            variant="AGATE_NO_BETA", z_subbag=z_subbag, z_heterogeneity=z_heterogeneity,
            raw_prediction_delta=raw_delta, correction=correction, alpha=alpha, image_count=counts,
            multi_image_mask=multi, pool=pool_diagnostics)
