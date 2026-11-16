#!/usr/bin/env python3
"""Multiview aggregation for PRINT."""
from __future__ import annotations


import torch
import torch.nn as nn


def _validate_patient_features(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("features must have shape [patients, images, feature_dim]")
    if features.shape[-1] != feature_dim:
        raise ValueError(f"expected feature width {feature_dim}, got {features.shape[-1]}")
    if image_mask.shape != features.shape[:2]:
        raise ValueError("image_mask shape mismatch")
    if not features.is_floating_point():
        raise TypeError("features must be floating point")
    valid = image_mask.bool()
    counts = valid.sum(dim=1)
    if (counts < 1).any():
        raise ValueError("every patient must have at least one valid image")
    if not torch.isfinite(features[valid]).all():
        raise FloatingPointError("valid image features contain NaN or Inf")
    return valid, counts


def _masked_mean(
    features: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    clean = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
    return clean.sum(dim=1) / counts.to(features.dtype).unsqueeze(-1)


def _zero_from(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


class HCALPooling(nn.Module):
    """Patient-adaptive, norm-bounded heterogeneity residual pooling.

    The heterogeneity descriptor is the concatenation of population SD and
    max-minus-mean.  A patient-specific sigmoid gate scales a learned residual,
    after which an architectural projection guarantees
    ``||delta|| / ||mean|| <= maximum_relative_deviation``.  The residual output
    layer is zero initialized, giving exact mean pooling at construction while
    retaining a non-zero first-step gradient.  Single-image patients are
    permanently, exactly mean pooled for every possible parameter value.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        gate_hidden_dim: int = 32,
        maximum_relative_deviation: float = 0.025,
        activation_threshold: float = 0.005,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or gate_hidden_dim < 1:
            raise ValueError("feature and hidden dimensions must be positive")
        if not 0.0 < maximum_relative_deviation <= 1.0:
            raise ValueError("maximum_relative_deviation must be in (0, 1]")
        if not 0.0 <= activation_threshold <= maximum_relative_deviation:
            raise ValueError("activation_threshold must lie inside the deviation bound")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.maximum_relative_deviation = float(maximum_relative_deviation)
        self.activation_threshold = float(activation_threshold)
        self.eps = float(eps)

        heterogeneity_dim = 2 * self.feature_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(heterogeneity_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(heterogeneity_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        nn.init.zeros_(self.gate_output.bias)

    @property
    def residual_output(self) -> nn.Linear:
        return self.residual_mlp[-1]

    @property
    def gate_output(self) -> nn.Linear:
        return self.gate_mlp[-1]

    def _heterogeneity(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        counts: torch.Tensor,
        mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = torch.where(
            valid.unsqueeze(-1), features - mean.unsqueeze(1), torch.zeros_like(features)
        )
        variance = centered.float().square().sum(dim=1) / counts.float().unsqueeze(-1)
        multi = counts > 1
        stable_sd = torch.sqrt(variance.clamp_min(0.0) + self.eps)
        standard_deviation = torch.where(
            multi.unsqueeze(-1), stable_sd, torch.zeros_like(stable_sd)
        ).to(features.dtype)

        lowest = torch.finfo(features.dtype).min
        maximum = features.masked_fill(~valid.unsqueeze(-1), lowest).max(dim=1).values
        maximum_minus_mean = maximum - mean
        descriptor = torch.cat([standard_deviation, maximum_minus_mean], dim=-1)
        return descriptor, standard_deviation, maximum_minus_mean

    def _bounded_delta(
        self,
        raw_delta: torch.Tensor,
        mean: torch.Tensor,
        multi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw32 = raw_delta.float()
        mean_norm = torch.linalg.vector_norm(mean.float(), dim=-1, keepdim=True)
        raw_norm = torch.linalg.vector_norm(raw32, dim=-1, keepdim=True)
        allowed_norm = self.maximum_relative_deviation * mean_norm
        projection_scale = torch.minimum(
            torch.ones_like(raw_norm), allowed_norm / raw_norm.clamp_min(self.eps)
        )
        projection_scale = torch.where(mean_norm > self.eps, projection_scale, torch.zeros_like(projection_scale))
        projection_scale = torch.where(multi.unsqueeze(-1), projection_scale, torch.zeros_like(projection_scale))
        delta32 = raw32 * projection_scale
        delta_norm = torch.linalg.vector_norm(delta32, dim=-1)
        denominator = mean_norm.squeeze(-1)
        relative = torch.where(
            denominator > self.eps,
            delta_norm / denominator.clamp_min(self.eps),
            torch.zeros_like(delta_norm),
        )
        return delta32.to(mean.dtype), projection_scale.squeeze(-1), relative

    def forward(
        self,
        features: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid, counts = _validate_patient_features(features, image_mask, self.feature_dim)
        mean = _masked_mean(features, valid, counts)
        multi = counts > 1
        descriptor, standard_deviation, maximum_minus_mean = self._heterogeneity(
            features, valid, counts, mean
        )
        raw_residual = self.residual_mlp(descriptor)
        gate_logit = self.gate_mlp(descriptor)
        gate = torch.sigmoid(gate_logit)
        raw_delta = gate * raw_residual
        delta, projection_scale, relative = self._bounded_delta(raw_delta, mean, multi)
        delta = torch.where(multi.unsqueeze(-1), delta, torch.zeros_like(delta))
        patient = mean + delta

        active_relative = relative[multi]
        zero = _zero_from(patient)
        if active_relative.numel():
            median = active_relative.median()
            p95 = torch.quantile(active_relative, 0.95)
            fraction_above = (active_relative > self.activation_threshold).float().mean()
        else:
            median = zero
            p95 = zero
            fraction_above = zero
        diagnostics = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "maximum_minus_mean": maximum_minus_mean,
            "heterogeneity": descriptor,
            "raw_residual": raw_residual,
            "gate_logit": gate_logit,
            "gate": gate,
            "raw_delta": raw_delta,
            "delta": delta,
            "projection_scale": projection_scale,
            "image_count": counts,
            "multi_image_mask": multi,
            "relative_deviation": relative,
            "multi_relative_deviation_median": median,
            "multi_relative_deviation_p95": p95,
            "multi_fraction_above_0p5pct": fraction_above,
        }
        return patient, diagnostics
