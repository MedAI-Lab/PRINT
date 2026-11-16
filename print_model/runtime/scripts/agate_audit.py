#!/usr/bin/env python3
"""Agate audit for PRINT."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _finite_1d(value: torch.Tensor, name: str, length: int | None = None) -> torch.Tensor:
    result = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} patient values")
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return result


def _patient_norm(value: torch.Tensor, name: str, length: int) -> torch.Tensor:
    result = torch.as_tensor(value).detach().float().cpu()
    if result.ndim < 2 or result.shape[0] != length:
        raise ValueError(f"{name} must have one vector per patient")
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return torch.linalg.vector_norm(result.reshape(length, -1), dim=1)


def _average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    sorted_value = value[order]
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and sorted_value[end] == sorted_value[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_or_none(first: torch.Tensor, second: torch.Tensor) -> float | None:
    """Tie-aware Spearman correlation, returning None for undefined constants."""
    x = _finite_1d(first, "first").numpy().astype(np.float64, copy=False)
    y = _finite_1d(second, "second", len(x)).numpy().astype(np.float64, copy=False)
    if len(x) < 2:
        return None
    xr, yr = _average_ranks(x), _average_ranks(y)
    if np.ptp(xr) == 0.0 or np.ptp(yr) == 0.0:
        return None
    answer = float(np.corrcoef(xr, yr)[0, 1])
    return answer if np.isfinite(answer) else None


def _distribution(value: torch.Tensor) -> dict[str, float | int | None]:
    if value.numel() == 0:
        return {key: None for key in ("min", "median", "mean", "p90", "p95", "max")} | {"n": 0}
    value = value.float()
    return {
        "n": int(value.numel()),
        "min": float(value.min()),
        "median": float(value.median()),
        "mean": float(value.mean()),
        "p90": float(torch.quantile(value, 0.90)),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
    }


class AGATEPatientAuditAccumulator:
    """Accumulate detached patient-level AGATE diagnostics across batches."""

    def __init__(
        self,
        *,
        maximum_alpha: float = 0.15,
        high_threshold: float = 0.12,
        saturation_threshold: float = 0.145,
        single_tolerance: float = 0.0,
    ) -> None:
        if not 0 < high_threshold < saturation_threshold <= maximum_alpha:
            raise ValueError("invalid AGATE alpha audit thresholds")
        if single_tolerance < 0:
            raise ValueError("single_tolerance must be non-negative")
        self.maximum_alpha = float(maximum_alpha)
        self.high_threshold = float(high_threshold)
        self.saturation_threshold = float(saturation_threshold)
        self.single_tolerance = float(single_tolerance)
        self._rows: dict[str, list[torch.Tensor]] = {
            name: [] for name in (
                "alpha", "image_count", "sd_norm", "max_minus_mean_norm",
                "raw_residual_norm", "correction_norm", "single_prediction_error",
                "single_alpha", "single_correction_norm", "orthogonality_error",
            )
        }

    def update(
        self,
        *,
        alpha: torch.Tensor,
        image_count: torch.Tensor,
        population_sd: torch.Tensor,
        max_minus_mean: torch.Tensor,
        raw_residual: torch.Tensor,
        correction: torch.Tensor,
        prediction: torch.Tensor,
        base_prediction: torch.Tensor,
        orthogonality_error: torch.Tensor,
    ) -> None:
        counts = _finite_1d(image_count, "image_count")
        if (counts < 1).any() or not torch.equal(counts, counts.round()):
            raise ValueError("image_count must contain positive integers")
        size = len(counts)
        a = _finite_1d(alpha, "alpha", size)
        if (a < 0).any() or (a > self.maximum_alpha + 1e-7).any():
            raise ValueError("AGATE alpha must lie between zero and the configured maximum.")
        correction_norm = _patient_norm(correction, "correction", size)
        prediction_error = _patient_norm(
            torch.as_tensor(prediction).detach().float().cpu()
            - torch.as_tensor(base_prediction).detach().float().cpu(),
            "prediction-minus-base", size,
        )
        orth = None if orthogonality_error is None else _finite_1d(orthogonality_error, "orthogonality_error", size).abs()
        multi = counts > 1
        single = ~multi
        values = {
            "alpha": a[multi],
            "image_count": counts[multi],
            "sd_norm": _patient_norm(population_sd, "population_sd", size)[multi],
            "max_minus_mean_norm": _patient_norm(max_minus_mean, "max_minus_mean", size)[multi],
            "raw_residual_norm": _patient_norm(raw_residual, "raw_residual", size)[multi],
            "correction_norm": correction_norm[multi],
            "single_prediction_error": prediction_error[single],
            "single_alpha": a[single],
            "single_correction_norm": correction_norm[single],
            "orthogonality_error": orth,
        }
        for name, value in values.items():
            if value is not None:
                self._rows[name].append(value)

    def finalize(self) -> dict[str, Any]:
        combined = {
            name: torch.cat(parts) if parts else torch.empty(0)
            for name, parts in self._rows.items()
        }
        alpha = combined["alpha"]
        counts = combined["image_count"]
        strata = {}
        for label, mask in (
            ("2_images", counts == 2),
            ("3_images", counts == 3),
            ("4plus_images", counts >= 4),
        ):
            strata[label] = _distribution(alpha[mask])
        correlations = {
            "alpha_vs_image_count": spearman_or_none(alpha, counts),
            "alpha_vs_population_sd_norm": spearman_or_none(alpha, combined["sd_norm"]),
            "alpha_vs_max_minus_mean_norm": spearman_or_none(alpha, combined["max_minus_mean_norm"]),
            "alpha_vs_raw_residual_norm": spearman_or_none(alpha, combined["raw_residual_norm"]),
            "alpha_vs_correction_norm": spearman_or_none(alpha, combined["correction_norm"]),
        }
        single_error = combined["single_prediction_error"]
        single_alpha = combined["single_alpha"]
        single_correction = combined["single_correction_norm"]
        single_max = max(
            (float(value.max()) if value.numel() else 0.0)
            for value in (single_error, single_alpha, single_correction)
        )
        orth = combined["orthogonality_error"]
        return {
            "multi_patient_count": int(alpha.numel()),
            "single_patient_count": int(single_error.numel()),
            "alpha_distribution": _distribution(alpha),
            "alpha_fraction_ge_0p12": float((alpha >= self.high_threshold).float().mean()) if alpha.numel() else None,
            "alpha_fraction_ge_0p145": float((alpha >= self.saturation_threshold).float().mean()) if alpha.numel() else None,
            "alpha_by_image_count": strata,
            "spearman": correlations,
            "single_identity": {
                "tolerance": self.single_tolerance,
                "max_abs_equivalent_error": single_max,
                "pass": single_max <= self.single_tolerance,
                "prediction_error_max": float(single_error.max()) if single_error.numel() else 0.0,
                "alpha_max": float(single_alpha.max()) if single_alpha.numel() else 0.0,
                "correction_norm_max": float(single_correction.max()) if single_correction.numel() else 0.0,
            },
            "orthogonality": {
                "max_abs": float(orth.max()) if orth.numel() else None,
                "all_finite": bool(torch.isfinite(orth).all()) if orth.numel() else None,
            },
        }


@torch.no_grad()
def count_counterfactual_audit(
    *,
    patient_gate: torch.nn.Module,
    heterogeneity_descriptor: torch.Tensor,
    observed_image_count: torch.Tensor,
    maximum_alpha: float,
    replacement_counts: tuple[int, ...] = (2, 3, 4),
) -> dict[str, Any]:
    """Read-only gate audit holding image-derived heterogeneity fixed.

    ``heterogeneity_descriptor`` is exactly ``[population_sd,max_minus_mean]``;
    the count coordinate is appended here.  No synthetic image or prediction is
    constructed, and the result must never be used for checkpoint selection.
    """
    descriptor = torch.as_tensor(heterogeneity_descriptor).detach().float()
    if descriptor.ndim != 2 or not torch.isfinite(descriptor).all():
        raise ValueError("heterogeneity_descriptor must be a finite matrix")
    counts = torch.as_tensor(observed_image_count, device=descriptor.device).detach().float().reshape(-1, 1)
    if len(counts) != len(descriptor) or (counts < 2).any():
        raise ValueError("count counterfactual is defined only for multi-image patients")
    device = next(patient_gate.parameters()).device
    descriptor, counts = descriptor.to(device), counts.to(device)

    def alpha_for(count: torch.Tensor) -> torch.Tensor:
        return float(maximum_alpha) * torch.sigmoid(patient_gate(torch.cat([descriptor, count], dim=1).float())).reshape(-1)

    observed = alpha_for(counts)
    scenarios = {}
    for replacement in replacement_counts:
        if replacement < 2:
            raise ValueError("replacement counts must preserve multi-image status")
        counterfactual = alpha_for(torch.full_like(counts, float(replacement)))
        difference = counterfactual - observed
        scenarios[str(replacement)] = {
            "replacement_count": int(replacement),
            "alpha_distribution": _distribution(counterfactual.cpu()),
            "difference_distribution": _distribution(difference.cpu()),
            "mean_absolute_change": float(difference.abs().mean().cpu()),
            "maximum_absolute_change": float(difference.abs().max().cpu()),
        }
    return {
        "audit_only_not_for_selection": True,
        "patient_count": int(len(observed)),
        "observed_alpha_distribution": _distribution(observed.cpu()),
        "observed_alpha_vs_count_spearman": spearman_or_none(observed.cpu(), counts.cpu()),
        "scenarios": scenarios,
    }


__all__ = [
    "AGATEPatientAuditAccumulator",
    "count_counterfactual_audit",
    "spearman_or_none",
]
