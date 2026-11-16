#!/usr/bin/env python3
"""Agate topk modules for PRINT."""
from __future__ import annotations

import math
from typing import Any

import torch

from factor_spatial_modules import PatientFactorSpatialGate


def _finite_float(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value).float()
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return result


def _distribution(value: torch.Tensor) -> dict[str, float | int | None]:
    value = value.detach().float().reshape(-1).cpu()
    if value.numel() == 0:
        return {
            "n": 0, "min": None, "mean": None, "median": None,
            "p90": None, "max": None,
        }
    return {
        "n": int(value.numel()),
        "min": float(value.min()),
        "mean": float(value.mean()),
        "median": float(value.median()),
        "p90": float(torch.quantile(value, 0.90)),
        "max": float(value.max()),
    }


def _residual_audit(
    gate: torch.Tensor,
    image_count: torch.Tensor,
    spatial_delta: torch.Tensor | None,
) -> dict[str, Any]:
    single = image_count == 1
    multi = image_count > 1
    result: dict[str, Any] = {
        "gate_all": _distribution(gate),
        "gate_single_image": _distribution(gate[single]),
        "gate_multi_image": _distribution(gate[multi]),
    }
    if spatial_delta is None:
        result.update({
            "residual_all": None,
            "residual_single_image": None,
            "residual_multi_image": None,
        })
        return result
    delta = _finite_float(spatial_delta, "spatial_delta")
    if delta.shape != gate.shape:
        raise ValueError("spatial_delta and gate must both be [patients,factors]")
    correction = gate * delta

    def correction_summary(mask: torch.Tensor) -> dict[str, float | int | None]:
        values = correction[mask]
        if values.numel() == 0:
            return {"n": 0, "rms": None, "mean_abs": None, "max_abs": None}
        return {
            "n": int(mask.sum().item()),
            "rms": float(torch.sqrt(values.detach().square().mean()).cpu()),
            "mean_abs": float(values.detach().abs().mean().cpu()),
            "max_abs": float(values.detach().abs().max().cpu()),
        }

    result.update({
        "residual_all": correction_summary(torch.ones_like(single)),
        "residual_single_image": correction_summary(single),
        "residual_multi_image": correction_summary(multi),
    })
    return result


class AllPatientFactorSpatialGate(PatientFactorSpatialGate):
    """Apply a factor-specific spatial gate for nodules with one or more views."""

    def __init__(
        self,
        context_dim: int = 768,
        *,
        num_factors: int = 13,
        hidden_dim: int = 32,
        initial: float = 0.25,
        single_image_scale: float = 0.25,
        enable_single_image_gate: bool = True,
    ) -> None:
        super().__init__(
            context_dim=context_dim,
            num_factors=num_factors,
            hidden_dim=hidden_dim,
            initial=initial,
        )
        if not 0.0 <= single_image_scale <= 1.0:
            raise ValueError("single_image_scale must lie in [0,1]")
        self.single_image_scale = float(single_image_scale)
        self.enable_single_image_gate = bool(enable_single_image_gate)

    def forward(
        self,
        patient_factor_context: torch.Tensor,
        image_count: torch.Tensor,
        *,
        spatial_delta: torch.Tensor | None = None,
        enable_single_image_gate: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if patient_factor_context.ndim != 3 or patient_factor_context.shape[1:] != (
            self.num_factors, self.context_dim
        ):
            raise ValueError(
                f"patient_factor_context must be [B,{self.num_factors},{self.context_dim}]"
            )
        if image_count.shape != (patient_factor_context.shape[0],):
            raise ValueError("image_count must be [B]")
        counts = torch.as_tensor(image_count, device=patient_factor_context.device)
        if (counts < 1).any() or not torch.equal(counts, counts.round()):
            raise ValueError("image_count must contain positive integers")
        enabled = (
            self.enable_single_image_gate
            if enable_single_image_gate is None
            else bool(enable_single_image_gate)
        )
        with torch.autocast(device_type=patient_factor_context.device.type, enabled=False):
            raw_gate = self.net(patient_factor_context.float()).squeeze(-1).sigmoid()
            if enabled:
                singleton = raw_gate * self.single_image_scale
            else:
                singleton = torch.zeros_like(raw_gate)
            gate = torch.where((counts > 1)[:, None], raw_gate, singleton)
        if gate.requires_grad:
            gate.retain_grad()
        multi = counts > 1
        eligible = int(multi.sum().item()) * self.num_factors
        saturation = ((gate < 0.01) | (gate > 0.99)) & multi[:, None]
        audit = {
            "mode": "all_patient" if enabled else "single_image_gate_disabled",
            "single_image_gate_enabled": enabled,
            "single_image_scale": self.single_image_scale,
            "multi_image_gate_function": "sigmoid",
            "gate_saturation_fraction_multi_image": (
                float(saturation.sum().detach().cpu()) / max(1, eligible)
            ),
            **_residual_audit(gate, counts, spatial_delta),
        }
        return gate, audit


class EpochFullValidationAuditAggregator:
    """Accumulate epoch training and complete-validation statistics.

    Batch means are weighted, eligible SUBBAG patients
    are summed, and patient-level validation tensors are concatenated before
    distributions/RMS are computed.
    """

    def __init__(self) -> None:
        self._epoch_batch_weight = 0
        self._epoch_sums: dict[str, float] = {}
        self._epoch_metric_weights: dict[str, int] = {}
        self._subbag_weight = 0
        self._subbag_sum = 0.0
        self._eligible_multi = 0
        self._active_subbag_batches = 0
        self._gradient_all_finite_nonzero = True
        self._gradient_observations = 0
        self._validation_batch_count = 0
        self._agate_alpha_batch_count = 0
        self._validation: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("image_count", "gate", "spatial_delta", "correction", "agate_alpha")
        }

    def update_epoch(
        self,
        *,
        batch_size: int,
        subbag_aux_loss: float,
        eligible_multi_image_patients: int,
        subbag_gradient_final_ratio: float | None = None,
        spatial_gradient_final_ratio: float | None = None,
        agate_budget_loss_value: float | None = None,
        all_factor_spatial_gradients_finite_nonzero: bool | None = None,
    ) -> None:
        if batch_size < 1 or eligible_multi_image_patients < 0:
            raise ValueError("invalid epoch batch counts")
        values = {
            "subbag_aux_loss_batch_weighted": subbag_aux_loss,
            "subbag_gradient_final_ratio": subbag_gradient_final_ratio,
            "spatial_gradient_final_ratio": spatial_gradient_final_ratio,
            "agate_budget_loss": agate_budget_loss_value,
        }
        for name, value in values.items():
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise FloatingPointError(f"{name} is non-finite")
            self._epoch_sums[name] = self._epoch_sums.get(name, 0.0) + batch_size * number
            self._epoch_metric_weights[name] = (
                self._epoch_metric_weights.get(name, 0) + batch_size
            )
        self._epoch_batch_weight += int(batch_size)
        eligible = int(eligible_multi_image_patients)
        self._eligible_multi += eligible
        if eligible > 0:
            self._subbag_sum += eligible * float(subbag_aux_loss)
            self._subbag_weight += eligible
            self._active_subbag_batches += 1
        if all_factor_spatial_gradients_finite_nonzero is not None:
            self._gradient_observations += 1
            self._gradient_all_finite_nonzero &= bool(
                all_factor_spatial_gradients_finite_nonzero
            )

    def update_validation(
        self,
        *,
        image_count: torch.Tensor,
        spatial_gate: torch.Tensor,
        spatial_delta: torch.Tensor,
        agate_alpha: torch.Tensor | None = None,
    ) -> None:
        counts = torch.as_tensor(image_count).detach().cpu().reshape(-1)
        gate = _finite_float(spatial_gate, "spatial_gate").detach().cpu()
        delta = _finite_float(spatial_delta, "spatial_delta").detach().cpu()
        if gate.ndim != 2 or gate.shape != delta.shape or len(counts) != len(gate):
            raise ValueError("validation tensors must be [patients,factors]")
        if (counts < 1).any() or not torch.equal(counts, counts.round()):
            raise ValueError("validation image counts must be positive integers")
        self._validation_batch_count += 1
        self._validation["image_count"].append(counts.float())
        self._validation["gate"].append(gate)
        self._validation["spatial_delta"].append(delta)
        self._validation["correction"].append(gate * delta)
        if agate_alpha is not None:
            alpha = _finite_float(agate_alpha, "agate_alpha").detach().cpu().reshape(-1)
            if len(alpha) != len(counts):
                raise ValueError("agate_alpha must have one value per patient")
            self._validation["agate_alpha"].append(alpha)
            self._agate_alpha_batch_count += 1

    def finalize(self) -> dict[str, Any]:
        has_epoch = self._epoch_batch_weight > 0
        has_validation = bool(self._validation["image_count"])
        if not has_epoch and not has_validation:
            raise RuntimeError("no epoch or validation batches were accumulated")
        epoch: dict[str, Any] | None = None
        if has_epoch:
            epoch = {
                name: total / self._epoch_metric_weights[name]
                for name, total in self._epoch_sums.items()
            }
            epoch.update({
                "batch_patient_weight": self._epoch_batch_weight,
                "eligible_multi_image_patients_total": self._eligible_multi,
                "active_subbag_batch_count": self._active_subbag_batches,
                "subbag_aux_loss_eligible_patient_weighted": (
                    self._subbag_sum / self._subbag_weight if self._subbag_weight else None
                ),
                "spatial_gradient_all_batches_finite_nonzero": (
                    self._gradient_all_finite_nonzero
                    if self._gradient_observations else None
                ),
                "spatial_gradient_audited_batch_count": self._gradient_observations,
                "aggregation_scope": "complete_epoch_not_last_batch",
            })
        if not has_validation:
            return {"epoch": epoch, "full_validation": None}
        if self._agate_alpha_batch_count not in (0, self._validation_batch_count):
            raise RuntimeError(
                "agate_alpha must be supplied for every validation batch or none"
            )
        values = {
            key: torch.cat(parts) if parts else torch.empty(0)
            for key, parts in self._validation.items()
        }
        counts = values["image_count"]
        single, multi = counts == 1, counts > 1
        correction = values["correction"]

        def residual(mask: torch.Tensor) -> dict[str, float | int | None]:
            x = correction[mask]
            if not x.numel():
                return {"n": 0, "rms": None, "mean_abs": None, "max_abs": None}
            return {
                "n": int(mask.sum()),
                "rms": float(torch.sqrt(x.square().mean())),
                "mean_abs": float(x.abs().mean()),
                "max_abs": float(x.abs().max()),
            }

        full_validation = {
            "patient_count": int(len(counts)),
            "single_image_patient_count": int(single.sum()),
            "multi_image_patient_count": int(multi.sum()),
            "gate_all": _distribution(values["gate"]),
            "gate_single_image": _distribution(values["gate"][single]),
            "gate_multi_image": _distribution(values["gate"][multi]),
            "residual_all": residual(torch.ones_like(single)),
            "residual_single_image": residual(single),
            "residual_multi_image": residual(multi),
            "agate_alpha": (
                _distribution(values["agate_alpha"])
                if values["agate_alpha"].numel() else None
            ),
            "aggregation_scope": "complete_outer_validation_not_last_batch",
        }
        return {"epoch": epoch, "full_validation": full_validation}
