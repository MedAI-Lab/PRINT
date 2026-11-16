#!/usr/bin/env python3
"""Factor loss balancing for PRINT."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from factor_modules import FACTOR_COUNT


def recompute_famo_objective_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Independently recompute FAMO's scalar objective using Python scalars only."""

    losses = audit.get("factor_losses")
    weights = audit.get("bounded_task_weights")
    if (not isinstance(losses, list) or not isinstance(weights, list)
            or len(losses) != FACTOR_COUNT or len(weights) != FACTOR_COUNT):
        raise ValueError("FAMO objective audit needs 13 serialized losses and weights")
    losses_float = [float(value) for value in losses]
    weights_float = [float(value) for value in weights]
    if (not all(math.isfinite(value) and value >= 0 for value in losses_float)
            or not all(math.isfinite(value) and value >= 0 for value in weights_float)):
        raise ValueError("FAMO objective audit contains invalid losses/weights")
    d = [value + 1e-8 for value in losses_float]
    independent_c = sum(weight / denominator for weight, denominator in zip(weights_float, d))
    if not math.isfinite(independent_c) or independent_c <= 0:
        raise ValueError("FAMO independently recomputed normalizer is invalid")
    independent_objective = sum(
        math.log(denominator) * weight / independent_c
        for denominator, weight in zip(d, weights_float)
    )
    recorded_c = float(audit.get("normalizer_c", float("nan")))
    recorded_objective = float(audit.get("objective_value", float("nan")))
    return {
        "independent_recomputed_normalizer_c": independent_c,
        "independent_recomputed_objective_value": independent_objective,
        "normalizer_recomposition_max_abs": abs(recorded_c - independent_c),
        "exact_recomposition_max_abs": abs(recorded_objective - independent_objective),
        "independent_recomposition_backend": "python_math_no_numpy_no_torch",
    }


def validate_famo_objective_audit(
    audit: Mapping[str, Any], *, absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Reject a self-consistent-looking but independently irreproducible audit."""

    if (not isinstance(audit, Mapping) or absolute_tolerance <= 0
            or relative_tolerance <= 0):
        raise ValueError("FAMO objective audit/tolerance is invalid")
    recomputed = recompute_famo_objective_audit(audit)

    def scale_aware_tolerance(left: float, right: float) -> float:
        return max(
            absolute_tolerance,
            relative_tolerance * max(abs(left), abs(right)),
        )

    for key, value in recomputed.items():
        if key in audit:
            if isinstance(value, str):
                if audit[key] != value:
                    raise ValueError("The recorded FAMO backend differs from the independent recomputation.")
            else:
                recorded = float(audit[key])
                expected = float(value)
                if abs(recorded - expected) > scale_aware_tolerance(recorded, expected):
                    raise ValueError("The recorded FAMO value differs from the independent recomputation beyond tolerance.")
    recorded_c = float(audit.get("normalizer_c", float("nan")))
    recomputed_c = float(recomputed["independent_recomputed_normalizer_c"])
    recorded_objective = float(audit.get("objective_value", float("nan")))
    recomputed_objective = float(recomputed["independent_recomputed_objective_value"])
    normalizer_tolerance = scale_aware_tolerance(recorded_c, recomputed_c)
    objective_tolerance = scale_aware_tolerance(recorded_objective, recomputed_objective)
    if (recomputed["normalizer_recomposition_max_abs"] > normalizer_tolerance
            or recomputed["exact_recomposition_max_abs"] > objective_tolerance):
        raise ValueError("FAMO objective cannot be independently recomposed")
    return {
        **recomputed,
        "normalizer_scale_aware_tolerance": normalizer_tolerance,
        "objective_scale_aware_tolerance": objective_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "pass": True,
    }

class BoundedFAMOFactorBalance(nn.Module):
    """Bounded FAMO for the 13 reconstruction targets."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(FACTOR_COUNT, dtype=torch.float32))

    def _weight_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.softmax(self.logits.float(), dim=0)
        raw_multipliers = FACTOR_COUNT * raw
        raw_clipped = torch.clamp(raw_multipliers, 0.5, 2.0)
        final_multipliers = raw_clipped / raw_clipped.mean()
        return raw_multipliers, raw_clipped, final_multipliers, final_multipliers / FACTOR_COUNT


    @staticmethod
    def factor_losses(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.ndim != 2 or predictions.shape != targets.shape or predictions.shape[1] != FACTOR_COUNT:
            raise ValueError("FAMO inputs must share shape [patients,13]")
        if not torch.isfinite(predictions.detach()).all() or not torch.isfinite(targets.detach()).all():
            raise FloatingPointError("FAMO inputs are non-finite")
        return F.smooth_l1_loss(predictions, targets.to(predictions), beta=1.0, reduction="none").mean(0)

    def objective(self, factor_losses: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if factor_losses.shape != (FACTOR_COUNT,) or not torch.isfinite(factor_losses.detach()).all() or bool((factor_losses.detach() < 0).any()):
            raise ValueError("FAMO factor_losses must be finite non-negative [13]")
        d = factor_losses.double() + 1e-8
        raw_multipliers, raw_clipped, multipliers, bounded = self._weight_state()
        bounded = bounded.double()
        c = torch.sum(bounded / d).detach()
        objective = torch.sum(torch.log(d) * bounded / c)
        if not torch.isfinite(objective.detach()) or float(c.cpu()) <= 0:
            raise FloatingPointError("FAMO objective is non-finite")
        audit = {
            "factor_losses": factor_losses.detach().double().cpu().tolist(),
            "effective_multipliers": multipliers.detach().cpu().tolist(),
            "bounded_task_weights": bounded.detach().cpu().tolist(),
            "raw_softmax_multipliers_pre_clip": raw_multipliers.detach().cpu().tolist(),
            "raw_clipped_multipliers_pre_normalization": raw_clipped.detach().cpu().tolist(),
            "raw_lower_clip_event_count": int((raw_multipliers.detach() < 0.5).sum().cpu()),
            "raw_upper_clip_event_count": int((raw_multipliers.detach() > 2.0).sum().cpu()),
            "final_post_normalization_multiplier_min": float(multipliers.detach().min().cpu()),
            "final_post_normalization_multiplier_max": float(multipliers.detach().max().cpu()),
            "normalizer_c": float(c.cpu()),
            "objective_value": float(objective.detach().cpu()),
            "multiplier_min": float(multipliers.detach().min().cpu()),
            "multiplier_max": float(multipliers.detach().max().cpu()),
            "finite": True,
        }
        independent = recompute_famo_objective_audit(audit)
        audit.update(independent)
        validate_famo_objective_audit(audit)
        return objective, audit

    def make_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam([self.logits], lr=0.025, weight_decay=0.01)

    def update_logits(
        self,
        before_losses: torch.Tensor,
        after_losses: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        if before_losses.shape != (FACTOR_COUNT,) or after_losses.shape != (FACTOR_COUNT,):
            raise ValueError("FAMO before/after losses must have shape [13]")
        if not torch.isfinite(before_losses).all() or not torch.isfinite(after_losses).all():
            raise FloatingPointError("FAMO improvement losses are non-finite")
        optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1
                or len(optimizer_parameters) != 1 or optimizer_parameters[0] is not self.logits
                or optimizer.param_groups[0].get("lr") != 0.025
                or optimizer.param_groups[0].get("weight_decay") != 0.01):
            raise ValueError("Expected one Adam parameter group containing only the FAMO logits, with learning rate 0.025 and weight decay 0.01.")
        optimizer.zero_grad(set_to_none=True)
        delta = torch.log(before_losses.detach().float() + 1e-8) - torch.log(after_losses.detach().float() + 1e-8)
        raw_multipliers, raw_clipped, multipliers, bounded = self._weight_state()
        update_objective = torch.sum(bounded * delta)
        update_objective.backward()
        if self.logits.grad is None or not torch.isfinite(self.logits.grad).all():
            raise RuntimeError("FAMO logit Jacobian gradient is missing/non-finite")
        gradient_norm = float(self.logits.grad.detach().norm().cpu())
        optimizer.step()
        updated_raw, updated_clipped, updated, _ = self._weight_state()
        return {
            "loss_log_improvement": delta.cpu().tolist(),
            "pre_update_multipliers": multipliers.detach().cpu().tolist(),
            "post_update_multipliers": updated.detach().cpu().tolist(),
            "pre_update_raw_softmax_multipliers": raw_multipliers.detach().cpu().tolist(),
            "pre_update_raw_clipped_multipliers": raw_clipped.detach().cpu().tolist(),
            "post_update_raw_softmax_multipliers": updated_raw.detach().cpu().tolist(),
            "post_update_raw_clipped_multipliers": updated_clipped.detach().cpu().tolist(),
            "pre_update_raw_lower_clip_event_count": int((raw_multipliers.detach() < 0.5).sum().cpu()),
            "pre_update_raw_upper_clip_event_count": int((raw_multipliers.detach() > 2.0).sum().cpu()),
            "post_update_raw_lower_clip_event_count": int((updated_raw.detach() < 0.5).sum().cpu()),
            "post_update_raw_upper_clip_event_count": int((updated_raw.detach() > 2.0).sum().cpu()),
            "pre_update_at_lower_bound_count": int((multipliers.detach() <= 0.5 + 1e-7).sum().cpu()),
            "pre_update_at_upper_bound_count": int((multipliers.detach() >= 2.0 - 1e-7).sum().cpu()),
            "post_update_at_lower_bound_count": int((updated.detach() <= 0.5 + 1e-7).sum().cpu()),
            "post_update_at_upper_bound_count": int((updated.detach() >= 2.0 - 1e-7).sum().cpu()),
            "post_update_final_multiplier_min": float(updated.detach().min().cpu()),
            "post_update_final_multiplier_max": float(updated.detach().max().cpu()),
            "logit_gradient_norm": gradient_norm,
            "internal_gradient_clip": False,
            "finite": bool(torch.isfinite(self.logits.detach()).all()),
        }


def capture_model_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu_rng": torch.get_rng_state().clone(),
        "torch_cuda_rng": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def recompute_with_isolated_rng(
    callback: Any,
    captured_rng_state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Run same-batch recompute at captured RNG, restoring entry RNG exactly."""

    if not callable(callback) or "torch_cpu_rng" not in captured_rng_state:
        raise ValueError("FAMO recompute requires callback and captured RNG state")
    entry = capture_model_rng_state()
    if torch.cuda.is_available():
        cuda = captured_rng_state.get("torch_cuda_rng")
        if not isinstance(cuda, list) or len(cuda) != torch.cuda.device_count():
            raise ValueError("FAMO captured CUDA RNG state is invalid")
    torch.set_rng_state(captured_rng_state["torch_cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)
    try:
        result = callback()
    finally:
        torch.set_rng_state(entry["torch_cpu_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(entry["torch_cuda_rng"])
    after = capture_model_rng_state()
    cpu_equal = torch.equal(entry["torch_cpu_rng"], after["torch_cpu_rng"])
    cuda_equal = all(torch.equal(a, b) for a, b in zip(entry["torch_cuda_rng"], after["torch_cuda_rng"]))
    if not cpu_equal or not cuda_equal:
        raise RuntimeError("FAMO isolated recompute advanced post-step RNG")
    return result, {"cpu_rng_bitwise_restored": cpu_equal,
                    "cuda_rng_bitwise_restored": cuda_equal,
                    "data_rng_objects_touched": False, "pass": True}
