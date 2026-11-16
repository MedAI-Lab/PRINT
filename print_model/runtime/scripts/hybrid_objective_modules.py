#!/usr/bin/env python3
"""Hybrid objective modules for PRINT."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


L2SP_CAPTURE_SCOPE = (
    "after_strict_full_mae_load_before_any_warmup_or_optimizer_step"
)
HYBRID_SOURCES = ("AGATE", "SUBBAG", "spatial")


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if result < 0 or not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _materialize_gradients(
    primary_grads: Sequence[torch.Tensor],
    auxiliary_grads: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    primary = list(primary_grads)
    auxiliary = list(auxiliary_grads)
    if not primary or len(primary) != len(auxiliary):
        raise ValueError("primary and auxiliary gradients must have equal non-zero length")
    for index, (base, extra) in enumerate(zip(primary, auxiliary)):
        if not isinstance(base, torch.Tensor) or not isinstance(extra, torch.Tensor):
            raise TypeError(f"gradient pair {index} must contain tensors")
        if base.shape != extra.shape or base.device != extra.device:
            raise ValueError(f"gradient pair {index} has mismatched shape or device")
        if not torch.isfinite(base.detach()).all():
            raise FloatingPointError(f"primary gradient {index} contains NaN or Inf")
        if not torch.isfinite(extra.detach()).all():
            raise FloatingPointError(f"auxiliary gradient {index} contains NaN or Inf")
    return primary, auxiliary


def _sqnorm(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(gradient.float().square().sum() for gradient in gradients)


def _stable_direction_audit(
    raw: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Accumulate direction checks in FP64 while preserving the FP32 gradient route."""

    if not raw or len(raw) != len(expected) or len(raw) != len(actual):
        raise ValueError("direction audit gradients must have equal non-zero length")

    def stable_sum(terms: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack([term.detach().to(torch.float64) for term in terms]).sum()

    raw_sq = stable_sum([gradient.float().square().sum() for gradient in raw])
    actual_sq = stable_sum(
        [gradient.float().square().sum() for gradient in actual]
    )
    raw_actual_dot = stable_sum(
        [
            (first.float() * second.float()).sum()
            for first, second in zip(raw, actual)
        ]
    )
    denominator = torch.sqrt(raw_sq * actual_sq)
    stable_cosine = (
        0.0
        if float(denominator.detach().cpu()) == 0.0
        else float((raw_actual_dot / denominator).detach().cpu())
    )

    error_sq = stable_sum(
        [
            (observed.float() - requested.float()).square().sum()
            for observed, requested in zip(actual, expected)
        ]
    )
    expected_sq = stable_sum(
        [gradient.float().square().sum() for gradient in expected]
    )
    error_norm = torch.sqrt(error_sq)
    expected_norm = torch.sqrt(expected_sq)
    relative_error = (
        0.0
        if float(expected_norm.detach().cpu()) == 0.0
        else float((error_norm / expected_norm).detach().cpu())
    )
    formula_recomposed = relative_error <= 1e-7
    stable_collinear = stable_cosine >= 1.0 - 1e-10
    return {
        "stable_raw_final_cosine": stable_cosine,
        "formula_recomposition_relative_l2_error": relative_error,
        "formula_recomposition_verified": formula_recomposed,
        "stable_collinearity_verified": stable_collinear,
        "direction_audit_accumulator_dtype": "torch.float64",
        "direction_audit_changes_routed_gradient": False,
        "direction_unchanged": formula_recomposed,
    }


def norm_only_cap_against_primary(
    primary_grads: Sequence[torch.Tensor],
    auxiliary_grads: Sequence[torch.Tensor],
    *,
    cap: float,
    name: str,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Uniformly scale an auxiliary gradient without changing its direction."""

    if not name:
        raise ValueError("name must be non-empty")
    cap = _finite_nonnegative(cap, "cap")
    primary, auxiliary = _materialize_gradients(primary_grads, auxiliary_grads)
    primary_fp32 = [gradient.detach().float() for gradient in primary]
    raw_fp32 = [gradient.detach().float() for gradient in auxiliary]
    primary_norm = torch.sqrt(_sqnorm(primary_fp32))
    raw_norm = torch.sqrt(_sqnorm(raw_fp32))
    primary_nonzero = float(primary_norm.detach().cpu()) > 0.0
    raw_nonzero = float(raw_norm.detach().cpu()) > 0.0
    if not primary_nonzero or not raw_nonzero or cap == 0.0:
        scale = raw_norm.new_zeros(())
    else:
        scale = torch.clamp(cap * primary_norm / raw_norm, max=1.0)
    final_fp32 = [scale * gradient for gradient in raw_fp32]
    final = [
        gradient.to(device=reference.device, dtype=reference.dtype)
        for gradient, reference in zip(final_fp32, primary)
    ]
    final_fp32_actual = [gradient.float() for gradient in final]
    final_norm = torch.sqrt(_sqnorm(final_fp32_actual))
    direction_audit = _stable_direction_audit(
        raw_fp32, final_fp32, final_fp32_actual
    )
    raw_final_cosine = direction_audit["stable_raw_final_cosine"]
    raw_ratio = raw_norm / primary_norm if primary_nonzero else raw_norm.new_zeros(())
    final_ratio = (
        final_norm / primary_norm if primary_nonzero else final_norm.new_zeros(())
    )
    direction_unchanged = (
        not raw_nonzero
        or float(scale.detach().cpu()) == 0.0
        or direction_audit["direction_unchanged"]
    )
    if not direction_unchanged:
        raise RuntimeError("norm-only cap changed auxiliary direction")
    direction_audit["direction_unchanged"] = direction_unchanged
    return final, {
        "name": name,
        "cap": cap,
        "primary_unchanged": True,
        "direction_projection": False,
        "norm_only": True,
        "primary_norm": float(primary_norm.detach().cpu()),
        "raw_auxiliary_norm": float(raw_norm.detach().cpu()),
        "final_auxiliary_norm": float(final_norm.detach().cpu()),
        "raw_auxiliary_to_primary": float(raw_ratio.detach().cpu()),
        "final_auxiliary_to_primary": float(final_ratio.detach().cpu()),
        "cap_scale": float(scale.detach().cpu()),
        "raw_final_cosine": raw_final_cosine,
        "direction_unchanged": direction_unchanged,
        **direction_audit,
        "input_finite": True,
        "output_finite": all(torch.isfinite(value).all().item() for value in final),
        "compute_dtype": "torch.float32",
    }


def merge_hybrid_auxiliary_gradients(
    primary_grads: Sequence[torch.Tensor],
    auxiliary_grads: Mapping[str, Sequence[torch.Tensor]],
    *,
    cap: float = 0.25,
    required_sources: Sequence[str] = HYBRID_SOURCES,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Ordinarily sum three auxiliaries, then apply one norm-only total cap."""

    primary = list(primary_grads)
    if float(cap) != 0.25:
        raise ValueError("The total auxiliary gradient cap must equal 0.25.")
    required = tuple(required_sources)
    if len(set(required)) != len(required) or not required:
        raise ValueError("required_sources must be unique and non-empty")
    if set(auxiliary_grads) != set(required):
        raise ValueError("auxiliary_grads must exactly match required_sources")
    raw_sum = [torch.zeros_like(gradient) for gradient in primary]
    source_raw_norms: dict[str, float] = {}
    for source in required:
        _, gradients = _materialize_gradients(primary, auxiliary_grads[source])
        raw_sum = [
            current + gradient.to(device=current.device, dtype=current.dtype)
            for current, gradient in zip(raw_sum, gradients)
        ]
        source_raw_norms[source] = float(
            torch.sqrt(_sqnorm([gradient.float() for gradient in gradients]))
            .detach()
            .cpu()
        )
    final_auxiliary, cap_audit = norm_only_cap_against_primary(
        primary, raw_sum, cap=cap, name="AGATE+SUBBAG+spatial"
    )
    merged = [base + extra for base, extra in zip(primary, final_auxiliary)]
    return merged, {
        **cap_audit,
        "sources": list(required),
        "source_raw_norms": source_raw_norms,
        "ordinary_weighted_sum": True,
        "per_source_direction_projection": False,
        "single_total_norm_cap": True,
        "combined_raw_auxiliary_norm": cap_audit["raw_auxiliary_norm"],
        "combined_final_auxiliary_norm": cap_audit["final_auxiliary_norm"],
        "combined_raw_auxiliary_to_primary": cap_audit[
            "raw_auxiliary_to_primary"
        ],
        "combined_final_auxiliary_to_primary": cap_audit[
            "final_auxiliary_to_primary"
        ],
        "combined_raw_final_cosine": cap_audit["raw_final_cosine"],
        "combined_direction_unchanged": cap_audit["direction_unchanged"],
    }


def _materialize_named_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> list[tuple[str, nn.Parameter]]:
    entries = list(named_parameters)
    if not entries:
        raise ValueError("named encoder parameters must be non-empty")
    names: set[str] = set()
    identities: set[int] = set()
    for name, parameter in entries:
        if not name or name in names:
            raise ValueError("encoder parameter names must be non-empty and unique")
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("encoder snapshots require nn.Parameter values")
        if id(parameter) in identities:
            raise ValueError("one parameter cannot appear more than once")
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError(f"encoder parameter {name!r} is non-finite")
        names.add(name)
        identities.add(id(parameter))
    return entries


def _stable_encoder_layer(name: str) -> str:
    block = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
    if block:
        return f"blocks.{int(block.group(1))}"
    if re.search(r"(?:^|\.)patch_embed(?:\.|$)", name):
        return "patch_embed"
    if re.search(r"(?:^|\.)norm(?:\.|$)", name):
        return "norm"
    if name.endswith("cls_token") or ".cls_token" in name:
        return "cls_token"
    return "other"


class EncoderL2SPSnapshot(nn.Module):
    """Frozen full-MAE encoder anchor with uniform L2-SP."""

    def __init__(
        self,
        named_encoder_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        mode: str,
        strict_load_verified: bool,
        optimizer_steps_completed: int,
        capture_scope: str = L2SP_CAPTURE_SCOPE,
        uniform_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        if not strict_load_verified:
            raise ValueError("strict full-MAE load must be verified before L2-SP capture")
        if optimizer_steps_completed != 0:
            raise ValueError("L2-SP reference must be captured before any optimizer step")
        if capture_scope != L2SP_CAPTURE_SCOPE:
            raise ValueError("invalid L2-SP capture scope")
        if mode != "uniform":
            raise ValueError("Expected uniform L2-SP")
        uniform_lambda = _finite_nonnegative(uniform_lambda, "uniform_lambda")
        if uniform_lambda != 0.01:
            raise ValueError("The uniform L2-SP coefficient must equal 0.01.")
        entries = _materialize_named_parameters(named_encoder_parameters)
        self.mode = mode
        self.capture_scope = capture_scope
        self.uniform_lambda = uniform_lambda
        self._entries: list[tuple[str, str, tuple[int, ...], str]] = []
        for index, (name, parameter) in enumerate(entries):
            buffer_name = f"reference_{index:05d}"
            self.register_buffer(buffer_name, parameter.detach().float().clone())
            self._entries.append(
                (name, buffer_name, tuple(parameter.shape), _stable_encoder_layer(name))
            )

    @classmethod
    def capture_after_strict_full_mae_load(
        cls,
        named_encoder_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        mode: str,
        strict_load_verified: bool,
        optimizer_steps_completed: int,
        uniform_lambda: float = 0.01,
    ) -> "EncoderL2SPSnapshot":
        return cls(
            named_encoder_parameters,
            mode=mode,
            strict_load_verified=strict_load_verified,
            optimizer_steps_completed=optimizer_steps_completed,
            uniform_lambda=uniform_lambda,
        )

    def _lambda_for(self, parameter: nn.Parameter) -> float:
        return self.uniform_lambda if parameter.requires_grad else 0.0

    @property
    def snapshot_audit(self) -> dict[str, Any]:
        return {
            "capture_scope": self.capture_scope,
            "capture_time": "after_strict_full_mae_load_before_any_training",
            "strict_full_mae_load_verified": True,
            "optimizer_steps_completed_at_capture": 0,
            "encoder_only": True,
            "mode": self.mode,
            "uniform_lambda": self.uniform_lambda,
            "validation_or_test_data_read": False,
            "snapshot_parameter_count": len(self._entries),
            "snapshot_trainable_parameters": 0,
        }

    def forward(
        self, named_encoder_parameters: Iterable[tuple[str, nn.Parameter]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        live = dict(_materialize_named_parameters(named_encoder_parameters))
        if set(live) != {name for name, _, _, _ in self._entries}:
            raise ValueError("live encoder parameter names do not match the snapshot")
        weighted_terms: list[torch.Tensor] = []
        unweighted_half_terms: list[torch.Tensor] = []
        layer_state: dict[str, dict[str, Any]] = {}
        for name, buffer_name, shape, layer in self._entries:
            parameter = live[name]
            if tuple(parameter.shape) != shape:
                raise ValueError(f"shape changed for L2-SP parameter {name!r}")
            reference = getattr(self, buffer_name).to(device=parameter.device)
            delta_sq = (parameter.float() - reference).square().sum()
            reference_sq = reference.square().sum()
            coefficient = self._lambda_for(parameter)
            weighted_terms.append(0.5 * coefficient * delta_sq)
            unweighted_half_terms.append(0.5 * delta_sq)
            state = layer_state.setdefault(
                layer,
                {
                    "delta_sq": delta_sq.new_zeros(()),
                    "reference_sq": reference_sq.new_zeros(()),
                    "parameter_count": 0,
                    "numel": 0,
                    "effective_lambdas": set(),
                    "trainable_parameter_count": 0,
                },
            )
            state["delta_sq"] = state["delta_sq"] + delta_sq
            state["reference_sq"] = state["reference_sq"] + reference_sq
            state["parameter_count"] += 1
            state["numel"] += parameter.numel()
            state["effective_lambdas"].add(coefficient)
            state["trainable_parameter_count"] += int(parameter.requires_grad)
        loss = torch.stack(weighted_terms).sum()
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("L2-SP loss contains NaN or Inf")
        layerwise_drift: dict[str, Any] = {}
        for layer, state in layer_state.items():
            delta_l2 = torch.sqrt(state["delta_sq"])
            reference_l2 = torch.sqrt(state["reference_sq"])
            effective_lambdas = sorted(float(x) for x in state["effective_lambdas"])
            layerwise_drift[layer] = {
                "parameter_count": int(state["parameter_count"]),
                "trainable_parameter_count": int(state["trainable_parameter_count"]),
                "numel": int(state["numel"]),
                "effective_lambdas": effective_lambdas,
                "delta_l2": float(delta_l2.detach().cpu()),
                "snapshot_l2": float(reference_l2.detach().cpu()),
                "relative_drift": float(
                    (delta_l2 / reference_l2.clamp_min(1e-12)).detach().cpu()
                ),
            }
        audit = self.snapshot_audit
        audit.update(
            {
                "loss_name": f"V12_{self.mode}_encoder_L2_SP",
                "loss": float(loss.detach().cpu()),
                "unweighted_half_sum_squared_distance": float(
                    torch.stack(unweighted_half_terms).sum().detach().cpu()
                ),
                "layerwise_drift": layerwise_drift,
                "frozen_patch_embed_cls_blocks0_3_reported": all(
                    layer in layerwise_drift
                    for layer in (
                        "patch_embed",
                        "cls_token",
                        "blocks.0",
                        "blocks.1",
                        "blocks.2",
                        "blocks.3",
                    )
                ),
                "direction_projection": False,
                "gradient_cap_api": "l2sp_gradients_norm_only_cap",
                "finite": True,
                "compute_dtype": str(loss.dtype),
            }
        )
        return loss, audit


def _autograd_gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
    create_graph: bool,
) -> tuple[list[torch.Tensor], int]:
    if not parameters:
        raise ValueError("gradient target parameters must be non-empty")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("gradient target parameters contain duplicates")
    if not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("gradient targets must be currently trainable parameters")
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("loss contains NaN or Inf")
    if loss.requires_grad:
        raw = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
    else:
        raw = tuple(None for _ in parameters)
    unused = sum(gradient is None for gradient in raw)
    gradients = [
        torch.zeros_like(parameter)
        if gradient is None
        else gradient.to(device=parameter.device, dtype=parameter.dtype)
        for parameter, gradient in zip(parameters, raw)
    ]
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise FloatingPointError("autograd returned NaN or Inf")
    return gradients, unused


def l2sp_gradients(
    l2sp_loss: torch.Tensor,
    encoder_parameters: Iterable[nn.Parameter],
    primary_grads: Sequence[torch.Tensor],
    *,
    cap: float = 0.05,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Request encoder-only L2-SP gradients and apply no projection."""

    parameters = list(encoder_parameters)
    if float(cap) != 0.05:
        raise ValueError("The L2-SP gradient cap must equal 0.05.")
    raw, unused = _autograd_gradients(
        l2sp_loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    final, audit = norm_only_cap_against_primary(
        primary_grads, raw, cap=cap, name="L2-SP"
    )
    audit.update(
        {
            "gradient_route": "currently_trainable_encoder_only",
            "parameter_count": len(parameters),
            "unused_parameter_count": unused,
            "writes_parameter_grad_fields": False,
            "direction_projection": False,
        }
    )
    return final, audit
