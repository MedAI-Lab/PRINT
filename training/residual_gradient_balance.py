"""Compute residual gradients with bounded factor loss balancing."""
import math
from collections.abc import Sequence
from typing import Any
import torch
from residual_gradient_objectives import _checked_gradient_lists, residual_gradient_scale, RESIDUAL_CAP_TO_BASE

def project_residual_gradients(
    uniform_gradients: Sequence[torch.Tensor],
    famo_gradients: Sequence[torch.Tensor],
    profile_gradients: Sequence[torch.Tensor],
    *,
    epoch: int,
    cap_to_base: float = RESIDUAL_CAP_TO_BASE,
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Add a projected/capped FAMO-minus-uniform residual to the base gradient.

    All routing arithmetic is FP32.  Returned tensors use the uniform-gradient
    dtype/device and no input tensor is mutated.
    """

    if cap_to_base not in (0.05, 0.15, 0.30) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("cap must be .05/.15/.30 and epsilon positive finite")
    schedule = residual_gradient_scale(epoch)
    uniform, famo, profile = _checked_gradient_lists(
        uniform_gradients, famo_gradients, profile_gradients
    )
    uniform32 = [value.detach().float() for value in uniform]
    famo32 = [value.detach().float() for value in famo]
    profile32 = [value.detach().float() for value in profile]
    base = [left + right for left, right in zip(uniform32, profile32)]
    residual = [right - left for left, right in zip(uniform32, famo32)]
    dot = sum((extra * primary).sum() for extra, primary in zip(residual, base))
    base_sq = sum(value.square().sum() for value in base)
    residual_sq = sum(value.square().sum() for value in residual)
    conflict = bool(float(dot.cpu()) < 0.0 and float(base_sq.cpu()) > 0.0)
    coefficient = dot / (base_sq + epsilon) if conflict else dot.new_zeros(())
    projected = [extra - coefficient * primary for extra, primary in zip(residual, base)]
    post_dot_before_cleanup = sum(
        (extra * primary).sum() for extra, primary in zip(projected, base)
    )
    projected_sq_before_cleanup = sum(value.square().sum() for value in projected)
    base_norm = torch.sqrt(base_sq)
    projected_norm_before_cleanup = torch.sqrt(projected_sq_before_cleanup)
    tolerance_before_cleanup = max(
        1e-7,
        1e-6 * float(projected_norm_before_cleanup.cpu()) * float(base_norm.cpu()),
    )
    cleanup_count = 0
    cleanup_coefficient = post_dot_before_cleanup.new_zeros(())
    if float(post_dot_before_cleanup.cpu()) < -tolerance_before_cleanup:
        cleanup_count = 1
        cleanup_coefficient = post_dot_before_cleanup / (base_sq + epsilon)
        projected = [
            extra - cleanup_coefficient * primary
            for extra, primary in zip(projected, base)
        ]
    post_dot = sum((extra * primary).sum() for extra, primary in zip(projected, base))
    projected_sq = sum(value.square().sum() for value in projected)
    projected_norm = torch.sqrt(projected_sq)
    tolerance = max(
        1e-7,
        1e-6 * float(projected_norm.cpu()) * float(base_norm.cpu()),
    )
    if float(post_dot.cpu()) < -tolerance:
        raise RuntimeError("Residual gradient one-sided projection failed after one numerical cleanup")
    cap_scale = torch.clamp(cap_to_base * base_norm / (projected_norm + epsilon), max=1.0)
    safe_residual = [value * cap_scale for value in projected]
    final32 = [primary + schedule * extra for primary, extra in zip(base, safe_residual)]
    if any(not torch.isfinite(value).all() for value in final32):
        raise FloatingPointError("Residual gradient routed gradient is non-finite")
    safe_norm = torch.sqrt(sum(value.square().sum() for value in safe_residual))
    ratio = float(safe_norm.cpu()) / max(float(base_norm.cpu()), epsilon)
    if ratio > cap_to_base + 1e-6:
        raise RuntimeError("Residual gradient residual norm cap audit failed")
    routed = [value.to(device=reference.device, dtype=reference.dtype)
              for value, reference in zip(final32, uniform)]
    return routed, {
        "route": "v27p_famo_minus_uniform_one_sided_projection_then_cap",
        "projection_compute_dtype": "torch.float32",
        "epoch": epoch,
        "residual_lambda": schedule,
        "conflict": conflict,
        "pre_dot_base": float(dot.cpu()),
        "post_projection_dot_before_cleanup": float(post_dot_before_cleanup.cpu()),
        "halfspace_tolerance_before_cleanup": tolerance_before_cleanup,
        "numerical_cleanup_count": cleanup_count,
        "numerical_cleanup_maximum": 1,
        "numerical_cleanup_coefficient": float(cleanup_coefficient.cpu()),
        "post_projection_dot_base": float(post_dot.cpu()),
        "projected_residual_norm_before_cap": float(projected_norm.cpu()),
        "halfspace_tolerance": tolerance,
        "post_projection_halfspace_pass": True,
        "projection_coefficient": float(coefficient.cpu()),
        "pre_residual_norm": float(torch.sqrt(residual_sq).cpu()),
        "base_norm": float(base_norm.cpu()),
        "cap_scale": float(cap_scale.cpu()),
        "safe_residual_to_base": ratio,
        "cap_to_base": cap_to_base,
        "inputs_mutated": False,
        "finite": True,
    }
