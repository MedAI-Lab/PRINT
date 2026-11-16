"""Spatial training schedules and gradient diagnostics."""
from __future__ import annotations

from typing import Any

SPATIAL_PROFILE_ARM = "G_SPATIAL_PROFILE"


_PROFILE_ROWS: dict[tuple[str, int], list[dict[str, Any]]] = {}


def record_spatial_gradient(
    arm: str, epoch: int, component: Any, gradients: list[Any], parameters: list[Any], model: Any,
) -> None:
    """Record the single combined spatial traversal by named consumer branch."""
    if arm != SPATIAL_PROFILE_ARM:
        return
    import torch
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: dict[str, list[Any]] = {
        "visual_backbone": [], "factor_decoder": [], "agate": [],
        "spatial_attention_and_head": [], "spatial_gate": [], "other": [],
    }
    for parameter, gradient in zip(parameters, gradients):
        name = names.get(id(parameter), "")
        if name.startswith("factor_v12_agate"):
            key = "agate"
        elif name.startswith("factor_v12_spatial_gate"):
            key = "spatial_gate"
        elif name.startswith("factor_v12_spatial"):
            key = "spatial_attention_and_head"
        elif name.startswith("factor_simple"):
            key = "factor_decoder"
        elif "blocks." in name or name.startswith(("patch_embed", "norm")):
            key = "visual_backbone"
        else:
            key = "other"
        groups[key].append(gradient)
    def norm(values: list[Any]) -> float:
        if not values:
            return 0.0
        return float(torch.sqrt(sum(value.detach().float().square().sum() for value in values)).cpu())
    row = {
        "profile_component": float(component.detach().cpu()),
        "combined_spatial_objective_single_gradient_traversal": True,
        "profile_component_separate_optimizer_gradient_added": False,
        "base_prediction_detached_by_construction": True,
        "branch_gradient_norms": {key: norm(value) for key, value in groups.items()},
        "all_combined_gradients_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
    }
    _PROFILE_ROWS.setdefault((arm, epoch), []).append(row)


