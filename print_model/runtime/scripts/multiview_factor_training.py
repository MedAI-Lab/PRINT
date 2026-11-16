"""Finite-gradient checks used during VPR training."""
import torch

import factor_guided_encoder as core


def finite_grads(grads):
    return all(bool(torch.isfinite(g).all().item()) for g in grads)


def gradients(loss, parameters, *, retain_graph=False, label="objective"):
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"{label} loss contains NaN or Inf")
    result = core.replace_none(torch.autograd.grad(loss, parameters, allow_unused=True,
                                                   retain_graph=retain_graph), parameters)
    result = [g.float() for g in result]
    if not finite_grads(result):
        raise FloatingPointError(f"{label} unscaled FP32 parameter gradient contains NaN or Inf")
    return result
