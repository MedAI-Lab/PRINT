"""Spatial modules for PRINT."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from spatial_attention_controls import FactorSoftAttention

class LocalResidualAttention(FactorSoftAttention):
    """Keep scoring/pooling and the gate's local context exactly as inherited."""

    def forward(self, patch_tokens, *, patch_mask=None):
        result = super().forward(patch_tokens, patch_mask=patch_mask)
        _, attention, train_map, logits, context, audit = result
        with torch.autocast(device_type=patch_tokens.device.type, enabled=False):
            head_context = self.context_fusion(torch.cat((context, context), -1))
            hidden = F.gelu(self.context_projection(head_context))
            raw = torch.einsum("ifh,fh->if", hidden, self.residual_weight) + self.residual_bias
            output = self.maximum_residual_z * torch.tanh(raw)
        if not torch.isfinite(output).all():
            raise FloatingPointError("spatial output non-finite")
        audit = dict(audit, v31_arm=self.v31_arm, gate_context="original_local_context",
                     output_role="bounded_residual",
                     maximum_absolute_delta_z=float(output.detach().abs().max()))
        return output, attention, train_map, logits, context, audit


def install_spatial_arm(model, arm):
    """Install the local residual module while preserving the encoder state."""
    if arm != "LOCAL_CAPACITY_CONTROL":
        raise ValueError(f"unsupported spatial configuration: {arm}")
    spatial = model.factor_v12_spatial
    if type(spatial) is not FactorSoftAttention:
        raise TypeError("requires a compatible spatial attention module")
    gate = model.factor_v12_spatial_gate
    if not hasattr(gate, "enable_single_image_gate") or not hasattr(gate, "single_image_scale"):
        raise TypeError("requires the existing AllPatientFactorSpatialGate")
    if spatial.maximum_residual_z != .15:
        raise ValueError("The spatial residual cap must equal 0.15.")
    spatial.__class__ = LocalResidualAttention
    spatial.v31_arm = arm
    gate.enable_single_image_gate = True
    gate.single_image_scale = 1.0
    dim = spatial.token_dim
    with torch.random.fork_rng(devices=[], enabled=True):
        layer = nn.Linear(2 * dim, dim)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:, :dim].copy_(torch.eye(dim))
        layer.bias.zero_()
    spatial.context_fusion = layer.to(device=spatial.residual_weight.device,
                                      dtype=spatial.residual_weight.dtype)
    added = sum(p.numel() for p in layer.parameters())
    return dict(arm=arm, singleton_spatial=True, singleton_scale=1.,
                agate_unchanged=True, cap=spatial.maximum_residual_z,
                added_parameters=added, gate_context="original_local_context",
                separate_spatial_target=False)
