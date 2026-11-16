#!/usr/bin/env python3
"""Spatial model components for PRINT."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from reconstruction_backbone import ReconstructionBackbone
from factor_spatial_modules import (
    aggregate_patient_factor_context,
    randomized_spatial_copy,
)
from patch_query_modules import aggregate_patient_patch_residual
from spatial_attention_controls import (
    FactorSoftAttention,
    _validate_anchor_decoder,
)


def _state_tensors_equal(
    module: nn.Module,
    reference: dict[str, torch.Tensor],
    names: tuple[str, ...] | None = None,
) -> bool:
    """Check whether the selected state tensors exactly match their reference values."""

    state = module.state_dict()
    selected = tuple(sorted(state)) if names is None else names
    if any(name not in state or name not in reference for name in selected):
        return False
    return all(
        torch.equal(state[name].detach().cpu(), reference[name].detach().cpu())
        for name in selected
    )


def _parameter_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the named parameter values to CPU for later comparison."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


class SpatialReconstructionModel(ReconstructionBackbone):
    """Reconstruct standardized VPRs from multiview features and local spatial evidence."""

    def __init__(
        self,
        init_path,
        candidate: str,
        locked: dict[str, Any],
        seed: int,
        geometry_artifact: dict[str, Any],
    ) -> None:
        config = locked["candidates"][candidate]
        if config["decoder"] != "v12_plain" or config["target_parameterization"] != "original":
            raise ValueError("Expected the plain reconstruction head and original target coordinates")
        super().__init__(init_path, locked, geometry_artifact)
        actual_seed, fold = seed // 100, seed % 100
        canonical_topk_state = {
            name: value.detach().cpu().clone()
            for name, value in self.factor_v12_spatial.state_dict().items()
        }
        _validate_anchor_decoder(self.factor_simple)
        self.factor_v12_spatial = FactorSoftAttention(seed=actual_seed, fold=fold)
        self.factor_v12_spatial.load_state_dict(canonical_topk_state, strict=True)
        self.spatial_initialization_audit = {
            "candidate": candidate,
            "canonical_full_spatial_state_exact": _state_tensors_equal(self.factor_v12_spatial, canonical_topk_state),
            "canonical_residual_state_exact": _state_tensors_equal(
                self.factor_v12_spatial, canonical_topk_state,
                ("context_projection.weight", "context_projection.bias", "residual_weight", "residual_bias"),
            ),
            "selector_after_initialization": "continuous_soft_attention",
        }
        if not self.spatial_initialization_audit.get("canonical_residual_state_exact", False):
            raise RuntimeError("Spatial residual parameters do not match the expected initial state.")
        self.spatial_initial_state = _parameter_snapshot(self.factor_v12_spatial)

    def _checkpoint_encoder_tail(self) -> bool:
        trainable_block_count = sum(
            any(parameter.requires_grad for parameter in block.parameters())
            for block in self.blocks
        )
        return (
            self.training and torch.is_grad_enabled() and trainable_block_count == 8
        )

    def _run_encoder_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder blocks, checkpointing trainable blocks when eight are trainable during gradient-enabled training."""
        checkpoint_depth8 = self._checkpoint_encoder_tail()
        for block in self.blocks:
            block_trainable = any(
                parameter.requires_grad for parameter in block.parameters()
            )
            if checkpoint_depth8 and block_trainable:
                x = torch_checkpoint(
                    block, x, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x = block(x)
        return x

    def factor_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Aggregate the normalized outputs of the final four encoder blocks using learned weights."""
        x = self.patch_embed(images)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(len(x), -1, -1)
        x = torch.cat((cls, x), dim=1)
        checkpoint_depth8 = self._checkpoint_encoder_tail()
        outputs = []
        for index, block in enumerate(self.blocks):
            block_trainable = any(
                parameter.requires_grad for parameter in block.parameters()
            )
            if checkpoint_depth8 and block_trainable:
                x = torch_checkpoint(
                    block, x, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x = block(x)
            if index >= 8:
                outputs.append(self.norm(x))
        weights = torch.softmax(self.factor_layer_logits, dim=0)
        return sum(weight * output for weight, output in zip(weights, outputs))

    def forward_encoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the visible patch tokens and class token for masked reconstruction."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        batch_size, _, embedding_dim = x.shape
        x = x[~mask].reshape(batch_size, -1, embedding_dim)

        if self.vis_mask_ratio > 0:
            vis_mask_token = self.vis_mask_token + self.pos_embed[:, 1:, :]
            vis_mask_token = vis_mask_token.expand(batch_size, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(
                batch_size, -1, embedding_dim
            )
            visible_length = x.size(1)
            noise = torch.rand(batch_size, visible_length, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(visible_length * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones(
                [batch_size, visible_length], device=x.device
            )
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(
                vis_mask, dim=1, index=ids_restore
            ).unsqueeze(-1)
            x = x * (1.0 - vis_mask) + vis_mask_token * vis_mask

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._run_encoder_blocks(x)
        return self.norm(x)

    def spatial_forward_selected(self, selected: torch.Tensor):
        return self.factor_v12_spatial(selected)

    def randomized_spatial_evaluation_adapter(self, *, seed: int):
        return randomized_spatial_copy(self.factor_v12_spatial, seed=int(seed))

    def predict_encoded(
        self,
        features: torch.Tensor,
        patch: torch.Tensor,
        mask: torch.Tensor,
        *,
        include_spatial: bool = True,
        details: bool = False,
    ):
        decoder_coordinate_calls: list[torch.Tensor] = []

        def tracked_decode_z(patient_features: torch.Tensor) -> torch.Tensor:
            coordinate = self.decode_u(patient_features)
            decoder_coordinate_calls.append(coordinate)
            return coordinate

        agate_prediction, heterogeneity = self.factor_v12_agate(features, mask, tracked_decode_z)
        if len(decoder_coordinate_calls) != 2:
            raise RuntimeError("Expected exactly two decoder calls during AGATE aggregation.")
        z_subbag = heterogeneity["z_subbag"].float()
        decoder_coordinate = decoder_coordinate_calls[0]
        spatial_delta = spatial_map = spatial_train_map = spatial_raw = None
        image_context = patient_context = counts = spatial_diag = None
        gate = gate_diag = None
        if include_spatial:
            valid = mask.bool()
            selected = patch[valid]
            patient_index = torch.arange(len(mask), device=mask.device)[:, None].expand_as(mask)[valid]
            (
                image_delta,
                spatial_map,
                spatial_train_map,
                spatial_raw,
                image_context,
                spatial_diag,
            ) = self.spatial_forward_selected(
                selected
            )
            spatial_delta = aggregate_patient_patch_residual(
                image_delta, patient_index, patient_count=len(mask)
            )
            patient_context, counts = aggregate_patient_factor_context(
                image_context, patient_index, patient_count=len(mask)
            )
            gate, gate_diag = self.factor_v12_spatial_gate(patient_context, counts, spatial_delta=spatial_delta)
            correction = gate * spatial_delta
            prediction = agate_prediction.float() + correction
        else:
            correction = torch.zeros_like(agate_prediction.float())
            prediction = agate_prediction.float()

        data = {
            "prediction": prediction,
            "z_subbag": z_subbag,
            "decoder_coordinate": decoder_coordinate,
            "agate_prediction": agate_prediction.float(),
            "spatial_correction": correction,
            "spatial_delta": spatial_delta,
            "spatial_map": spatial_map,
            "spatial_train_map": spatial_train_map,
            "spatial_raw": spatial_raw,
            "spatial_image_context": image_context,
            "spatial_patient_context": patient_context,
            "spatial_image_count": counts,
            "spatial": spatial_diag,
            "spatial_gate": gate,
            "spatial_gate_diag": gate_diag,
            "heterogeneity": heterogeneity,
            "features": features,
            "patch": patch,
            "spatial_initialization_audit": self.spatial_initialization_audit,
        }
        return (prediction, data) if details else prediction
