"""Factor-specific soft attention for VPR reconstruction."""
from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from factor_modules import FACTOR_COUNT


def _module_seed(seed: int, fold: int, offset: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("fold must be a nonnegative integer")
    return int(seed) * 10000 + int(fold) * 100 + int(offset)


def _validate_anchor_decoder(anchor_decoder: nn.Sequential) -> None:
    """Validate the reconstruction head dimensions and layer order."""

    if not isinstance(anchor_decoder, nn.Sequential) or len(anchor_decoder) != 5:
        raise TypeError("anchor_decoder must be a five-layer reconstruction head")
    norm, projection, activation, dropout, output = list(anchor_decoder)
    valid = (
        isinstance(norm, nn.LayerNorm)
        and tuple(norm.normalized_shape) == (768,)
        and isinstance(projection, nn.Linear)
        and projection.in_features == 768
        and projection.out_features == 256
        and isinstance(activation, nn.GELU)
        and isinstance(dropout, nn.Dropout)
        and dropout.p == 0.1
        and isinstance(output, nn.Linear)
        and output.in_features == 256
        and output.out_features == FACTOR_COUNT
    )
    if not valid:
        raise ValueError("The anchor decoder layers do not match the expected architecture.")


def _validate_tokens(
    patch_tokens: torch.Tensor,
    patch_mask: torch.Tensor | None,
    *,
    token_dim: int,
) -> torch.Tensor:
    if not isinstance(patch_tokens, torch.Tensor) or not patch_tokens.is_floating_point():
        raise TypeError("patch_tokens must be a floating tensor")
    if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != token_dim:
        raise ValueError(f"patch_tokens must have shape [images,patches,{token_dim}]")
    if patch_tokens.shape[0] < 1 or patch_tokens.shape[1] < 1:
        raise ValueError("patch_tokens must contain at least one image and patch")
    if not torch.isfinite(patch_tokens.detach()).all():
        raise FloatingPointError("patch_tokens contain NaN or infinity")
    if patch_mask is None:
        return torch.ones(patch_tokens.shape[:2], dtype=torch.bool, device=patch_tokens.device)
    if not isinstance(patch_mask, torch.Tensor) or patch_mask.dtype != torch.bool:
        raise TypeError("patch_mask must be a bool tensor")
    if patch_mask.shape != patch_tokens.shape[:2]:
        raise ValueError("patch_mask must have shape [images,patches]")
    if not patch_mask.any(dim=1).all():
        raise ValueError("every image must contain at least one valid patch")
    return patch_mask


class FactorSoftAttention(nn.Module):
    """Pool image patches separately for each protein representation."""


    def __init__(
        self,
        token_dim: int = 768,
        *,
        num_factors: int = FACTOR_COUNT,
        score_hidden_dim: int = 128,
        residual_hidden_dim: int = 64,
        attention_temperature: float = 0.10,
        maximum_residual_z: float = 0.15,
        seed: int,
        fold: int,
    ) -> None:
        super().__init__()
        if min(token_dim, num_factors, score_hidden_dim, residual_hidden_dim) < 1:
            raise ValueError("All attention dimensions must be positive")
        if num_factors != FACTOR_COUNT:
            raise ValueError(f"Expected {FACTOR_COUNT} protein representations")
        if attention_temperature != 0.10 or maximum_residual_z != 0.15:
            raise ValueError("Expected attention_temperature=0.10 and maximum_residual_z=0.15")
        self.token_dim = int(token_dim)
        self.num_factors = int(num_factors)
        self.attention_temperature = float(attention_temperature)
        self.maximum_residual_z = float(maximum_residual_z)
        self.module_seed = _module_seed(seed, fold, 2002)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(self.module_seed)
            self.score_norm = nn.LayerNorm(token_dim)
            self.score_hidden = nn.Linear(token_dim, score_hidden_dim)
            self.score_output = nn.Linear(score_hidden_dim, num_factors)
            self.context_projection = nn.Linear(token_dim, residual_hidden_dim)
            self.residual_weight = nn.Parameter(torch.empty(num_factors, residual_hidden_dim))
            self.residual_bias = nn.Parameter(torch.zeros(num_factors))
            nn.init.trunc_normal_(self.score_output.weight, std=0.02)
            nn.init.normal_(self.residual_weight, std=1e-3)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        mask = _validate_tokens(patch_tokens, patch_mask, token_dim=self.token_dim)
        with torch.autocast(device_type=patch_tokens.device.type, enabled=False):
            tokens = patch_tokens.float()
            logits = self.score_output(
                F.gelu(self.score_hidden(self.score_norm(tokens)))
            ).transpose(1, 2)
            logits = logits / self.attention_temperature
            logits = logits.masked_fill(~mask[:, None, :], -torch.inf)
            attention = logits.softmax(dim=-1)
            context = torch.einsum("ifp,ipd->ifd", attention, tokens)
            hidden = F.gelu(self.context_projection(context))
            raw = torch.einsum("ifh,fh->if", hidden, self.residual_weight)
            raw = raw + self.residual_bias
            delta = self.maximum_residual_z * torch.tanh(raw)

        valid_logits = logits.masked_select(mask[:, None, :].expand_as(logits))
        entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
        valid_count = mask.sum(1).float().clamp_min(2)
        normalized_entropy = entropy / valid_count.log().view(-1, 1)
        finite = bool(
            torch.isfinite(delta.detach()).all()
            and torch.isfinite(attention.detach()).all()
            and torch.isfinite(context.detach()).all()
        )
        if not finite:
            raise FloatingPointError("Spatial soft-attention output is non-finite")
        audit = {
            "selector": "factor_soft_attention",
            "pooling": "ordinary_factor_specific_soft_attention",
            "factor_specific_score_rows": True,
            "attention_temperature": self.attention_temperature,
            "attention_sum_max_abs_error": float(
                (attention.sum(-1) - 1.0).detach().abs().max().cpu()
            ),
            "normalized_attention_entropy_mean": float(
                normalized_entropy.detach().mean().cpu()
            ),
            "score_std": float(valid_logits.detach().std(unbiased=False).cpu()),
            "score_span": float(
                (valid_logits.detach().max() - valid_logits.detach().min()).cpu()
            ),
            "maximum_absolute_delta_z": float(delta.detach().abs().max().cpu()),
            "padding_attention_max": float(
                attention.detach().masked_select(~mask[:, None, :]).max().cpu()
            ) if (~mask).any() else 0.0,
            "padding_selection_count": 0,
            "finite": True,
        }
        return delta, attention, attention, logits, context, audit
