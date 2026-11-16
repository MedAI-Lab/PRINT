"""Factor model factory for PRINT."""

import torch
from spatial_attention_controls import FactorSoftAttention


def build_reconstruction_model(
    init_path,
    candidate,
    locked,
    seed,
    geometry_artifact,
):
    from spatial_model_components import SpatialReconstructionModel, _parameter_snapshot

    if candidate != "CP_F30":
        raise ValueError(f"Unsupported reconstruction configuration: {candidate}")
    model = SpatialReconstructionModel(
        init_path, candidate, locked, seed, geometry_artifact
    )
    cfg = locked["architecture"]["topk_spatial"]
    with torch.random.fork_rng(devices=[], enabled=True):
        common = dict(
            token_dim=cfg["token_dim"],
            num_factors=cfg["factor_count"],
            score_hidden_dim=cfg["score_hidden_dim"],
            residual_hidden_dim=cfg["residual_hidden_dim"],
        )
        replacement = FactorSoftAttention(
            **common, seed=seed // 100, fold=seed % 100, attention_temperature=0.1
        )
    replacement.load_state_dict(model.factor_v12_spatial.state_dict(), strict=True)
    model.factor_v12_spatial = replacement
    model.spatial_initial_state = _parameter_snapshot(replacement)
    return model
