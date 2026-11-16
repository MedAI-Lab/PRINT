"""Multiview VPR reconstruction backbone."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from factor_guided_encoder import FactorGuidedURFM
from factor_geometry_modules import TargetWhitening
from factor_spatial_modules import SpatialParameterInitializer
from heterogeneity_modules import MultiviewResidualGate
from agate_topk_modules import AllPatientFactorSpatialGate


class ReconstructionBackbone(FactorGuidedURFM):
    def __init__(self, init_path: Path, locked: dict, geometry_artifact: dict[str, Any]):
        super().__init__(init_path)
        self.factor_v14_geometry = TargetWhitening(geometry_artifact.get("v14_geometry", geometry_artifact))
        agate = locked["architecture"]["agate"]
        spatial = locked["architecture"]["topk_spatial"]
        global_rng = torch.get_rng_state()
        self.factor_v12_agate = MultiviewResidualGate(
            heterogeneity_hidden_dim=agate["hidden_dim"],
            heterogeneity_gate_hidden_dim=agate["gate_hidden_dim"],
            feature_maximum_relative_deviation=agate["feature_maximum_relative_deviation"],
            agate_initial_alpha=agate["initial_alpha"],
            agate_maximum_alpha=agate["maximum_alpha"],
            agate_hidden_dim=agate["gate_hidden_dim"],
        )
        self.factor_v12_spatial = SpatialParameterInitializer(
            spatial["token_dim"],
            num_factors=spatial["factor_count"],
            score_hidden_dim=spatial["score_hidden_dim"],
            residual_hidden_dim=spatial["residual_hidden_dim"],
        )
        self.factor_v12_spatial_gate = AllPatientFactorSpatialGate(
            context_dim=spatial["spatial_gate_descriptor_dim"],
            num_factors=spatial["factor_count"],
            hidden_dim=spatial["spatial_gate_hidden_dim"],
            initial=spatial["spatial_gate_initial"],
            single_image_scale=0.0,
            enable_single_image_gate=False,
        )
        self.spatial_initial_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.factor_v12_spatial.named_parameters()
        }
        torch.set_rng_state(global_rng)

    def encode_bag(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views = images.shape[:2]
        tokens = self.factor_tokens(images.flatten(0, 1))
        patch = tokens[:, 1:].view(batch, views, tokens.shape[1] - 1, 768)
        return patch.mean(2), patch

    def decode_u(self, patient_features: torch.Tensor) -> torch.Tensor:
        """Decode the patient features in the standardized VPR coordinates."""
        return self.factor_simple(patient_features)

    def factor_forward_details(self, images: torch.Tensor, mask: torch.Tensor, *, include_spatial: bool = True):
        features, patch = self.encode_bag(images)
        _, data = self.predict_encoded(features, patch, mask, include_spatial=include_spatial, details=True)
        data.update({
            "coordinate_prediction": data["z_subbag"],
            "ordinal_logits": None,
            "patient_feature": None,
            "residual": data["spatial_correction"],
            "mil": data["heterogeneity"],
            "candidate_details": data,
        })
        return data
