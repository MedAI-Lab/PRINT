#!/usr/bin/env python3
"""Factor modules for PRINT."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from factor_geometry_modules import (
    canonical_sha256,
    fit_outer_train_geometry,
    validate_geometry_artifact,
)


FORMAT_VERSION = "v15_outer_train_factor_artifact_v1"
FIT_SCOPE = "stagea_outer_train_only"
FACTOR_COUNT = 13


def _finite_matrix(value: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().double().numpy()
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != FACTOR_COUNT:
        raise ValueError(f"{name} must have shape [patients>=2, 13]")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _unique_ids(values: Sequence[Any]) -> list[str]:
    ids = [str(value) for value in values]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("patient_ids must be non-empty and unique")
    return ids


def _body(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "artifact_sha256"}


def fit_v15_factor_artifact(
    outer_train_targets: torch.Tensor | np.ndarray,
    patient_ids: Sequence[Any],
    *,
    outer_fold: int,
    factor_names: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the fold geometry and metadata used by the released manifest format."""

    targets = _finite_matrix(outer_train_targets, "outer_train_targets")
    ids = _unique_ids(patient_ids)
    if len(ids) != len(targets):
        raise ValueError("patient_ids and targets row counts differ")
    geometry = fit_outer_train_geometry(
        targets,
        ids,
        outer_fold=outer_fold,
        factor_names=factor_names,
        minimum_eigenvalue=1e-4,
    )
    a, b = np.triu_indices(len(targets), k=1)
    distances = np.linalg.norm(targets[a] - targets[b], axis=1) / math.sqrt(FACTOR_COUNT)
    if not np.isfinite(distances).all() or not len(distances):
        raise FloatingPointError("outer-train pair distances are invalid")
    tertiles = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    body: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "fit_scope": FIT_SCOPE,
        "outer_fold": int(outer_fold),
        "factor_count": FACTOR_COUNT,
        "patient_count": len(ids),
        "patient_order_sha256": canonical_sha256(ids),
        "outer_train_targets_fp64_sha256": canonical_sha256(targets.tolist()),
        "outer_validation_targets_read": False,
        "v14_geometry": geometry,
        "r2_partial_whitening_alpha": 0.5,
        "r3_distance_definition": f"euclidean_standardized_z_div_sqrt_{FACTOR_COUNT}",
        "r3_quantile_method": "numpy_linear",
        "r3_boundary_rule": "(-inf,q1],(q1,q2],(q2,inf)",
        "r3_pair_count": int(len(distances)),
        "r3_distance_tertiles": [float(value) for value in tertiles],
    }
    body["artifact_sha256"] = canonical_sha256(body)
    validate_v15_factor_artifact(
        body,
        expected_patient_ids=ids,
        expected_outer_train_targets=targets,
        expected_outer_fold=outer_fold,
    )
    return body


def validate_v15_factor_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_patient_ids: Sequence[Any] | None = None,
    expected_outer_train_targets: torch.Tensor | np.ndarray | None = None,
    expected_outer_fold: int | None = None,
) -> dict[str, Any]:
    """Fail closed on provenance, hashes, fixed constants, and tertile order."""

    if not isinstance(artifact, Mapping):
        raise TypeError("factor artifact must be a mapping")
    if artifact.get("format_version") != FORMAT_VERSION or artifact.get("fit_scope") != FIT_SCOPE:
        raise ValueError("Unsupported factor artifact format or fitting scope.")
    if artifact.get("artifact_sha256") != canonical_sha256(_body(artifact)):
        raise ValueError("factor artifact canonical hash mismatch")
    if artifact.get("outer_validation_targets_read") is not False:
        raise ValueError("artifact does not prove validation targets were unread")
    if artifact.get("factor_count") != FACTOR_COUNT:
        raise ValueError("factor count must be 13")
    if artifact.get("r2_partial_whitening_alpha") != 0.5:
        raise ValueError("Unexpected partial-whitening metadata in the fold manifest")
    if artifact.get("r3_quantile_method") != "numpy_linear" or artifact.get("r3_boundary_rule") != "(-inf,q1],(q1,q2],(q2,inf)":
        raise ValueError("Unexpected distance-tertile metadata in the fold manifest")
    bounds = np.asarray(artifact.get("r3_distance_tertiles"), dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("Invalid distance tertiles in the fold manifest")
    if expected_outer_fold is not None and artifact.get("outer_fold") != expected_outer_fold:
        raise ValueError("artifact outer_fold mismatch")
    ids = None if expected_patient_ids is None else _unique_ids(expected_patient_ids)
    if ids is not None:
        if artifact.get("patient_count") != len(ids) or artifact.get("patient_order_sha256") != canonical_sha256(ids):
            raise ValueError("artifact patient order/hash mismatch")
    targets = None
    if expected_outer_train_targets is not None:
        targets = _finite_matrix(expected_outer_train_targets, "expected_outer_train_targets")
        if artifact.get("patient_count") != len(targets):
            raise ValueError("artifact target row count mismatch")
        if artifact.get("outer_train_targets_fp64_sha256") != canonical_sha256(targets.tolist()):
            raise ValueError("artifact target hash mismatch")
    validate_geometry_artifact(
        artifact.get("v14_geometry"),
        expected_patient_ids=ids,
        expected_outer_train_targets=targets,
        expected_outer_fold=expected_outer_fold,
    )
    return {
        "artifact_sha256": artifact["artifact_sha256"],
        "v14_geometry_sha256": artifact["v14_geometry"]["artifact_sha256"],
        "patient_order_sha256": artifact["patient_order_sha256"],
        "r3_distance_tertiles": bounds.tolist(),
        "pass": True,
    }
