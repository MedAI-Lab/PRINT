#!/usr/bin/env python3
"""Factor geometry modules for PRINT."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf


FORMAT_VERSION = "v14_outer_train_factor_geometry_v1"
FIT_SCOPE = "stagea_outer_train_only"


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload using the repository canonical encoding."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_unique_strings(values: Sequence[Any], name: str) -> list[str]:
    result = [str(value) for value in values]
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain non-empty identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers")
    return result


def _finite_2d_targets(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().double().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("outer_train_targets must have shape [patients>=2, factors>=1]")
    if not np.isfinite(array).all():
        raise ValueError("outer_train_targets contains NaN or infinity")
    return np.ascontiguousarray(array, dtype=np.float64)


def _matrix_to_list(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=np.float64).tolist()


def _artifact_body(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "artifact_sha256"}


def fit_outer_train_geometry(
    outer_train_targets: torch.Tensor | np.ndarray,
    patient_ids: Sequence[Any],
    *,
    outer_fold: int,
    factor_names: Sequence[Any] | None = None,
    minimum_eigenvalue: float = 1e-4,
) -> dict[str, Any]:
    """Estimate target geometry from the supplied training partition using Ledoit–Wolf shrinkage.

    Patient order and the exact FP64 target matrix are hashed independently.
    The returned ``artifact_sha256`` covers every other field, so edits to
    provenance or numerical values are detected before trainer consumption.
    """

    targets = _finite_2d_targets(outer_train_targets)
    ids = _ordered_unique_strings(patient_ids, "patient_ids")
    if len(ids) != targets.shape[0]:
        raise ValueError("patient_ids and outer_train_targets row counts differ")
    if not isinstance(outer_fold, int) or outer_fold < 0:
        raise ValueError("outer_fold must be a non-negative integer")
    if not np.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 0:
        raise ValueError("minimum_eigenvalue must be finite and positive")
    factors = (
        [f"factor_{index}" for index in range(targets.shape[1])]
        if factor_names is None
        else _ordered_unique_strings(factor_names, "factor_names")
    )
    if len(factors) != targets.shape[1]:
        raise ValueError("factor_names and outer_train_targets column counts differ")

    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(targets)
    mean = np.asarray(estimator.location_, dtype=np.float64)
    raw_covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    raw_covariance = 0.5 * (raw_covariance + raw_covariance.T)
    raw_eigenvalues, eigenvectors = np.linalg.eigh(raw_covariance)
    eigenvalues = np.maximum(raw_eigenvalues, float(minimum_eigenvalue))
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    target_sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(target_sd, target_sd)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    whitening = (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T
    inverse_whitening = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    whitening = 0.5 * (whitening + whitening.T)
    inverse_whitening = 0.5 * (inverse_whitening + inverse_whitening.T)

    target_rows = _matrix_to_list(targets)
    body: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "fit_scope": FIT_SCOPE,
        "outer_fold": outer_fold,
        "patient_count": len(ids),
        "factor_count": len(factors),
        "factor_names": factors,
        "patient_order_sha256": canonical_sha256(ids),
        "outer_train_targets_fp64_sha256": canonical_sha256(target_rows),
        "outer_validation_targets_read": False,
        "estimator": "sklearn.covariance.LedoitWolf",
        "assume_centered": False,
        "store_precision": False,
        "shrinkage": float(estimator.shrinkage_),
        "decomposition": "numpy.linalg.eigh_symmetric_float64",
        "minimum_eigenvalue": float(minimum_eigenvalue),
        "raw_minimum_eigenvalue": float(raw_eigenvalues.min()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "mean": _matrix_to_list(mean),
        "covariance": _matrix_to_list(covariance),
        "correlation": _matrix_to_list(correlation),
        "target_sd": _matrix_to_list(target_sd),
        "eigenvalues": _matrix_to_list(eigenvalues),
        "whitening_matrix": _matrix_to_list(whitening),
        "inverse_whitening_matrix": _matrix_to_list(inverse_whitening),
    }
    body["artifact_sha256"] = canonical_sha256(body)
    validate_geometry_artifact(body, expected_patient_ids=ids)
    return body


def validate_geometry_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_patient_ids: Sequence[Any] | None = None,
    expected_outer_train_targets: torch.Tensor | np.ndarray | None = None,
    expected_outer_fold: int | None = None,
    tolerance: float = 2e-10,
) -> dict[str, Any]:
    """Deeply validate provenance, hash, symmetry, and transform identities."""

    if not isinstance(artifact, Mapping):
        raise TypeError("geometry artifact must be a mapping")
    if artifact.get("format_version") != FORMAT_VERSION or artifact.get("fit_scope") != FIT_SCOPE:
        raise ValueError("unsupported geometry artifact format or fit scope")
    observed_hash = artifact.get("artifact_sha256")
    expected_hash = canonical_sha256(_artifact_body(artifact))
    if observed_hash != expected_hash:
        raise ValueError("geometry artifact canonical hash mismatch")
    if artifact.get("outer_validation_targets_read") is not False:
        raise ValueError("geometry artifact does not prove validation targets were unread")
    if expected_outer_fold is not None and artifact.get("outer_fold") != expected_outer_fold:
        raise ValueError("geometry artifact outer_fold mismatch")
    if expected_patient_ids is not None:
        ids = _ordered_unique_strings(expected_patient_ids, "expected_patient_ids")
        if artifact.get("patient_count") != len(ids):
            raise ValueError("geometry artifact patient_count mismatch")
        if artifact.get("patient_order_sha256") != canonical_sha256(ids):
            raise ValueError("geometry artifact patient order/hash mismatch")
    if expected_outer_train_targets is not None:
        expected_targets = _finite_2d_targets(expected_outer_train_targets)
        if artifact.get("patient_count") != expected_targets.shape[0]:
            raise ValueError("geometry artifact target row count mismatch")
        if artifact.get("factor_count") != expected_targets.shape[1]:
            raise ValueError("geometry artifact target factor count mismatch")
        if artifact.get("outer_train_targets_fp64_sha256") != canonical_sha256(
            _matrix_to_list(expected_targets)
        ):
            raise ValueError("geometry artifact outer-train target hash mismatch")

    count = int(artifact.get("factor_count", 0))
    if count < 1 or len(artifact.get("factor_names", [])) != count:
        raise ValueError("invalid factor metadata")
    vector_names = ("mean", "target_sd", "eigenvalues")
    matrix_names = ("covariance", "correlation", "whitening_matrix", "inverse_whitening_matrix")
    arrays: dict[str, np.ndarray] = {}
    for name in vector_names:
        value = np.asarray(artifact.get(name), dtype=np.float64)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f"artifact {name} is invalid")
        arrays[name] = value
    for name in matrix_names:
        value = np.asarray(artifact.get(name), dtype=np.float64)
        if value.shape != (count, count) or not np.isfinite(value).all():
            raise ValueError(f"artifact {name} is invalid")
        if not np.allclose(value, value.T, atol=tolerance, rtol=0.0):
            raise ValueError(f"artifact {name} is not symmetric")
        arrays[name] = value
    if np.any(arrays["target_sd"] <= 0) or np.any(arrays["eigenvalues"] <= 0):
        raise ValueError("artifact has non-positive scale/eigenvalue")
    if not np.allclose(np.sqrt(np.diag(arrays["covariance"])), arrays["target_sd"], atol=tolerance, rtol=0):
        raise ValueError("artifact target_sd is inconsistent with covariance")
    derived_correlation = arrays["covariance"] / np.outer(arrays["target_sd"], arrays["target_sd"])
    if not np.allclose(derived_correlation, arrays["correlation"], atol=tolerance, rtol=0):
        raise ValueError("artifact correlation is inconsistent with covariance")
    identity = arrays["whitening_matrix"] @ arrays["inverse_whitening_matrix"]
    identity_error = float(np.max(np.abs(identity - np.eye(count))))
    if identity_error > 1e-9:
        raise ValueError("artifact whitening matrices are not inverses")
    return {
        "artifact_sha256": observed_hash,
        "patient_order_sha256": artifact["patient_order_sha256"],
        "factor_count": count,
        "symmetric_float64": True,
        "transform_identity_max_abs": identity_error,
        "pass": True,
    }


class TargetWhitening(nn.Module):
    """Fixed symmetric outer-train whitening/inverse-whitening transform."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        super().__init__()
        validate_geometry_artifact(artifact)
        self.artifact_sha256 = str(artifact["artifact_sha256"])
        self.factor_count = int(artifact["factor_count"])
        self.register_buffer("mean", torch.as_tensor(artifact["mean"], dtype=torch.float64))
        self.register_buffer("covariance", torch.as_tensor(artifact["covariance"], dtype=torch.float64))
        self.register_buffer("correlation", torch.as_tensor(artifact["correlation"], dtype=torch.float64))
        self.register_buffer("target_sd", torch.as_tensor(artifact["target_sd"], dtype=torch.float64))
        self.register_buffer("eigenvalues", torch.as_tensor(artifact["eigenvalues"], dtype=torch.float64))
        self.register_buffer("whitening_matrix", torch.as_tensor(artifact["whitening_matrix"], dtype=torch.float64))
        self.register_buffer("inverse_whitening_matrix", torch.as_tensor(artifact["inverse_whitening_matrix"], dtype=torch.float64))

    def _check(self, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError("values must be a floating torch.Tensor")
        if values.ndim < 1 or values.shape[-1] != self.factor_count:
            raise ValueError(f"values must have final dimension {self.factor_count}")
        if not torch.isfinite(values.detach()).all():
            raise FloatingPointError("values contain NaN or infinity")

    def whiten(self, values: torch.Tensor) -> torch.Tensor:
        self._check(values)
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        matrix = self.whitening_matrix.to(device=values.device, dtype=values.dtype)
        return (values - mean) @ matrix.T

    def inverse_whiten(self, whitened: torch.Tensor) -> torch.Tensor:
        self._check(whitened)
        mean = self.mean.to(device=whitened.device, dtype=whitened.dtype)
        matrix = self.inverse_whitening_matrix.to(device=whitened.device, dtype=whitened.dtype)
        return whitened @ matrix.T + mean

    def forward(self, whitened: torch.Tensor) -> torch.Tensor:
        """Map whitened coordinates back to standardized VPR coordinates."""

        return self.inverse_whiten(whitened)
