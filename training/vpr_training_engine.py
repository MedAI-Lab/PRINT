"""Train and validate the Stage 2 VPR reconstruction model."""
from __future__ import annotations
import json
from pathlib import Path
from patient_bag_data import PatientBag, collate, metrics
from spatial_residual_training import post_model
from encoder_schedule import depth as encoder_depth, encoder_lr
from training_hooks import spatial_profile_audit, check_encoder_schedule, record_spatial_gradient, spatial_objective as compute_spatial_objective, map_correlation_or_none, singleton_pass, stage_technical_audit, technical_map_audit
import hashlib

import math

from typing import Any
import numpy as np
from print_model.data_paths import study_input
import pandas as pd
import torch
import torch.nn.functional as F


from torch.utils.data import DataLoader
import factor_guided_encoder as core
from multiview_factor_training import gradients
from subbag_modules import split_patient_subbags

from patch_query_modules import patch_residual_activation_audit
from agate_audit import AGATEPatientAuditAccumulator, count_counterfactual_audit
from factor_spatial_modules import spatial_parameter_displacement, spatial_randomization_audit, topk_flip_audit, attention_flip_consistency_loss
from agate_topk_modules import EpochFullValidationAuditAggregator
from hybrid_objective_modules import EncoderL2SPSnapshot, l2sp_gradients, merge_hybrid_auxiliary_gradients
from factor_modules import fit_v15_factor_artifact
ORDERED_FACTORS = tuple(json.loads((Path(__file__).resolve().parents[1] / "print_model/factor_order.json").read_text()))
from factor_model_factory import build_reconstruction_model
from residual_gradient_balance import project_residual_gradients



from factor_loss_balancing import BoundedFAMOFactorBalance, capture_model_rng_state, recompute_with_isolated_rng
REPORT_METRICS = tuple(core.REPORT_METRICS) + (
    () if "factor_pearson_mean" in core.REPORT_METRICS else ("factor_pearson_mean",)
)


def factor_pearson_macro(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, list[float]]:
    """Return the 13-factor macro Pearson; never replace invalid factors by zero."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 13:
        raise ValueError("Pearson inputs must be aligned N x 13 arrays")
    values: list[float] = []
    for index in range(13):
        left, right = prediction[:, index], truth[:, index]
        if (not np.isfinite(left).all() or not np.isfinite(right).all()
                or np.std(left) <= 0.0 or np.std(right) <= 0.0):
            raise ValueError(f"factor {index} has non-finite or zero-variance Pearson input")
        value = float(np.corrcoef(left, right)[0, 1])
        if not math.isfinite(value):
            raise ValueError(f"factor {index} Pearson is non-finite")
        values.append(value)
    return float(np.mean(values)), values




def n0_loss(pred: torch.Tensor, target: torch.Tensor, profile_weight: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target) + profile_weight * (
        1 - F.cosine_similarity(pred, target, dim=1)
    ).mean()


def _sum_grads(*groups: list[torch.Tensor]) -> list[torch.Tensor]:
    if not groups:
        raise ValueError("at least one gradient group required")
    return [sum(values) for values in zip(*groups)]


def _add_grads(first: list[torch.Tensor], second: list[torch.Tensor]) -> list[torch.Tensor]:
    if len(first) != len(second):
        raise ValueError("gradient list length mismatch")
    return [a + b for a, b in zip(first, second)]


def _zeros(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter) for parameter in parameters]


def _norm(grads: list[torch.Tensor]) -> float:
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(g.float().square().sum() for g in grads)).detach().cpu())




def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def factor_geometry_validation_audit(
    prediction: np.ndarray,
    truth: np.ndarray,
    factor_names: list[str],
    train_abs_tail_thresholds: np.ndarray,
) -> dict[str, Any]:
    """Complete-fold collapse audit in original standardized z coordinates."""
    from scipy.stats import rankdata

    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != len(factor_names):
        raise ValueError("factor geometry audit shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise FloatingPointError("factor geometry audit contains non-finite values")
    tail_thresholds = np.asarray(train_abs_tail_thresholds, dtype=np.float64)
    if tail_thresholds.shape != (len(factor_names),) or not np.isfinite(tail_thresholds).all():
        raise ValueError("outer-train tail thresholds must be finite [factor_count]")

    def correlation_matrix(values: np.ndarray, *, spearman: bool) -> np.ndarray:
        work = np.apply_along_axis(rankdata, 0, values) if spearman else values
        centered = work - work.mean(0, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=0))
        matrix = centered.T @ centered / np.maximum(np.outer(norm, norm), 1e-12)
        matrix = np.clip(0.5 * (matrix + matrix.T), -1.0, 1.0)
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def matrix_audit(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
        upper = np.triu_indices(predicted.shape[0], 1)
        absolute = np.abs(predicted[upper])
        difference = predicted - observed
        frobenius = float(np.linalg.norm(difference, ord="fro"))
        denominator = float(np.linalg.norm(observed, ord="fro"))
        excess = np.abs(predicted[upper]) - np.abs(observed[upper])
        order = np.argsort(excess)[::-1][:5]
        pairs = []
        for location in order:
            first, second = int(upper[0][location]), int(upper[1][location])
            pairs.append({
                "factor_a": factor_names[first], "factor_b": factor_names[second],
                "predicted": float(predicted[first, second]),
                "true": float(observed[first, second]),
                "absolute_excess": float(excess[location]),
            })
        return {
            "predicted_matrix": predicted.tolist(), "true_matrix": observed.tolist(),
            "off_diagonal_absolute_mean": float(np.mean(absolute)),
            "off_diagonal_absolute_median": float(np.median(absolute)),
            "off_diagonal_absolute_p90": float(np.quantile(absolute, 0.90)),
            "off_diagonal_absolute_maximum": float(np.max(absolute)),
            "off_diagonal_count_ge_0p8": int(np.sum(absolute >= 0.8)),
            "off_diagonal_count_ge_0p9": int(np.sum(absolute >= 0.9)),
            "frobenius_distance": frobenius,
            "normalized_frobenius_distance": frobenius / max(denominator, 1e-12),
            "strongest_five_excess_correlation_pairs": pairs,
        }

    pearson_pred = correlation_matrix(pred, spearman=False)
    pearson_true = correlation_matrix(true, spearman=False)
    spearman_pred = correlation_matrix(pred, spearman=True)
    spearman_true = correlation_matrix(true, spearman=True)
    pred_variance = np.var(pred, axis=0, ddof=0)
    true_variance = np.var(true, axis=0, ddof=0)
    per_factor = []
    for index, name in enumerate(factor_names):
        error = pred[:, index] - true[:, index]
        pred_column, true_column = pred[:, index], true[:, index]
        true_variance_scalar = float(np.var(true_column, ddof=0))
        if true_variance_scalar > 1e-12:
            calibration_slope = float(np.cov(true_column, pred_column, ddof=0)[0, 1] / true_variance_scalar)
            calibration_intercept = float(np.mean(pred_column) - calibration_slope * np.mean(true_column))
        else:
            calibration_slope, calibration_intercept = 0.0, float(np.mean(pred_column))
        pred_iqr = float(np.quantile(pred_column, 0.75) - np.quantile(pred_column, 0.25))
        true_iqr = float(np.quantile(true_column, 0.75) - np.quantile(true_column, 0.25))
        tail = np.abs(true_column) >= tail_thresholds[index]
        per_factor.append({
            "factor": name,
            "pearson": float(correlation_matrix(
                np.column_stack((pred[:, index], true[:, index])), spearman=False
            )[0, 1]),
            "spearman": float(correlation_matrix(
                np.column_stack((pred[:, index], true[:, index])), spearman=True
            )[0, 1]),
            "mse": float(np.mean(error * error)),
            "mae": float(np.mean(np.abs(error))),
            "predicted_variance": float(pred_variance[index]),
            "true_variance": float(true_variance[index]),
            "variance_ratio": float(pred_variance[index] / max(true_variance[index], 1e-12)),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "robust_iqr_ratio": pred_iqr / max(true_iqr, 1e-12),
            "central_mae": float(np.mean(np.abs(error[~tail]))) if bool((~tail).any()) else None,
            "tail_mae": float(np.mean(np.abs(error[tail]))) if bool(tail.any()) else None,
        })
    pred_profile_variance = np.var(pred, axis=1, ddof=0)
    true_profile_variance = np.var(true, axis=1, ddof=0)
    upper = np.triu_indices(len(pred), 1)
    pred_profile_distance = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=2)[upper]
    true_profile_distance = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=2)[upper]
    distance_preservation = float(np.corrcoef(rankdata(pred_profile_distance), rankdata(true_profile_distance))[0, 1]) if len(pred_profile_distance) > 1 else 0.0
    singular_values = np.linalg.svd(pred - pred.mean(0, keepdims=True), compute_uv=False)
    singular_probability = singular_values / max(float(singular_values.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(singular_probability * np.log(singular_probability + 1e-12))))
    positive_singular = singular_values[singular_values > 1e-12]
    return {
        "scope": "complete_evaluated_fold_fold_patient_weighted",
        "coordinate_system": "original_fold_standardized_z",
        "patient_count": int(len(pred)), "factor_count": int(pred.shape[1]),
        "factor_names": factor_names,
        "tail_threshold_source": "stage2_outer_train_standardized_targets_abs_q0p75_per_factor",
        "tail_thresholds": tail_thresholds.tolist(),
        "tail_thresholds_sha256": hashlib.sha256(tail_thresholds.astype(np.float64).tobytes()).hexdigest(),
        "pearson": matrix_audit(pearson_pred, pearson_true),
        "spearman": matrix_audit(spearman_pred, spearman_true),
        "per_factor": per_factor,
        "prediction_matrix_singular_values": singular_values.tolist(),
        "prediction_matrix_effective_rank_definition": "exp_entropy_of_normalized_singular_values",
        "prediction_matrix_effective_rank": effective_rank,
        "prediction_matrix_condition_number": float(positive_singular.max() / positive_singular.min()) if len(positive_singular) else None,
        "materially_subunit_calibration_slope_threshold": 0.8,
        "materially_subunit_calibration_slope_count": int(sum(row["calibration_slope"] < 0.8 for row in per_factor)),
        "factor_variance_ratio_mean": float(np.mean(pred_variance / np.maximum(true_variance, 1e-12))),
        "patient_profile_variance_predicted_mean": float(np.mean(pred_profile_variance)),
        "patient_profile_variance_true_mean": float(np.mean(true_profile_variance)),
        "patient_profile_variance_ratio": float(
            np.mean(pred_profile_variance) / max(float(np.mean(true_profile_variance)), 1e-12)
        ),
        "profile_log_variance_distance_mean": float(np.mean(np.abs(
            np.log(pred_profile_variance + 1e-8) - np.log(true_profile_variance + 1e-8)
        ))),
        "within_profile_contrast_ratio": float(np.mean(np.std(pred, axis=1)) / max(float(np.mean(np.std(true, axis=1))), 1e-12)),
        "profile_distance_spearman_preservation": 0.0 if not np.isfinite(distance_preservation) else distance_preservation,
        "factor_mse_gt_1_count": int(sum(row["mse"] > 1.0 for row in per_factor)),
        "finite": True,
    }


def _safe_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
    label: str,
) -> list[torch.Tensor]:
    return gradients(loss, parameters, retain_graph=retain_graph, label=label) if parameters else []




def reconstruction_loss(details, target_z, profile_weight, factor_loss_balancer):
    factor_losses = factor_loss_balancer.factor_losses(details["z_subbag"].float(), target_z.float())
    pointwise, _ = factor_loss_balancer.objective(factor_losses)
    loss = pointwise + profile_weight * (
        1 - F.cosine_similarity(details["z_subbag"].float(), target_z.float(), dim=1)
    ).mean()
    return loss, {"factor_losses_before": factor_losses.detach().cpu().tolist()}


def subbag_objective(
    model: torch.nn.Module,
    features: torch.Tensor,
    patch: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    profile_weight: float,
    consistency_weight: float,
    include_spatial: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    first, second, eligible = split_patient_subbags(mask, generator=generator)
    if not eligible.any():
        return features.sum() * 0, {"eligible_multi_image_patients": 0, "weighted_total_loss": 0.0}
    prediction_first = model.predict_encoded(features[eligible], patch[eligible], first[eligible], include_spatial=include_spatial)
    prediction_second = model.predict_encoded(features[eligible], patch[eligible], second[eligible], include_spatial=include_spatial)
    truth = target[eligible]
    loss = (
        n0_loss(prediction_first, truth, profile_weight)
        + n0_loss(prediction_second, truth, profile_weight)
        + consistency_weight * F.smooth_l1_loss(prediction_first, prediction_second)
    )
    return loss, {
        "eligible_multi_image_patients": int(eligible.sum()),
        "weighted_total_loss": float(loss.detach()),
    }


def singleton_gate_contract_audit(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype
) -> dict[str, Any]:
    """Check gate values and per-factor gradients for single-view nodules."""
    model.eval()
    for images, mask, target, *_ in loader:
        single = mask.sum(1) == 1
        if not single.any():
            continue
        images, mask, target = images[single].to(device), mask[single].to(device), target[single].to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            details = model.factor_forward_details(images, mask)
        gate = details["spatial_gate"]
        context = details["spatial_patient_context"]
        with torch.autocast("cuda", enabled=False):
            raw_gate = model.factor_v12_spatial_gate.net(context.float()).squeeze(-1).sigmoid()
            expected = raw_gate
            loss = F.smooth_l1_loss(
                details["agate_prediction"].detach().float() + gate.float() * details["spatial_delta"].float(),
                target.float(),
            )
        gate_parameters = list(model.factor_v12_spatial_gate.parameters())
        gate_parameter_gradients = torch.autograd.grad(
            loss, gate_parameters, allow_unused=True, retain_graph=True
        )
        gate_parameter_gradients = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(gate_parameters, gate_parameter_gradients)
        ]
        factor_losses = []
        factor_parameter_gradient_norms = []
        for factor in range(13):
            factor_loss = F.smooth_l1_loss(
                details["agate_prediction"][:, factor].detach().float()
                + gate[:, factor].float() * details["spatial_delta"][:, factor].float(),
                target[:, factor].float(),
            )
            factor_losses.append(factor_loss)
            gradients_per_factor = torch.autograd.grad(
                factor_loss, gate_parameters, retain_graph=factor < target.shape[1] - 1, allow_unused=True
            )
            norm_sq = sum(
                (torch.zeros_like(parameter) if gradient is None else gradient).float().square().sum()
                for parameter, gradient in zip(gate_parameters, gradients_per_factor)
            )
            factor_parameter_gradient_norms.append(float(torch.sqrt(norm_sq).detach().cpu()))
        correction = gate.detach().float() * details["spatial_delta"].detach().float()
        exact = torch.equal(gate.detach().float(), expected.detach().float())
        return {
            "patient_count": int(len(gate)),
            "audit_scope": "outer_train_singleton_only",
            "mode": "V31_single_image_spatial_learned_gate_enabled",
            "effective_equals_locked_expression_bitwise": exact,
            "effective_equals_locked_expression_allclose": bool(torch.allclose(gate.detach().float(), expected.detach().float(), atol=0, rtol=1e-6)),
            "effective_gate_mean": float(gate.detach().mean().cpu()),
            "effective_gate_max": float(gate.detach().max().cpu()),
            "correction_rms": float(torch.sqrt(correction.square().mean()).cpu()),
            "gate_parameter_gradient_norm": _norm(gate_parameter_gradients),
            "gate_parameter_gradients_finite": all(bool(torch.isfinite(gradient).all()) for gradient in gate_parameter_gradients),
            "gate_parameter_gradients_nonzero": _norm(gate_parameter_gradients) > 0,
            "factor_gate_parameter_gradient_norms": factor_parameter_gradient_norms,
            "all_factor_gate_parameter_gradients_finite_nonzero": all(math.isfinite(value) and value > 0 for value in factor_parameter_gradient_norms),
        }
    raise RuntimeError("The loader contains no single-view nodules for the gate check.")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    random_seed: int,
):
    model.eval()
    predictions: list[np.ndarray] = []
    plain_predictions: list[np.ndarray] = []
    spatial_off_predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    patient_ids: list[str] = []
    maps: list[torch.Tensor] = []
    flip_maps: list[torch.Tensor] = []
    randomized_maps: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    gate_values: list[torch.Tensor] = []
    image_counts: list[torch.Tensor] = []
    spatial_rows: list[dict] = []
    topk_rows: list[dict] = []
    full = EpochFullValidationAuditAggregator()
    agate_audit = AGATEPatientAuditAccumulator(maximum_alpha=0.15)
    randomized = model.randomized_spatial_evaluation_adapter(
        seed=random_seed
    ).to(device).eval()
    cf_descriptors: list[torch.Tensor] = []
    cf_counts: list[torch.Tensor] = []
    edge_selected_counts = torch.zeros(13, dtype=torch.int64)
    edge_valid_counts = torch.zeros(13, dtype=torch.int64)
    total_selected_counts = torch.zeros(13, dtype=torch.int64)
    total_valid_counts = torch.zeros(13, dtype=torch.int64)
    padding_selection_count = 0
    all_spatial_outputs_finite = True
    patient_aggregation_exact = True
    for images, mask, target, _, pids in loader:
        images, mask = images.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            identity = model.factor_forward_details(images, mask)
            flipped = model.factor_forward_details(torch.flip(images, dims=[4]), mask)
            prediction = (identity["prediction"] + flipped["prediction"]) / 2
            plain_prediction = (identity["z_subbag"] + flipped["z_subbag"]) / 2
            spatial_off_prediction = (identity["agate_prediction"] + flipped["agate_prediction"]) / 2
        predictions.append(prediction.float().cpu().numpy())
        plain_predictions.append(plain_prediction.float().cpu().numpy())
        spatial_off_predictions.append(spatial_off_prediction.float().cpu().numpy())
        truths.append(target.numpy())
        patient_ids.extend(pids)
        valid = mask.bool()
        tokens = identity["patch"][valid]
        _, random_map, _, _, _, _ = randomized(tokens)
        maps.append(identity["spatial_map"].cpu())
        flip_maps.append(flipped["spatial_map"].cpu())
        randomized_maps.append(random_map.cpu())
        residuals.append(identity["spatial_correction"].cpu())
        gate_values.append(identity["spatial_gate"].cpu())
        image_counts.append(identity["heterogeneity"]["image_count"].cpu())
        technical_map_audit(identity)
        technical_map_audit(flipped)
        spatial_row = {**identity["spatial"], "image_count": int(valid.sum())}
        weight = identity["spatial_map"].detach().float()
        entropy = -(weight.clamp_min(1e-12) * weight.clamp_min(1e-12).log()).sum(-1)
        spatial_row["attention_entropy_normalized_mean"] = float(
            (entropy / math.log(weight.shape[-1])).mean().cpu()
        )
        spatial_row["effective_support_fraction_mean"] = float(
            (entropy.exp() / weight.shape[-1]).mean().cpu()
        )
        spatial_rows.append(spatial_row)
        padding_selection_count += int(identity["spatial"]["padding_selection_count"])
        all_spatial_outputs_finite &= bool(identity["spatial"]["finite"] and flipped["spatial"]["finite"])
        patient_aggregation_exact &= (
            int(identity["spatial_image_count"].sum().detach().cpu()) == int(valid.sum())
            and torch.equal(identity["spatial_image_count"].detach(), mask.sum(1).detach())
        )
        selection_count = max(1, int(round(0.20 * identity["spatial_map"].shape[-1])))
        hard = torch.zeros_like(identity["spatial_map"], dtype=torch.bool)
        hard.scatter_(-1, identity["spatial_map"].topk(selection_count, dim=-1).indices, True)
        flipped_hard = torch.zeros_like(flipped["spatial_map"], dtype=torch.bool)
        flipped_hard.scatter_(
            -1, flipped["spatial_map"].topk(selection_count, dim=-1).indices, True
        )
        hard = hard.cpu()
        edge = torch.zeros((14, 14), dtype=torch.bool)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        edge = edge.flatten().view(1, 1, 196)
        edge_selected_counts += (hard & edge).sum((0, 2))
        total_selected_counts += hard.sum((0, 2))
        edge_valid_counts += int(valid.sum()) * edge.sum() * torch.ones(13, dtype=torch.int64)
        total_valid_counts += int(valid.sum()) * 196 * torch.ones(13, dtype=torch.int64)
        topk_rows.append({**topk_flip_audit(
            hard.to(device), flipped_hard,
            identity["spatial_raw"], flipped["spatial_raw"], patch_grid=(14, 14)
        ), "image_count": int(valid.sum())})
        h = identity["heterogeneity"]
        full.update_validation(
            image_count=h["image_count"],
            spatial_gate=identity["spatial_gate"],
            spatial_delta=identity["spatial_delta"],
            agate_alpha=h["alpha"].squeeze(-1),
        )
        pool = h["pool"]
        agate_audit.update(
            alpha=h["alpha"].squeeze(-1), image_count=h["image_count"],
            population_sd=pool["standard_deviation"], max_minus_mean=pool["maximum_minus_mean"],
            raw_residual=h["raw_prediction_delta"], correction=h["correction"],
            prediction=h["z_subbag"] + h["correction"], base_prediction=h["z_subbag"],
            orthogonality_error=None,
        )
        multi = h["image_count"] > 1
        if multi.any():
            cf_descriptors.append(pool["heterogeneity"][multi].detach())
            cf_counts.append(h["image_count"][multi].detach())
    spatial_map = torch.cat(maps)
    spatial_flip = torch.cat(flip_maps)
    mapped_flip = torch.flip(spatial_flip.reshape(*spatial_flip.shape[:2], 14, 14), (-1,)).flatten(2)
    randomization = spatial_randomization_audit(spatial_map, torch.cat(randomized_maps), reference_similarity=1.0)
    rms = patch_residual_activation_audit(torch.cat(residuals), torch.ones(13))
    full_validation = full.finalize()["full_validation"]
    concatenated_gates = torch.cat(gate_values)
    multi_gates = concatenated_gates[torch.cat(image_counts) > 1]
    gate_saturation = float(((multi_gates < 0.01) | (multi_gates > 0.99)).float().mean()) if multi_gates.numel() else 0.0
    edge_per_factor = []
    for factor in range(13):
        selected_rate = float(edge_selected_counts[factor] / total_selected_counts[factor].clamp_min(1))
        valid_rate = float(edge_valid_counts[factor] / total_valid_counts[factor].clamp_min(1))
        edge_per_factor.append({
            "factor_index": factor,
            "factor_name": ORDERED_FACTORS[factor],
            "edge_selected_count": int(edge_selected_counts[factor]),
            "total_selected_count": int(total_selected_counts[factor]),
            "edge_valid_count": int(edge_valid_counts[factor]),
            "total_valid_count": int(total_valid_counts[factor]),
            "edge_selected_rate": selected_rate,
            "edge_valid_rate": valid_rate,
            "edge_enrichment": selected_rate / max(valid_rate, 1e-12),
        })
    agate_result = agate_audit.finalize()
    activation = {
        "attention_entropy_normalized_mean": (
            float(np.average(
                [row["attention_entropy_normalized_mean"] for row in spatial_rows],
                weights=[row["image_count"] for row in spatial_rows],
            ))
        ),
        "effective_support_fraction_mean": (
            float(np.average(
                [row["effective_support_fraction_mean"] for row in spatial_rows],
                weights=[row["image_count"] for row in spatial_rows],
            ))
        ),
        "score_span": float(np.max([row["score_span"] for row in spatial_rows])),
        "flip_spatial_cosine": float(F.cosine_similarity(spatial_map, mapped_flip, dim=-1).mean()),
        "flip_topk_jaccard": float(np.average([row["flip_topk_jaccard"] for row in topk_rows], weights=[row["image_count"] for row in topk_rows])),
        "flip_score_cosine": float(np.average([row["flip_score_cosine"] for row in topk_rows], weights=[row["image_count"] for row in topk_rows])),
        "factor_map_absolute_correlation_mean": map_correlation_or_none(spatial_map),
        "randomized_similarity_drop": randomization["randomized_similarity_drop"],
        "spatial_residual_rms_fraction": rms["residual_rms_fraction_mean"],
        "spatial_gate_mean": full_validation["gate_multi_image"]["mean"],
        "spatial_gate_saturation_fraction": gate_saturation,
        "full_validation_gate_residual_audit": full_validation,
        "agate_alpha_multi_distribution": agate_result["alpha_distribution"],
        "agate_alpha_multi_fraction_ge_0p145": agate_result["alpha_fraction_ge_0p145"],
        "padding_mask_audit": {
            "padding_selection_count": padding_selection_count,
            "pass": padding_selection_count == 0,
        },
        "patient_aggregation_audit": {
            "mask_counts_equal_spatial_counts_all_batches": patient_aggregation_exact,
            "patient_count": len(patient_ids),
            "pass": patient_aggregation_exact and len(patient_ids) == sum(len(row) for row in predictions),
        },
        "tta_audit": {
            "views": ["identity", "horizontal_flip"],
            "vertical_flip_used": False,
            "prediction_is_arithmetic_mean": True,
            "pass": True,
        },
        "finite_audit": {
            "spatial_outputs_finite": all_spatial_outputs_finite,
            "predictions_finite": bool(np.isfinite(np.concatenate(predictions)).all()),
            "plain_predictions_finite": bool(np.isfinite(np.concatenate(plain_predictions)).all()),
            "spatial_off_predictions_finite": bool(np.isfinite(np.concatenate(spatial_off_predictions)).all()),
            "truth_finite": bool(np.isfinite(np.concatenate(truths)).all()),
        },
        "edge_ring_per_factor_audit_only": edge_per_factor,
        "spatial_gate_context_dim": model.factor_v12_spatial_gate.context_dim,
        "spatial_gate_context_dim_matches_topk": model.factor_v12_spatial_gate.context_dim == 768,
    }
    activation["finite_audit"]["pass"] = all(activation["finite_audit"].values())
    activation["common_protocol_audits_pass"] = all(
        activation[name]["pass"]
        for name in ("padding_mask_audit", "patient_aggregation_audit", "tta_audit", "finite_audit")
    )
    if cf_descriptors:
        count_counterfactual_audit(
            patient_gate=model.factor_v12_agate.patient_gate,
            heterogeneity_descriptor=torch.cat(cf_descriptors),
            observed_image_count=torch.cat(cf_counts),
            maximum_alpha=model.factor_v12_agate.agate_maximum_alpha,
        )
    return (
        np.concatenate(predictions), np.concatenate(plain_predictions), np.concatenate(spatial_off_predictions), np.concatenate(truths),
        patient_ids, activation, agate_result,
    )


def train(cli, fold, *, inner=False):
    from types import SimpleNamespace
    settings = json.loads((Path(__file__).parent / "vpr_config.json").read_text())
    args = SimpleNamespace(**settings["parameters"])
    args.fold = cli.fold
    config = settings["architecture"]
    candidate_config = config["candidates"][args.reconstruction_model]
    famo_cap = candidate_config["famo_cap"]
    amp_dtype = torch.bfloat16
    device = torch.device("cuda")
    core.seed_all(args.seed)
    factors = json.loads((Path(__file__).resolve().parents[1] / "print_model/factor_order.json").read_text())
    if hasattr(cli, "paired"):
        from stage_tables import paired_targets
        clinical = cli.paired
        raw = paired_targets(clinical, factors, cli.targets)
    else:
        raw = pd.read_csv(cli.targets).set_index("ID", verify_integrity=True)[factors]
        clinical = pd.read_excel(study_input(cli.data, "stage2_clinical.xlsx")).set_index("ID", drop=False, verify_integrity=True)
    train_ids, val_ids = fold["train_patient_ids"], fold["validation_patient_ids"]
    if set(train_ids) & set(val_ids) or len(set(train_ids + val_ids)) != len(train_ids + val_ids):
        raise ValueError("Duplicate or overlapping split IDs")
    train_frame, val_frame = clinical.loc[train_ids], clinical.loc[val_ids]
    out = cli.out
    out.mkdir(parents=True, exist_ok=False)
    factor_mean = raw.loc[train_frame.ID].mean().to_numpy(np.float32)
    factor_std = raw.loc[train_frame.ID].std(ddof=0).to_numpy(np.float32)
    factor_std[factor_std < 1e-6] = 1
    targets = {
        patient_id: torch.tensor((raw.loc[patient_id].to_numpy(np.float32) - factor_mean) / factor_std)
        for patient_id in train_frame.ID
    }
    train_z = np.stack([targets[patient_id].numpy() for patient_id in train_frame.ID])
    train_abs_tail_thresholds = np.quantile(np.abs(train_z.astype(np.float64)), .75, axis=0)
    geometry_artifact = fit_v15_factor_artifact(train_z, train_frame.ID.tolist(), outer_fold=cli.fold, factor_names=factors)
    evaluation_frame = val_frame if inner else train_frame
    evaluation_targets = {
        pid: torch.tensor((raw.loc[pid].to_numpy(np.float32) - factor_mean) / factor_std)
        for pid in evaluation_frame.ID
    }
    train_dataset = PatientBag(train_frame, targets, True)
    train_audit_dataset = PatientBag(train_frame, targets, False)
    train_generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(train_dataset, batch_size=args.factor_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate, generator=train_generator)
    train_audit_loader = DataLoader(
        train_audit_dataset, batch_size=args.factor_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate,
    )
    evaluation_loader = DataLoader(
        PatientBag(evaluation_frame, evaluation_targets, False), batch_size=args.factor_batch_size,
        shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate,
    ) if inner else train_audit_loader
    model = build_reconstruction_model(cli.mae, args.reconstruction_model, config,
                      args.seed * 100 + args.fold, geometry_artifact).to(device)
    post_model(args.spatial_mode, model)
    factor_loss_balancer = BoundedFAMOFactorBalance().to(device)
    factor_weight_optimizer = factor_loss_balancer.make_optimizer()
    core.set_encoder_depth(model, 0)
    named_encoder_all = [(name, parameter) for name, parameter in model.named_parameters() if core.is_encoder_name(name) and not name.startswith("pos_embed")]
    encoder_all = [parameter for _, parameter in named_encoder_all]
    l2sp_config = config["losses"]["l2sp"]
    l2sp_snapshot = EncoderL2SPSnapshot.capture_after_strict_full_mae_load(
        named_encoder_all,
        mode=candidate_config["l2sp_mode"],
        strict_load_verified=model.mae_init_status.startswith("strict:"),
        optimizer_steps_completed=0,
        uniform_lambda=l2sp_config["uniform_lambda"],
    ).to(device)
    l2sp_snapshot(named_encoder_all)
    spatial_modules = [model.factor_v12_spatial, model.factor_v12_spatial_gate]
    spatial_params = []
    spatial_seen: set[int] = set()
    for module in spatial_modules:
        for parameter in module.parameters():
            if id(parameter) not in spatial_seen:
                spatial_params.append(parameter)
                spatial_seen.add(id(parameter))
    spatial_ids = {id(parameter) for parameter in spatial_params}
    agate_params = list(model.factor_v12_agate.parameters())
    agate_ids = {id(parameter) for parameter in agate_params}
    plain_params = [parameter for name, parameter in model.named_parameters() if name.startswith("factor_") and id(parameter) not in spatial_ids | agate_ids]
    visual_aggregation_named = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name == "factor_layer_logits"
    ]
    if len(visual_aggregation_named) != 1 or tuple(visual_aggregation_named[0][1].shape) != (4,):
        raise RuntimeError("Expected one factor_layer_logits parameter with shape (4,).")
    visual_aggregation_params = [visual_aggregation_named[0][1]]
    visual_aggregation_ids = {id(parameter) for parameter in visual_aggregation_params}
    if not visual_aggregation_ids <= {id(parameter) for parameter in plain_params}:
        raise RuntimeError("visual layer-aggregation parameter is not in the plain optimizer group")
    decoder_params = [parameter for name, parameter in model.named_parameters() if core.is_decoder_name(name)]
    groups_for_audit = {
        "encoder": encoder_all, "plain": plain_params, "agate": agate_params,
        "spatial": spatial_params, "decoder": decoder_params,
    }
    group_id_sets = {name: {id(parameter) for parameter in values} for name, values in groups_for_audit.items()}
    overlaps = {
        f"{first}__{second}": len(group_id_sets[first] & group_id_sets[second])
        for index, first in enumerate(group_id_sets)
        for second in list(group_id_sets)[index + 1:]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"optimizer parameter groups overlap: {overlaps}")
    covered = set().union(*group_id_sets.values())
    intended = {id(parameter) for name, parameter in model.named_parameters() if not name.startswith("pos_embed")}
    if covered != intended:
        raise RuntimeError(f"optimizer parameter coverage mismatch: missing={len(intended-covered)} extra={len(covered-intended)}")
    optimizer_groups = [
        {"params": encoder_all, "lr": args.backbone_lr, "name": "encoder"},
        {"params": plain_params, "lr": args.plain_head_lr, "name": "plain"},
        {"params": agate_params, "lr": args.agate_lr, "name": "agate"},
        {"params": spatial_params, "lr": args.spatial_head_lr, "name": "spatial"},
        {"params": decoder_params, "lr": args.decoder_lr, "name": "decoder"},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=config["common"]["weight_decay"],
        betas=tuple(config["common"]["optimizer_betas"]),
    )

    subbag_generator = torch.Generator(device=device).manual_seed(args.seed + 4000 + args.fold)
    common = config["common"]
    hybrid_cap = config["losses"]["hybrid_auxiliary_route"]["norm_only_total_cap_to_primary"]
    mechanism_ever = {
        "hybrid_route_all_steps_pass": True,
        "hybrid_route_observed_steps": 0,
        "hybrid_source_finite_nonzero_ever": {name: False for name in ("AGATE", "SUBBAG", "spatial")},
        "mae_route_all_steps_pass": True,
        "mae_route_observed_steps": 0,
        "l2sp_positive": False,
        "l2sp_direction_projection_absent": True,
        "l2sp_route_all_steps_pass": True,
        "l2sp_route_observed_steps": 0,
        "agate_own_gradient_all_steps_finite_nonzero": True,
        "agate_own_gradient_observed_steps": 0,
        "agate_own_gradient_skipped_singleton_only_steps": 0,
        "plain_vs_agate_nonalias": False,
        "plain_vs_agate_value_difference": False,
    }
    (out / "geometry.json").write_text(json.dumps(geometry_artifact))
    (out / "agate.json").write_text(json.dumps("no-beta"))
    for epoch in range(1, cli.epochs + 1):
        phase = "plain_n0_warmup" if epoch <= args.plain_base_warmup_epochs else "new_branch_warmup" if epoch < args.joint_start_epoch else "joint"
        depth = encoder_depth(epoch)
        core.set_encoder_depth(model, depth)
        shared = [parameter for parameter in encoder_all if parameter.requires_grad]
        shared_plain = shared + plain_params
        lr_scale = (
            (0.0 if phase != 'joint' else min(1.0, (epoch - args.joint_start_epoch + 1) / args.joint_lr_warmup_epochs))
        )
        optimizer.param_groups[0]["lr"] = encoder_lr(epoch)
        optimizer.param_groups[1]["lr"] = args.plain_head_lr * (
            min(1.0, epoch / args.plain_base_warmup_epochs)
            if phase == "plain_n0_warmup" else (0.0 if phase == "new_branch_warmup" else 1.0)
        )
        optimizer.param_groups[2]["lr"] = args.agate_lr
        optimizer.param_groups[3]["lr"] = 0.0 if phase == "plain_n0_warmup" else args.spatial_head_lr
        optimizer.param_groups[4]["lr"] = args.decoder_lr * lr_scale
        model.train()
        epoch_audit = EpochFullValidationAuditAggregator()
        primary_sum = 0.0
        patient_total = steps = 0
        protection_rows: list[dict] = []
        subbag_route_rows: list[dict] = []
        all_spatial_grad = True
        spatial_gradient_observed_steps = 0
        spatial_gradient_skipped_singleton_only_steps = 0
        primary_bitwise_unchanged_all_steps = True
        for images, image_mask, target, _, _batch_ids in train_loader:
            images, image_mask, target = images.to(device), image_mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            include_spatial = phase != "plain_n0_warmup"
            forward_rng = capture_model_rng_state() if phase == "joint" else None
            with torch.autocast("cuda", dtype=amp_dtype):
                details = model.factor_forward_details(images, image_mask, include_spatial=include_spatial)
                primary, primary_coordinate_audit = reconstruction_loss(details, target, 0.10, factor_loss_balancer)
                agate_aux = n0_loss(details["agate_prediction"], target, 0.10)
                subbag, subbag_diag = subbag_objective(
                    model, details["features"], details["patch"], image_mask, target,
                    subbag_generator, 0.10, 0.10, include_spatial,
                )
            spatial_objective = details["z_subbag"].sum() * 0.0
            spatial_profile_component = details["z_subbag"].sum() * 0.0
            if include_spatial:
                valid = image_mask.bool()
                with torch.autocast("cuda", dtype=amp_dtype):
                    flipped_patch = model.encode_bag(torch.flip(images, dims=[4]))[1][valid]
                    flipped_spatial = model.spatial_forward_selected(flipped_patch)
                with torch.autocast("cuda", enabled=False):
                    fidelity = F.smooth_l1_loss(
                        (details['agate_prediction'].detach().float() + details['spatial_correction'].float()),
                        target.float(),
                    )
                    flip_loss = attention_flip_consistency_loss(
                        details["spatial_train_map"], flipped_spatial[2], patch_grid=(14, 14)
                    )
                    spatial_objective, spatial_profile_component = compute_spatial_objective(
                        fidelity, flip_loss, details, target
                    )
            else:
                fidelity = flip_loss = primary * 0.0
            targets_all = shared_plain + agate_params + spatial_params
            uniform_pointwise = F.smooth_l1_loss(details["z_subbag"].float(), target.float())
            profile_component = 0.10 * (
                1 - F.cosine_similarity(details["z_subbag"].float(), target.float(), dim=1)
            ).mean()
            current_factor_losses = factor_loss_balancer.factor_losses(
                details["z_subbag"].float(), target.float()
            )
            famo_pointwise, _ = factor_loss_balancer.objective(current_factor_losses)
            uniform_grads = _safe_gradients(
                uniform_pointwise, targets_all, retain_graph=True, label="uniform_pointwise"
            )
            famo_grads = _safe_gradients(
                famo_pointwise, targets_all, retain_graph=True, label="famo_pointwise"
            )
            profile_grads = _safe_gradients(
                profile_component, targets_all, retain_graph=True, label="profile_only"
            )
            primary_grads, _ = project_residual_gradients(
                uniform_grads, famo_grads, profile_grads, epoch=epoch, cap_to_base=famo_cap
            )
            split_shared_plain = len(shared_plain)
            primary_shared_plain = primary_grads[:split_shared_plain]
            primary_reference_clones = [gradient.detach().clone() for gradient in primary_shared_plain]
            primary_agate = primary_grads[split_shared_plain:split_shared_plain + len(agate_params)]
            primary_spatial = primary_grads[split_shared_plain + len(agate_params):]
            final_shared_plain = list(primary_shared_plain)
            direct_agate = _zeros(agate_params)
            direct_spatial = _zeros(spatial_params)

            agate_grads = _safe_gradients(agate_aux, targets_all, retain_graph=True, label="AGATE_aux")
            subbag_grads = _safe_gradients(
                subbag,
                targets_all,
                retain_graph=include_spatial,
                label="SUBBAG_aux",
            )
            diagnostic_gradient = _zeros(shared_plain)
            geometry_gradient = _zeros(shared_plain)

            spatial_grads = (
                _safe_gradients(
                    spatial_objective,
                    targets_all,
                    retain_graph=False,
                    label="spatial_aux",
                )
                if include_spatial
                else _zeros(targets_all)
            )
            record_spatial_gradient(
                args.spatial_mode, epoch, spatial_profile_component, spatial_grads,
                targets_all, model,
            )
            if phase == "joint":
                final_shared_plain, protection_diag = merge_hybrid_auxiliary_gradients(
                    primary_shared_plain,
                    {
                        "AGATE": agate_grads[:split_shared_plain],
                        "SUBBAG": subbag_grads[:split_shared_plain],
                        "spatial": spatial_grads[:split_shared_plain],
                    },
                    cap=hybrid_cap,
                )
                if any(torch.count_nonzero(g).item() for g in diagnostic_gradient):
                    raise RuntimeError("Expected zero diagnostic gradients during VPR training.")
                final_shared_plain = _add_grads(final_shared_plain, diagnostic_gradient)
                final_shared_plain = _add_grads(final_shared_plain, geometry_gradient)
                protection_diag["primary_bitwise_unchanged"] = all(
                    torch.equal(before, after)
                    for before, after in zip(primary_reference_clones, primary_shared_plain)
                )
                protection_rows.append(protection_diag)
                route_pass = (
                    protection_diag["primary_unchanged"] is True
                    and protection_diag["output_finite"] is True
                    and protection_diag["ordinary_weighted_sum"] is True
                    and protection_diag["per_source_direction_projection"] is False
                    and protection_diag["combined_final_auxiliary_to_primary"] <= hybrid_cap + 1e-6
                    and protection_diag["direction_unchanged"] is True
                )
                mechanism_ever["hybrid_route_observed_steps"] += 1
                mechanism_ever["hybrid_route_all_steps_pass"] &= route_pass
                for source, norm in protection_diag["source_raw_norms"].items():
                    mechanism_ever["hybrid_source_finite_nonzero_ever"][source] |= math.isfinite(norm) and norm > 0
                raw_subbag_ratio = _norm(subbag_grads[:split_shared_plain]) / max(_norm(primary_shared_plain), 1e-12)
                combined_scale = float(protection_diag["cap_scale"])
                subbag_route_rows.append({
                    "mode": "ordinary_sum_then_shared_norm_cap_0p25",
                    "raw_ratio": raw_subbag_ratio,
                    "final_ratio": raw_subbag_ratio * combined_scale,
                })
            primary_bitwise_unchanged = all(
                torch.equal(before, after)
                for before, after in zip(primary_reference_clones, primary_shared_plain)
            )
            primary_bitwise_unchanged_all_steps &= primary_bitwise_unchanged
            if not primary_bitwise_unchanged:
                raise RuntimeError("immutable primary gradient changed during routing")
            direct_agate = _sum_grads(
                primary_agate,
                agate_grads[split_shared_plain:split_shared_plain + len(agate_params)],
                (
                    _zeros(agate_params)
                    if phase != "joint"
                    else subbag_grads[split_shared_plain:split_shared_plain + len(agate_params)]
                ),
                (
                    _zeros(agate_params)
                    if phase != "joint"
                    else spatial_grads[split_shared_plain:split_shared_plain + len(agate_params)]
                ),
            )
            direct_spatial = _sum_grads(
                primary_spatial,
                agate_grads[-len(spatial_params):],
                _zeros(spatial_params) if phase != "joint" else subbag_grads[-len(spatial_params):],
                spatial_grads[-len(spatial_params):],
            )

            l2sp_loss = primary * 0.0
            if phase == "joint" and shared:
                l2sp_loss, l2sp_diag = l2sp_snapshot(named_encoder_all)
                protected_l2sp, l2sp_protection = l2sp_gradients(
                    l2sp_loss,
                    shared,
                    primary_shared_plain[:len(shared)],
                    cap=l2sp_config["gradient_norm_cap_to_primary"],
                    retain_graph=False,
                )
                final_shared_plain[:len(shared)] = _add_grads(final_shared_plain[:len(shared)], protected_l2sp)
                mechanism_ever["l2sp_positive"] |= float(l2sp_loss.detach()) > 0
                mechanism_ever["l2sp_direction_projection_absent"] &= l2sp_protection["direction_projection"] is False
                l2sp_route_pass = (
                    l2sp_diag["finite"] is True
                    and l2sp_protection["output_finite"] is True
                    and l2sp_protection["direction_projection"] is False
                    and l2sp_protection["final_auxiliary_to_primary"] <= 0.050001
                )
                mechanism_ever["l2sp_route_observed_steps"] += 1
                mechanism_ever["l2sp_route_all_steps_pass"] &= l2sp_route_pass
            elif phase == "joint":
                l2sp_loss, l2sp_diag = l2sp_snapshot(named_encoder_all)
                if float(l2sp_loss.detach()) != 0.0:
                    raise RuntimeError("frozen encoder produced non-zero L2-SP loss")


            if phase == "joint":
                mechanism_ever["plain_vs_agate_nonalias"] |= details["z_subbag"].data_ptr() != details["agate_prediction"].data_ptr()
                mechanism_ever["plain_vs_agate_value_difference"] |= float((details["z_subbag"] - details["agate_prediction"]).detach().abs().max()) > 0
            if phase == "joint":
                if bool((image_mask.sum(1) > 1).any()):
                    agate_own_norm = _norm(direct_agate)
                    mechanism_ever["agate_own_gradient_observed_steps"] += 1
                    mechanism_ever["agate_own_gradient_all_steps_finite_nonzero"] &= (
                        agate_own_norm > 0
                        and all(bool(torch.isfinite(gradient).all()) for gradient in direct_agate)
                    )
                else:
                    mechanism_ever["agate_own_gradient_skipped_singleton_only_steps"] += 1

            for parameter, gradient in zip(shared_plain, final_shared_plain):
                parameter.grad = gradient
            for parameter, gradient in zip(agate_params, direct_agate):
                parameter.grad = gradient
            for parameter, gradient in zip(spatial_params, direct_spatial):
                parameter.grad = gradient
            if phase == "plain_n0_warmup":
                for parameter in shared + spatial_params + decoder_params:
                    parameter.grad = None
            elif phase == "new_branch_warmup":
                for parameter in shared_plain + decoder_params:
                    parameter.grad = None
            spatial_parameter_grads = [parameter.grad for parameter in spatial_params]
            batch_spatial_active = bool(
                include_spatial and spatial_parameter_grads
                and all(
                    gradient is not None and bool(torch.isfinite(gradient).all())
                    for gradient in spatial_parameter_grads
                )
                and sum(
                    float(gradient.detach().abs().sum())
                    for gradient in spatial_parameter_grads if gradient is not None
                ) > 0.0
            )
            spatial_gradient_eligible = bool(include_spatial and (image_mask.sum(1) >= 1).any())
            if spatial_gradient_eligible:
                spatial_gradient_observed_steps += 1
                all_spatial_grad &= batch_spatial_active
            elif include_spatial:
                spatial_gradient_skipped_singleton_only_steps += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), common["global_gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if phase == "joint":
                before_losses = torch.as_tensor(
                    primary_coordinate_audit["factor_losses_before"], device=device, dtype=torch.float32
                )
                data_rng_before = {
                    "train_data_order_rng": train_generator.get_state().clone(),
                    "subbag_rng": subbag_generator.get_state().clone(),
                }
                def _same_batch_after_losses():
                    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                        after_details = model.factor_forward_details(
                            images, image_mask, include_spatial=include_spatial
                        )
                    return factor_loss_balancer.factor_losses(after_details["z_subbag"].float(), target.float())
                after_losses, rng_audit = recompute_with_isolated_rng(
                    _same_batch_after_losses, forward_rng
                )
                data_rng_after = {
                    "train_data_order_rng": train_generator.get_state(),
                    "subbag_rng": subbag_generator.get_state(),
                }
                rng_audit["data_rng_bitwise_restored"] = all(
                    torch.equal(data_rng_before[name], data_rng_after[name])
                    for name in data_rng_before
                )
                rng_audit["pass"] = bool(rng_audit.get("pass") and rng_audit["data_rng_bitwise_restored"])
                if not rng_audit["pass"]:
                    raise RuntimeError("FAMO same-batch recompute advanced a model or data RNG stream")
                factor_loss_balancer.update_logits(before_losses, after_losses, factor_weight_optimizer)
            optimizer.zero_grad(set_to_none=True)

            patient_count = len(target)
            epoch_audit.update_epoch(
                batch_size=patient_count,
                subbag_aux_loss=subbag_diag["weighted_total_loss"],
                eligible_multi_image_patients=subbag_diag["eligible_multi_image_patients"],
                subbag_gradient_final_ratio=(
                    subbag_route_rows[-1]["final_ratio"] if subbag_route_rows else None
                ),
                spatial_gradient_final_ratio=(
                    protection_rows[-1]["combined_final_auxiliary_to_primary"] if protection_rows else None
                ),
                agate_budget_loss_value=None,
                all_factor_spatial_gradients_finite_nonzero=(
                    batch_spatial_active if spatial_gradient_eligible else None
                ),
            )
            primary_sum += float(primary.detach()) * patient_count
            patient_total += patient_count
            steps += 1
            images = image_mask = target = details = None
            primary = agate_aux = subbag = spatial_objective = None
            fidelity = flip_loss = l2sp_loss = None
            flipped_patch = flipped_spatial = None
            valid = None
            targets_all = primary_grads = primary_shared_plain = primary_reference_clones = None
            primary_agate = primary_spatial = final_shared_plain = None
            direct_agate = direct_spatial = diagnostic_gradient = None
            agate_grads = subbag_grads = spatial_grads = None
            protected_l2sp = None
            if cli.steps and steps >= cli.steps:
                receipt = dict(epoch=epoch, updates=steps, loss=primary_sum/patient_total)
                (out / "smoke.json").write_text(json.dumps(receipt))
                torch.save({"model": model.state_dict()}, out / "smoke_checkpoint.pt")
                return
        if phase == "joint":
            _, epoch_l2sp_layerwise = l2sp_snapshot(named_encoder_all)
        else:
            epoch_l2sp_layerwise = None
        prediction, plain_prediction, spatial_off_prediction, truth, patient_ids, activation, agate_result = evaluate(
            model, evaluation_loader, device, amp_dtype, args.seed * 10000 + args.fold + 999
        )
        if list(patient_ids) != list(evaluation_frame.ID):
            raise RuntimeError("Evaluation did not cover the assigned partition in order")
        raw_metrics = metrics(prediction, truth)
        raw_metrics["factor_pearson_mean"], per_factor_pearson = factor_pearson_macro(prediction, truth)
        factor_pearson_macro(plain_prediction, truth)
        factor_pearson_macro(spatial_off_prediction, truth)
        validation = {metric: raw_metrics[metric] for metric in REPORT_METRICS}
        validation["factor_pearson_per_factor"] = per_factor_pearson
        validation["pearson_valid"] = True
        geometry_validation = factor_geometry_validation_audit(
            prediction, truth, factors, train_abs_tail_thresholds
        )
        validation.update({
            "patient_profile_cosine_error": float(np.mean(
                1.0 - np.sum(prediction * truth, axis=1) /
                np.maximum(np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1), 1e-12)
            )),
            "mean_tail_mae": float(np.mean([row["tail_mae"] for row in geometry_validation["per_factor"]])),
            "mean_absolute_calibration_slope_error": float(np.mean([
                abs(row["calibration_slope"] - 1.0) for row in geometry_validation["per_factor"]
            ])),
        })
        epoch_aggregate = epoch_audit.finalize()["epoch"]
        epoch_singleton_contract = singleton_gate_contract_audit(
            model, train_audit_loader, device, amp_dtype
        )
        epoch_singleton_pass = singleton_pass(epoch_singleton_contract)
        epoch_singleton_contract["pass"] = epoch_singleton_pass
        spatial_parameter_displacement(
            model.factor_v12_spatial, model.spatial_initial_state
        )
        activation.update({
            "all_factor_spatial_gradients_finite_nonzero": all_spatial_grad if include_spatial else False,
            "all_gate_gradients_finite_nonzero": all_spatial_grad if include_spatial else False,
            "spatial_gradient_eligible_observed_steps": spatial_gradient_observed_steps,
            "spatial_gradient_skipped_singleton_only_steps": spatial_gradient_skipped_singleton_only_steps,
            "subbag_epoch_aggregate": epoch_aggregate,
            "subbag_gradient_raw_ratio_mean": _mean(subbag_route_rows, "raw_ratio"),
            "subbag_gradient_raw_ratio_max": max((row["raw_ratio"] for row in subbag_route_rows), default=None),
            "subbag_gradient_final_ratio_mean": _mean(subbag_route_rows, "final_ratio"),
            "subbag_gradient_final_ratio_max": max((row["final_ratio"] for row in subbag_route_rows), default=None),
            "subbag_gradient_active_this_epoch": any(row["final_ratio"] > 0 for row in subbag_route_rows),
            "subbag_aux_loss": epoch_aggregate["subbag_aux_loss_eligible_patient_weighted"],
            "eligible_multi_image_patients": epoch_aggregate["eligible_multi_image_patients_total"],
            "subbag_gradient_final_ratio": epoch_aggregate.get("subbag_gradient_final_ratio"),
            "spatial_gradient_final_ratio": epoch_aggregate.get("spatial_gradient_final_ratio"),
            "agate_orthogonality_max_abs": agate_result["orthogonality"]["max_abs"],
            "mechanism_ever_joint": dict(mechanism_ever),
            "v29_spatial_profile": spatial_profile_audit(args.spatial_mode, epoch),
            "single_image_gate_gradient_audit": epoch_singleton_contract,
            "v31_arm_audit": model.v31_arm_audit,
            "l2sp_layerwise_latest": epoch_l2sp_layerwise,
            "v14_factor_geometry_validation": geometry_validation,
            "agate_own_parameter_gradient_audit": {
                "observed_joint_steps": mechanism_ever["agate_own_gradient_observed_steps"],
                "all_joint_steps_finite_nonzero": mechanism_ever["agate_own_gradient_all_steps_finite_nonzero"],
            },
        })
        spatial_active = (
            include_spatial
            and activation["all_factor_spatial_gradients_finite_nonzero"]
            and activation["all_gate_gradients_finite_nonzero"]
            and activation["spatial_gradient_eligible_observed_steps"] > 0
            and activation["eligible_multi_image_patients"] > 0
            and activation["subbag_gradient_active_this_epoch"] is True
            and activation["subbag_gradient_final_ratio_max"] is not None
            and activation["subbag_gradient_final_ratio_max"] > 0
            and math.isfinite(float(activation["score_span"]))
            and math.isfinite(float(activation["flip_topk_jaccard"]))
            and math.isfinite(float(activation["flip_score_cosine"]))
            and activation["common_protocol_audits_pass"] is True
            and mechanism_ever["agate_own_gradient_observed_steps"] > 0
            and mechanism_ever["agate_own_gradient_all_steps_finite_nonzero"]
        )
        if phase == "joint":
            spatial_active = (
                spatial_active and mechanism_ever["l2sp_positive"]
                and mechanism_ever["l2sp_direction_projection_absent"]
                and mechanism_ever["l2sp_route_observed_steps"] > 0
                and mechanism_ever["l2sp_route_all_steps_pass"]
                and mechanism_ever["hybrid_route_observed_steps"] > 0
                and mechanism_ever["hybrid_route_all_steps_pass"]
            )
        single_gate = activation["full_validation_gate_residual_audit"]["gate_single_image"]
        single_residual = activation["full_validation_gate_residual_audit"]["residual_single_image"]
        spatial_active = spatial_active and 0 < single_gate["mean"] <= 1 and 0 < single_gate["max"] <= 1 and math.isfinite(single_residual["rms"])
        if phase == "joint":
            spatial_active = spatial_active and mechanism_ever["mae_route_observed_steps"] == 0
        spatial_active = spatial_active and mechanism_ever["plain_vs_agate_nonalias"]
        if activation["common_protocol_audits_pass"] is not True:
            raise RuntimeError("technical padding/patient/TTA/finite validation failed")
        if not primary_bitwise_unchanged_all_steps:
            raise RuntimeError("The primary gradient changed during gradient routing.")
        if not math.isfinite(float(validation["factor_pearson_mean"])):
            raise RuntimeError("Non-finite validation mean Pearson")
        stage_technical_audit(
            epoch, phase, args.selection_min_epoch, activation, mechanism_ever)
        feasible = (
            epoch >= args.selection_min_epoch and spatial_active
            and validation["pearson_valid"] is True
        )
        check_encoder_schedule(model, core, epoch, depth,
                         optimizer.param_groups[0]["lr"], famo_cap)
        metrics_row = {"epoch": epoch, "evaluation_split": "inner_validation" if inner else "outer_train"}
        metrics_row.update({key: validation[key] for key in ("factor_pearson_mean", "profile_pcc_mean", "mse_z", "mae_z") if key in validation})
        metrics_row["checkpoint_eligible"] = bool(feasible)
        with (out / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metrics_row) + "\n")
        print(json.dumps(metrics_row), flush=True)
    if not inner:
        torch.save({"model": model.state_dict()}, out / "best_joint_checkpoint.pt")
        heldout_targets = {
            pid: torch.tensor((raw.loc[pid].to_numpy(np.float32) - factor_mean) / factor_std)
            for pid in val_frame.ID
        }
        heldout_loader = DataLoader(PatientBag(val_frame, heldout_targets, False),
                                    batch_size=args.factor_batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True, collate_fn=collate)
        from print_model.network import factor_batch
        model.eval()
        predictions, reference = [], []
        for images, mask, target, *_ in heldout_loader:
            predictions.append(factor_batch(model, images.to(device), mask.to(device)))
            reference.append(target.numpy())
        pred_z = np.concatenate(predictions)
        mean_r, per_factor = factor_pearson_macro(pred_z, np.concatenate(reference))
        oof = pd.DataFrame(pred_z * factor_std + factor_mean, columns=["pred_" + f for f in factors])
        oof.insert(0, "ID", val_frame.ID.tolist())
        oof.insert(1, "fold", cli.fold)
        for name in factors:
            oof["true_" + name] = raw.loc[val_frame.ID, name].to_numpy()
        oof.to_csv(out / "oof.csv", index=False)
        (out / "heldout_metrics.json").write_text(json.dumps({
            "evaluation_split": "discovery_A_outer_holdout", "epoch": cli.epochs,
            "n": len(val_frame), "factor_pearson_mean": mean_r, "factor_pearson_per_factor": per_factor,
        }))
