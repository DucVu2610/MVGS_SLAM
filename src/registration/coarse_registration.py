"""Gaussian-to-Gaussian coarse registration for MAGMA-SLAM."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .gaussian_matching import mutual_nearest_neighbors, mutual_topk_candidates
from .geometric_consistency import pairwise_rigidity_consistency
from .landmark_selection import select_spatially_covered_correspondences
from .multiview_descriptor import (
    build_gaussian_landmark_descriptors,
    descriptor_cache_key,
    points_in_camera_frame,
)


def is_valid_rigid_transform(transform, atol: float = 1e-4) -> bool:
    """Validate a finite, proper SE(3) matrix without repairing bad output."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
        and np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol)
    )


def estimate_rigid_transform(source_points, target_points, weights=None) -> np.ndarray:
    """Least-squares source-to-target rigid transform (Kabsch/Umeyama)."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be [N, 3]")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("correspondences contain non-finite values")
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(source) or not np.isfinite(weights).all():
        raise ValueError("weights must be finite and match correspondences")
    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()
    source_centroid = np.sum(source * weights[:, None], axis=0)
    target_centroid = np.sum(target * weights[:, None], axis=0)
    covariance = (
        (source - source_centroid) * weights[:, None]
    ).T @ (target - target_centroid)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    if not is_valid_rigid_transform(transform):
        raise ValueError("Kabsch produced an invalid rigid rotation")
    return transform


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _non_degenerate_triplet(points, epsilon=1e-8):
    first = points[1] - points[0]
    second = points[2] - points[0]
    return np.linalg.norm(np.cross(first, second)) > float(epsilon)


def ransac_rigid_transform(
    source_points,
    target_points,
    distance_threshold_m: float = 0.05,
    max_iterations: int = 5000,
    seed: int = 0,
    refit_inliers: bool = True,
    confidence=None,
    scoring: str = "count",
    prosac: bool = False,
    tight_threshold_m: float | None = None,
    min_iterations: int = 0,
    early_inlier_ratio: float = 1.0,
    early_median_m: float = 0.0,
    return_diagnostics: bool = False,
):
    """Robust rigid estimation with optional V3 confidence/PROSAC scoring."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be [N, 3]")
    if len(source) < 3:
        result = (None, np.zeros(len(source), dtype=bool))
        if return_diagnostics:
            return (*result, {"score": float("-inf"), "iterations": 0})
        return result
    threshold = float(distance_threshold_m)
    if threshold <= 0:
        raise ValueError("distance_threshold_m must be positive")
    generator = np.random.default_rng(int(seed))
    if confidence is None:
        confidence = np.ones(len(source), dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if len(confidence) != len(source) or not np.isfinite(confidence).all():
        raise ValueError("confidence must be finite and match correspondences")
    confidence = np.clip(confidence, 1e-8, None)
    order = np.argsort(-confidence, kind="stable")
    source, target, confidence = source[order], target[order], confidence[order]
    best_mask = np.zeros(len(source), dtype=bool)
    best_error = np.inf
    best_score = float("-inf")
    best_transform = None
    iterations_run = 0
    tight_threshold = float(
        tight_threshold_m if tight_threshold_m is not None
        else min(threshold, 0.03))

    def consensus(residual):
        inliers = residual <= threshold
        tight = residual <= tight_threshold
        if not np.any(inliers):
            return inliers, float("-inf"), float("inf")
        median = float(np.median(residual[inliers]))
        if str(scoring).lower() in {"v3", "composite", "weighted"}:
            ratio = float(inliers.mean())
            tight_ratio = float(tight.mean())
            weighted_ratio = float(
                confidence[inliers].sum() / confidence.sum())
            residual_quality = 1.0 - min(median / threshold, 1.0)
            score = (
                0.50 * weighted_ratio
                + 0.20 * ratio
                + 0.20 * tight_ratio
                + 0.10 * residual_quality
            )
        else:
            score = float(inliers.sum())
        return inliers, score, median

    iterations = max(1, int(max_iterations))
    for iteration in range(iterations):
        iterations_run = iteration + 1
        if prosac:
            fraction = iteration / max(1, iterations - 1)
            pool_size = min(
                len(source), min(12, len(source))
                + int(fraction * max(0, len(source) - min(12, len(source)))))
            probabilities = confidence[:pool_size]
            probabilities = probabilities / probabilities.sum()
            sample = generator.choice(
                pool_size, size=3, replace=False, p=probabilities)
        else:
            sample = generator.choice(len(source), size=3, replace=False)
        if (not _non_degenerate_triplet(source[sample])
                or not _non_degenerate_triplet(target[sample])):
            continue
        try:
            hypothesis = estimate_rigid_transform(
                source[sample], target[sample], confidence[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(
            transform_points(source, hypothesis) - target, axis=1)
        inliers, score, mean_error = consensus(residual)
        count = int(inliers.sum())
        if count < 3:
            continue
        if score > best_score or (
                np.isclose(score, best_score) and mean_error < best_error):
            best_mask = inliers
            best_error = mean_error
            best_score = score
            best_transform = hypothesis
            if (iterations_run >= int(min_iterations)
                    and float(inliers.mean()) >= float(early_inlier_ratio)
                    and mean_error <= float(early_median_m)):
                break
    if best_transform is None:
        result = (None, np.zeros(len(order), dtype=bool))
        if return_diagnostics:
            return (*result, {"score": best_score, "iterations": iterations_run})
        return result
    if refit_inliers and int(best_mask.sum()) >= 3:
        best_transform = estimate_rigid_transform(
            source[best_mask], target[best_mask], confidence[best_mask])
        residual = np.linalg.norm(
            transform_points(source, best_transform) - target, axis=1)
        best_mask = residual <= threshold
        if int(best_mask.sum()) >= 3:
            # Refit on the complete consensus, rather than the sampled triplet.
            best_transform = estimate_rigid_transform(
                source[best_mask], target[best_mask], confidence[best_mask])
            residual = np.linalg.norm(
                transform_points(source, best_transform) - target, axis=1)
            best_mask = residual <= threshold
    if (not is_valid_rigid_transform(best_transform)
            or int(best_mask.sum()) < 3):
        best_transform = None
        best_mask[:] = False
    if best_transform is not None:
        residual = np.linalg.norm(
            transform_points(source, best_transform) - target, axis=1)
        best_mask, best_score, best_error = consensus(residual)
    restored_mask = np.zeros(len(order), dtype=bool)
    restored_mask[order] = best_mask
    diagnostics = {
        "score": float(best_score),
        "iterations": int(iterations_run),
        "inlier_ratio_5cm": float(best_mask.mean()),
        "inlier_ratio_3cm": float((residual <= tight_threshold).mean()),
        "median_inlier_residual_m": (
            float(best_error)
            if best_transform is not None and np.any(best_mask) else None),
    }
    if return_diagnostics:
        return best_transform, restored_mask, diagnostics
    return best_transform, restored_mask


def _pair_quality(source_landmarks, target_landmarks, source_indices,
                  target_indices, similarity, support):
    similarity = np.clip(np.asarray(similarity, dtype=np.float64), 0.0, 1.0)
    support = np.asarray(support, dtype=np.float64)
    support_quality = support / max(float(support.max(initial=0)), 1.0)
    source_reliability = np.asarray(
        source_landmarks.get("reliability", np.ones(len(source_landmarks["xyz"]))))
    target_reliability = np.asarray(
        target_landmarks.get("reliability", np.ones(len(target_landmarks["xyz"]))))
    metadata = np.sqrt(np.clip(
        source_reliability[source_indices] * target_reliability[target_indices],
        0.0, 1.0))
    return 0.45 * similarity + 0.35 * support_quality + 0.20 * metadata


def _greedy_unique(source_indices, target_indices, quality):
    order = np.argsort(-np.asarray(quality), kind="stable")
    used_source, used_target, keep = set(), set(), []
    for index in order:
        source = int(source_indices[index])
        target = int(target_indices[index])
        if source in used_source or target in used_target:
            continue
        used_source.add(source)
        used_target.add(target)
        keep.append(int(index))
    return np.asarray(keep, dtype=np.int64)


def _frame_pose(submap, frame_id):
    offset = int(frame_id) - int(submap["submap_start_frame_id"])
    poses = submap["submap_c2ws"]
    if offset < 0 or offset >= len(poses):
        raise IndexError(f"frame {frame_id} is outside its submap pose array")
    pose = poses[offset]
    return np.asarray(
        pose.detach().cpu().numpy() if hasattr(pose, "detach") else pose,
        dtype=np.float64,
    )


def register_landmark_sets(
    source_landmarks: Dict[str, Any],
    target_landmarks: Dict[str, Any],
    source_c2w,
    target_c2w,
    options: Dict[str, Any] | None = None,
):
    """Match shared Gaussian landmarks with V3 coverage and hard recovery."""
    options = options or {}
    source_points = points_in_camera_frame(source_landmarks["xyz"], source_c2w)
    target_points = points_in_camera_frame(target_landmarks["xyz"], target_c2w)
    diagnostics = {
        "num_source_features": int(len(source_points)),
        "num_target_features": int(len(target_points)),
        "num_feature_matches": 0,
        "num_mutual_matches": 0,
        "num_confidence_matches": 0,
        "num_depth_valid_correspondences": 0,
        "num_geometry_consistent_correspondences": 0,
        "num_correspondences": 0,
        "num_ransac_inliers": 0,
        "ransac_inlier_ratio": 0.0,
        "correspondence_retention_ratio": 0.0,
        "mean_match_similarity": None,
        "mean_match_margin": None,
        "mean_compatibility_support": None,
        "coverage_input_pairs": 0,
        "coverage_strict_pairs": 0,
        "coverage_filled_pairs": 0,
        "fallback_triggered": False,
        "fallback_reason": None,
        "fallback_selected": False,
        "selected_stage": "failed",
        "failure_reason": None,
    }

    match_start = time.perf_counter()
    fast_matches = mutual_nearest_neighbors(
        source_landmarks["descriptors"],
        target_landmarks["descriptors"],
        similarity_threshold=float(options.get("similarity_threshold", -1.0)),
        ambiguity_margin=float(options.get("ambiguity_margin", 0.0)),
    )
    diagnostics["matching_runtime_s"] = float(time.perf_counter() - match_start)
    fast_source_indices = fast_matches.source_indices.cpu().numpy()
    diagnostics["num_feature_matches"] = int(fast_matches.num_mutual)
    diagnostics["num_mutual_matches"] = int(fast_matches.num_mutual)
    diagnostics["num_confidence_matches"] = int(len(fast_source_indices))
    diagnostics["num_depth_valid_correspondences"] = int(len(fast_source_indices))
    if len(fast_source_indices):
        diagnostics["mean_match_similarity"] = float(
            fast_matches.similarity.mean().item())
        diagnostics["mean_match_margin"] = float(
            fast_matches.ambiguity_margin.mean().item())

    minimum = int(options.get("min_correspondences", 8))
    geometry_mode = str(options.get("geometric_consistency", "distance")).lower()

    def run_path(matches, stage, enforce_unique=False):
        source_indices = matches.source_indices.detach().cpu().numpy()
        target_indices = matches.target_indices.detach().cpu().numpy()
        similarities = matches.similarity.detach().cpu().numpy()
        if len(source_indices) < minimum:
            return None, {"failure_reason": "insufficient_descriptor_matches"}
        matched_source = source_points[source_indices]
        matched_target = target_points[target_indices]
        geometry_start = time.perf_counter()
        if geometry_mode in {"none", "off", "false"}:
            keep = np.ones(len(matched_source), dtype=bool)
            support = np.full(
                len(matched_source), max(0, len(matched_source) - 1),
                dtype=np.int64)
        else:
            use_scale = "scale" in geometry_mode
            use_axis = "axis" in geometry_mode
            keep, support = pairwise_rigidity_consistency(
                matched_source,
                matched_target,
                distance_threshold_m=float(
                    options.get("compatibility_threshold_m", 0.03)),
                min_support=int(options.get("min_compatibility_support", 16)),
                source_scales=(
                    np.asarray(source_landmarks["scales"])[source_indices]
                    if use_scale else None),
                target_scales=(
                    np.asarray(target_landmarks["scales"])[target_indices]
                    if use_scale else None),
                scale_log_threshold=(
                    float(options.get("scale_log_threshold", 0.5))
                    if use_scale else None),
                source_axes=(
                    np.asarray(source_landmarks["axes"])[source_indices]
                    if use_axis else None),
                target_axes=(
                    np.asarray(target_landmarks["axes"])[target_indices]
                    if use_axis else None),
                relative_axis_cos_threshold=(
                    float(options.get("relative_axis_cos_threshold", 0.15))
                    if use_axis else None),
            )
        geometry_runtime = float(time.perf_counter() - geometry_start)
        quality = _pair_quality(
            source_landmarks, target_landmarks,
            source_indices, target_indices, similarities, support)
        if (enforce_unique and matches.margin_quality is not None
                and matches.rank_quality is not None):
            margin_quality = matches.margin_quality.detach().cpu().numpy()
            rank_quality = matches.rank_quality.detach().cpu().numpy()
            quality = (
                0.60 * quality
                + 0.25 * margin_quality
                + 0.15 * rank_quality
            )
        source_indices, target_indices = source_indices[keep], target_indices[keep]
        matched_source, matched_target = matched_source[keep], matched_target[keep]
        support, quality = support[keep], quality[keep]
        if enforce_unique and len(source_indices):
            unique = _greedy_unique(source_indices, target_indices, quality)
            source_indices, target_indices = (
                source_indices[unique], target_indices[unique])
            matched_source, matched_target = (
                matched_source[unique], matched_target[unique])
            support, quality = support[unique], quality[unique]
            prefilter_limit = int(options.get(
                "hard_case_prefilter_max_correspondences", 180))
            if len(quality) > prefilter_limit:
                prefilter = np.argsort(
                    -quality, kind="stable")[:prefilter_limit]
                source_indices, target_indices = (
                    source_indices[prefilter], target_indices[prefilter])
                matched_source, matched_target = (
                    matched_source[prefilter], matched_target[prefilter])
                support, quality = support[prefilter], quality[prefilter]
        geometry_pairs = int(len(matched_source))
        mean_support = float(np.mean(support)) if len(support) else None
        if geometry_pairs < minimum:
            return None, {
                "failure_reason": "insufficient_geometry_consistent_matches",
                "geometry_runtime_s": geometry_runtime,
                "geometry_pairs": geometry_pairs,
                "mean_support": mean_support,
            }

        coverage_indices, coverage = select_spatially_covered_correspondences(
            matched_source,
            matched_target,
            quality,
            max_correspondences=int(options.get(
                "hard_case_max_correspondences" if enforce_unique
                else "max_ransac_correspondences",
                144 if enforce_unique else 96)),
            voxel_size_m=float(options.get("coverage_voxel_size_m", 0.10)),
            max_per_source_voxel=int(options.get(
                "hard_case_max_per_source_voxel" if enforce_unique
                else "max_per_source_voxel",
                3 if enforce_unique else 2)),
            max_per_target_voxel=int(options.get(
                "hard_case_max_per_target_voxel" if enforce_unique
                else "max_per_target_voxel",
                3 if enforce_unique else 2)),
            min_keep=minimum,
        )
        matched_source = matched_source[coverage_indices]
        matched_target = matched_target[coverage_indices]
        quality = quality[coverage_indices]
        if len(matched_source) < minimum:
            return None, {
                "failure_reason": "insufficient_spatial_coverage",
                "geometry_runtime_s": geometry_runtime,
                "geometry_pairs": geometry_pairs,
                "mean_support": mean_support,
                "coverage": coverage,
            }
        ransac_start = time.perf_counter()
        transform, inlier_mask, ransac = ransac_rigid_transform(
            matched_source,
            matched_target,
            distance_threshold_m=float(
                options.get("ransac_distance_threshold_m", 0.05)),
            max_iterations=int(options.get("ransac_max_iterations", 5000)),
            seed=int(options.get("ransac_seed", 0)),
            refit_inliers=bool(options.get("refit_ransac_inliers", True)),
            confidence=quality,
            scoring=str(options.get("ransac_scoring", "composite")),
            prosac=bool(options.get("prosac_sampling", True)),
            tight_threshold_m=float(
                options.get("ransac_tight_threshold_m", 0.03)),
            min_iterations=int(options.get("ransac_min_iterations", 120)),
            early_inlier_ratio=float(
                options.get("ransac_early_inlier_ratio", 0.82)),
            early_median_m=float(
                options.get("ransac_early_median_m", 0.02)),
            return_diagnostics=True,
        )
        ransac_runtime = float(time.perf_counter() - ransac_start)
        inlier_count = int(inlier_mask.sum())
        if transform is None or inlier_count < int(
                options.get("min_ransac_inliers", 3)):
            transform = None
            failure_reason = "ransac_failed"
        elif not is_valid_rigid_transform(transform):
            transform = None
            failure_reason = "invalid_rigid_rotation"
        else:
            failure_reason = None
        return transform, {
            "failure_reason": failure_reason,
            "stage": stage,
            "geometry_runtime_s": geometry_runtime,
            "geometry_pairs": geometry_pairs,
            "mean_support": mean_support,
            "coverage": coverage,
            "selected_pairs": int(len(matched_source)),
            "inlier_count": inlier_count,
            "inlier_ratio": (
                float(inlier_mask.mean()) if len(inlier_mask) else 0.0),
            "ransac_runtime_s": ransac_runtime,
            "ransac": ransac,
        }

    fast_transform, fast = run_path(fast_matches, "fast")
    fallback_reasons = []
    if fast_transform is None:
        fallback_reasons.append(fast.get("failure_reason", "fast_failed"))
    elif float(fast.get("inlier_ratio", 0.0)) < float(
            options.get("hard_case_min_inlier_ratio", 0.80)):
        fallback_reasons.append("fast_inlier_ratio_low")
    elif float(fast.get("ransac", {}).get(
            "inlier_ratio_3cm", 0.0)) < float(options.get(
                "hard_case_min_tight_inlier_ratio", 0.70)):
        fallback_reasons.append("fast_tight_inlier_ratio_low")
    diagnostics["fallback_triggered"] = bool(fallback_reasons)
    diagnostics["fallback_reason"] = "|".join(fallback_reasons) or None
    selected_transform, selected = fast_transform, fast

    if diagnostics["fallback_triggered"] and bool(
            options.get("hard_case_recovery", True)):
        hard_start = time.perf_counter()
        hard_matches = mutual_topk_candidates(
            source_landmarks["descriptors"],
            target_landmarks["descriptors"],
            top_k=int(options.get("hard_case_top_k", 5)),
            similarity_threshold=float(options.get(
                "hard_case_similarity_threshold",
                options.get("similarity_threshold", -1.0))),
        )
        diagnostics["matching_runtime_s"] += float(
            time.perf_counter() - hard_start)
        diagnostics["hard_candidate_pairs"] = int(
            len(hard_matches.source_indices))
        hard_transform, hard = run_path(
            hard_matches, "hard_recovery", enforce_unique=True)
        fast_score = float(fast.get("ransac", {}).get("score", -np.inf))
        hard_score = float(hard.get("ransac", {}).get("score", -np.inf))
        if hard_transform is not None and (
                fast_transform is None
                or hard_score >= fast_score
                + float(options.get("hard_case_min_score_gain", 0.0))):
            selected_transform, selected = hard_transform, hard
            diagnostics["fallback_selected"] = True

    diagnostics["selected_stage"] = selected.get("stage", "failed")
    diagnostics["failure_reason"] = selected.get("failure_reason")
    diagnostics["geometric_consistency_runtime_s"] = float(
        selected.get("geometry_runtime_s", 0.0))
    diagnostics["mean_compatibility_support"] = selected.get("mean_support")
    diagnostics["num_geometry_consistent_correspondences"] = int(
        selected.get("geometry_pairs", 0))
    diagnostics["num_correspondences"] = int(selected.get("selected_pairs", 0))
    coverage = selected.get("coverage", {})
    diagnostics["coverage_input_pairs"] = int(coverage.get("input_pairs", 0))
    diagnostics["coverage_strict_pairs"] = int(
        coverage.get("strict_coverage_pairs", 0))
    diagnostics["coverage_filled_pairs"] = int(
        coverage.get("filled_pairs", 0))
    diagnostics["num_ransac_inliers"] = int(selected.get("inlier_count", 0))
    diagnostics["ransac_inlier_ratio"] = float(
        selected.get("inlier_ratio", 0.0))
    diagnostics["ransac_runtime_s"] = float(
        selected.get("ransac_runtime_s", 0.0))
    diagnostics["ransac_score"] = selected.get("ransac", {}).get("score")
    diagnostics["ransac_iterations"] = selected.get(
        "ransac", {}).get("iterations", 0)
    diagnostics["correspondence_retention_ratio"] = (
        float(diagnostics["num_correspondences"]) / fast_matches.num_mutual
        if fast_matches.num_mutual else 0.0)
    if (selected_transform is None
            or not is_valid_rigid_transform(selected_transform)):
        diagnostics["failure_reason"] = (
            diagnostics["failure_reason"] or "invalid_or_missing_transform")
        return None, diagnostics
    return selected_transform, diagnostics


def register_gaussian_landmarks(
    source_submap,
    target_submap,
    source_frame_id: int,
    target_frame_id: int,
    source_dataset,
    target_dataset,
    feature_extractor,
    descriptor_mode: str,
    options: Dict[str, Any] | None = None,
    cache: Dict[Any, Any] | None = None,
):
    """Build/cache two Gaussian landmark sets and estimate source-to-target pose."""
    options = options or {}
    cache = cache if cache is not None else {}

    def landmarks(submap, dataset):
        cache_key = descriptor_cache_key(
            submap, dataset, feature_extractor, options)
        if cache_key not in cache:
            cache[cache_key] = build_gaussian_landmark_descriptors(
                submap, dataset, feature_extractor, "multiview", options)
        shared = cache[cache_key]
        view = dict(shared)
        key = (
            "descriptor_single_best_view" if descriptor_mode == "single"
            else "descriptor_multi_view"
        )
        view["descriptors"] = shared[key]
        view["descriptor_mode"] = descriptor_mode
        return view

    descriptor_start = time.perf_counter()
    source_landmarks = landmarks(source_submap, source_dataset)
    target_landmarks = landmarks(target_submap, target_dataset)
    descriptor_runtime = float(time.perf_counter() - descriptor_start)
    transform, diagnostics = register_landmark_sets(
        source_landmarks,
        target_landmarks,
        _frame_pose(source_submap, source_frame_id),
        _frame_pose(target_submap, target_frame_id),
        options,
    )
    diagnostics["descriptor_runtime_s"] = descriptor_runtime
    diagnostics["descriptor_mode"] = descriptor_mode
    diagnostics["num_source_landmarks"] = len(source_landmarks["xyz"])
    diagnostics["num_target_landmarks"] = len(target_landmarks["xyz"])
    for prefix, landmark_set in (
            ("source", source_landmarks), ("target", target_landmarks)):
        for name, value in landmark_set.get("diagnostics", {}).items():
            diagnostics[f"{prefix}_{name}"] = value
    # Backward-compatible aggregate fields used by the existing logger.
    diagnostics["num_gaussian_landmarks"] = min(
        diagnostics["num_source_landmarks"], diagnostics["num_target_landmarks"])
    diagnostics["gaussian_min_opacity"] = float(
        options.get("gaussian_min_opacity", 0.05))
    diagnostics["gaussian_depth_tolerance_m"] = float(
        options.get("gaussian_depth_tolerance_m", 0.05))
    diagnostics["gaussian_max_landmarks"] = int(
        options.get("max_landmarks", options.get("gaussian_max_landmarks", 512)))
    diagnostics["gaussian_refit_inliers"] = bool(
        options.get("refit_ransac_inliers", True))
    return transform, diagnostics
