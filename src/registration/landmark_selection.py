"""Reliability scoring and spatial selection for Gaussian landmarks."""

from __future__ import annotations

import numpy as np


def landmark_reliability(
    stability,
    observation_count,
    mean_depth_residual,
    opacity,
    depth_temperature_m: float,
    mode: str = "paper",
):
    """Compute landmark reliability using the paper or legacy formula.

    Opacity is applied once at landmark level.  It is deliberately not part of
    the per-view descriptor aggregation weights because multiplying every view
    of one Gaussian by the same opacity does not change the normalized result.
    """
    stability = np.clip(np.asarray(stability, dtype=np.float64), 0.0, 1.0)
    observations = np.asarray(observation_count, dtype=np.float64)
    residual = np.asarray(mean_depth_residual, dtype=np.float64)
    opacity = np.clip(np.asarray(opacity, dtype=np.float64), 0.0, 1.0)
    tau = float(depth_temperature_m)
    if tau <= 0:
        raise ValueError("depth_temperature_m must be positive")
    mode = str(mode).lower()
    observations = np.maximum(observations, 0.0)
    residual = np.maximum(residual, 0.0)
    if mode == "paper":
        return (
            stability
            * np.log1p(observations)
            * (0.5 + 0.5 * opacity)
            / (1.0 + residual / tau)
        )
    if mode == "legacy_exp":
        observation_term = 1.0 - np.exp(-observations / 2.0)
        residual_term = np.exp(-residual / tau)
        return stability * observation_term * residual_term * opacity
    raise ValueError("reliability_mode must be 'paper' or 'legacy_exp'")


def select_spatially_balanced_landmarks(
    xyz,
    reliability,
    max_landmarks: int,
    voxel_size_m: float,
):
    """Select high-reliability landmarks while covering occupied voxels.

    The first pass takes the best landmark per voxel.  If fewer than ``K``
    voxels are occupied, a second pass fills the budget in reliability order.
    This mirrors the validated notebook selection without retaining notebook
    state or hard-coded scene parameters.
    """
    points = np.asarray(xyz, dtype=np.float64)
    scores = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(points) != len(scores):
        raise ValueError("xyz and reliability lengths differ")
    limit = int(max_landmarks)
    if limit < 1:
        raise ValueError("max_landmarks must be positive")
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
    valid_indices = np.flatnonzero(finite)
    order = valid_indices[np.argsort(-scores[valid_indices], kind="stable")]
    if float(voxel_size_m) <= 0:
        return order[:limit].astype(np.int64, copy=False)

    voxels = np.floor(points / float(voxel_size_m)).astype(np.int64)
    selected = []
    selected_set = set()
    occupied = set()
    for index in order:
        voxel = tuple(voxels[index].tolist())
        if voxel in occupied:
            continue
        occupied.add(voxel)
        selected.append(int(index))
        selected_set.add(int(index))
        if len(selected) == limit:
            break

    if len(selected) < min(limit, len(order)):
        for index in order:
            index = int(index)
            if index in selected_set:
                continue
            selected.append(index)
            if len(selected) == limit:
                break
    return np.asarray(selected, dtype=np.int64)


def select_spatially_covered_correspondences(
    source_points,
    target_points,
    quality,
    max_correspondences: int = 96,
    voxel_size_m: float = 0.10,
    max_per_source_voxel: int = 2,
    max_per_target_voxel: int = 2,
    min_keep: int = 8,
):
    """Return indices with source-and-target voxel coverage.

    The coverage pass is deliberately run for every input size.  V2 returned
    early below the cap, which is the coverage bug called out in the V3
    notebook.  Only a minimal confidence-ordered fill is allowed when strict
    coverage leaves too few correspondences for pose estimation.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    scores = np.asarray(quality, dtype=np.float64).reshape(-1)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be [N, 3]")
    if len(source) != len(scores):
        raise ValueError("correspondence quality length differs from points")
    if len(source) == 0:
        return np.empty(0, dtype=np.int64), {
            "input_pairs": 0,
            "strict_coverage_pairs": 0,
            "filled_pairs": 0,
            "output_pairs": 0,
        }
    if min(max_correspondences, max_per_source_voxel,
           max_per_target_voxel, min_keep) < 1:
        raise ValueError("coverage limits must be positive")
    if float(voxel_size_m) <= 0:
        raise ValueError("voxel_size_m must be positive")

    finite = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(scores)
    )
    order = np.flatnonzero(finite)
    order = order[np.argsort(-scores[order], kind="stable")]
    source_voxels = np.floor(source / float(voxel_size_m)).astype(np.int64)
    target_voxels = np.floor(target / float(voxel_size_m)).astype(np.int64)
    source_counts, target_counts = {}, {}
    kept = []
    limit = min(int(max_correspondences), len(order))
    for index in order:
        source_voxel = tuple(source_voxels[index].tolist())
        target_voxel = tuple(target_voxels[index].tolist())
        if source_counts.get(source_voxel, 0) >= int(max_per_source_voxel):
            continue
        if target_counts.get(target_voxel, 0) >= int(max_per_target_voxel):
            continue
        kept.append(int(index))
        source_counts[source_voxel] = source_counts.get(source_voxel, 0) + 1
        target_counts[target_voxel] = target_counts.get(target_voxel, 0) + 1
        if len(kept) >= limit:
            break

    strict_count = len(kept)
    minimum = min(int(min_keep), limit)
    if len(kept) < minimum:
        selected = set(kept)
        for index in order:
            index = int(index)
            if index in selected:
                continue
            kept.append(index)
            selected.add(index)
            if len(kept) >= minimum:
                break
    indices = np.asarray(kept, dtype=np.int64)
    return indices, {
        "input_pairs": int(len(source)),
        "strict_coverage_pairs": int(strict_count),
        "filled_pairs": int(len(indices) - strict_count),
        "output_pairs": int(len(indices)),
    }
