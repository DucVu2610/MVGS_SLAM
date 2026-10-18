"""Dependency-light invariants for the fixed-loop Single/Multi benchmark."""

from __future__ import annotations

from collections import Counter
import numpy as np

from src.utils.dataset_utils import resolve_agent_input


def find_submap_index(submaps, frame_id: int) -> int:
    """Map a frame to finalized checkpoints using inclusive saved bounds."""
    matches = [
        index for index, submap in enumerate(submaps)
        if int(submap["submap_start_frame_id"]) <= int(frame_id)
        <= int(submap["submap_end_frame_id"])
    ]
    if len(matches) != 1:
        raise KeyError(
            f"frame {frame_id} maps to {len(matches)} finalized submaps")
    return matches[0]


def validate_finalized_pose_ids(submaps, saved_keyframe_ids, optimized_poses, label=""):
    """Require exact ID/order agreement between submaps and optimized poses."""
    expected = np.concatenate([
        np.asarray(submap["keyframe_ids"]).reshape(-1).astype(np.int64)
        for submap in submaps
    ])
    saved = np.asarray(saved_keyframe_ids).reshape(-1).astype(np.int64)
    optimized = np.asarray(optimized_poses)
    prefix = f"{label}: " if label else ""
    if optimized.ndim != 3 or optimized.shape[1:] != (4, 4):
        raise ValueError(f"{prefix}optimized poses must have shape [N, 4, 4]")
    if len(saved) != len(optimized):
        raise ValueError(
            f"{prefix}{len(saved)} pose IDs != {len(optimized)} poses")
    if not np.array_equal(saved, expected):
        raise ValueError(
            f"{prefix}optimized pose IDs do not match finalized submap IDs")
    if len(np.unique(saved)) != len(saved):
        raise ValueError(f"{prefix}optimized pose IDs contain duplicates")
    return expected


def landmark_view(shared_cache, mode: str, k: int):
    """Slice the same ordered Gaussian identities for Single and Multi."""
    mode = str(mode).lower()
    if mode not in {"single", "multiview"}:
        raise ValueError("mode must be single or multiview")
    count = min(int(k), len(shared_cache["xyz"]))
    if count < 3:
        raise ValueError(f"only {count} shared landmarks are available")
    result = {}
    for key, value in shared_cache.items():
        if hasattr(value, "shape") and len(value.shape) and value.shape[0] == len(
                shared_cache["xyz"]):
            result[key] = value[:count]
        else:
            result[key] = value
    descriptor_key = (
        "descriptor_single_best_view" if mode == "single"
        else "descriptor_multi_view"
    )
    result["descriptors"] = result[descriptor_key]
    result["descriptor_mode"] = mode
    return result


def summarize_trials(rows):
    """Pass rate uses all trials; pose errors are conditional on valid output."""
    rows = list(rows)
    attempts = len(rows)
    valid = [row for row in rows if bool(row.get("valid_estimate", False))]
    errors = Counter(
        str(row.get("failure_reason") or "unspecified")
        for row in rows if not bool(row.get("passed", False)))
    metrics = {}
    for field in ("rotation_error_deg", "translation_error_cm"):
        values = np.asarray([
            row[field] for row in valid
            if row.get(field) is not None and np.isfinite(row[field])
        ], dtype=np.float64)
        metrics[f"conditional_{field}_mean"] = (
            float(values.mean()) if len(values) else None)
        metrics[f"conditional_{field}_median"] = (
            float(np.median(values)) if len(values) else None)
    for stage, valid_field in (
            ("coarse", "coarse_valid"), ("final", "valid_estimate")):
        stage_valid = [row for row in rows if bool(row.get(valid_field, False))]
        for metric in ("rotation_error_deg", "translation_error_cm"):
            field = metric if stage == "final" else f"coarse_{metric}"
            values = np.asarray([
                row[field] for row in stage_valid
                if row.get(field) is not None and np.isfinite(row[field])
            ], dtype=np.float64)
            metrics[f"conditional_{stage}_{metric}_mean"] = (
                float(values.mean()) if len(values) else None)
            metrics[f"conditional_{stage}_{metric}_median"] = (
                float(np.median(values)) if len(values) else None)
    return {
        "attempts": attempts,
        "valid_estimates": len(valid),
        "passed": sum(bool(row.get("passed", False)) for row in rows),
        "pass_rate_percent": (
            100.0 * sum(bool(row.get("passed", False)) for row in rows) / attempts
            if attempts else 0.0),
        "failure_reasons": dict(sorted(errors.items())),
        **metrics,
    }
