"""Pairwise geometric-consistency filters for landmark matches."""

from __future__ import annotations

import numpy as np


def pairwise_rigidity_consistency(
    source_points,
    target_points,
    distance_threshold_m: float = 0.03,
    min_support: int = 0,
    source_scales=None,
    target_scales=None,
    scale_log_threshold: float | None = None,
    source_axes=None,
    target_axes=None,
    relative_axis_cos_threshold: float | None = None,
):
    """Return matches supported by rigid pairwise relationships.

    Distance preservation is the default production gate.  Per-landmark scale
    agreement and sign-invariant relative-axis agreement are optional ablation
    terms and are activated only when their metadata and thresholds are given.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be [N, 3]")
    count = len(source)
    if count == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    if float(distance_threshold_m) <= 0 or int(min_support) <= 0:
        return np.ones(count, dtype=bool), np.full(count, count - 1, dtype=np.int64)

    source_distances = np.linalg.norm(source[:, None] - source[None, :], axis=2)
    target_distances = np.linalg.norm(target[:, None] - target[None, :], axis=2)
    compatible = (
        np.abs(source_distances - target_distances)
        <= float(distance_threshold_m)
    )
    np.fill_diagonal(compatible, False)

    if scale_log_threshold is not None and source_scales is not None and target_scales is not None:
        source_scale = np.maximum(np.asarray(source_scales, dtype=np.float64), 1e-12)
        target_scale = np.maximum(np.asarray(target_scales, dtype=np.float64), 1e-12)
        if source_scale.shape != target_scale.shape or source_scale.shape[0] != count:
            raise ValueError("matched scale metadata has an invalid shape")
        scale_error = np.linalg.norm(
            np.log(np.sort(source_scale, axis=1))
            - np.log(np.sort(target_scale, axis=1)), axis=1)
        scale_ok = scale_error <= float(scale_log_threshold)
        compatible &= scale_ok[:, None] & scale_ok[None, :]

    if (relative_axis_cos_threshold is not None and source_axes is not None
            and target_axes is not None):
        source_axes = np.asarray(source_axes, dtype=np.float64)
        target_axes = np.asarray(target_axes, dtype=np.float64)
        if source_axes.shape != (count, 3, 3) or target_axes.shape != (count, 3, 3):
            raise ValueError("matched axes must have shape [N, 3, 3]")
        threshold = float(relative_axis_cos_threshold)
        for first_axis in range(3):
            for second_axis in range(3):
                source_cos = np.abs(
                    source_axes[:, :, first_axis]
                    @ source_axes[:, :, second_axis].T)
                target_cos = np.abs(
                    target_axes[:, :, first_axis]
                    @ target_axes[:, :, second_axis].T)
                compatible &= np.abs(source_cos - target_cos) <= threshold

    support = compatible.sum(axis=1).astype(np.int64)
    return support >= int(min_support), support
