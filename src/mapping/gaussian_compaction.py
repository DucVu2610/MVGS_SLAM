"""Post-PGO Gaussian compaction.

Compaction happens only after loop registration and pose-graph optimization;
the Gaussian population used to construct landmarks is therefore unchanged.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def select_compaction_indices(
    xyz,
    opacity,
    min_opacity: float = 0.05,
    voxel_size_m: float = 0.005,
    max_gaussians: int | None = None,
    reliability=None,
):
    """Select one best Gaussian per voxel, then enforce a global cap."""
    points = np.asarray(xyz, dtype=np.float64)
    alpha = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(alpha):
        raise ValueError("xyz must be [N, 3] and opacity must have length N")
    score = alpha if reliability is None else np.asarray(reliability, dtype=np.float64).reshape(-1)
    if len(score) != len(points):
        raise ValueError("reliability must have length N")
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(alpha)
        & np.isfinite(score)
        & (alpha >= float(min_opacity))
    )
    candidates = np.flatnonzero(valid)
    # Stable secondary key by original index makes compaction reproducible.
    order = candidates[np.lexsort((candidates, -alpha[candidates], -score[candidates]))]
    if float(voxel_size_m) > 0:
        voxel = np.floor(points / float(voxel_size_m)).astype(np.int64)
        selected = []
        occupied = set()
        for index in order:
            key = tuple(voxel[index].tolist())
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(int(index))
        order = np.asarray(selected, dtype=np.int64)
    cap = None if max_gaussians is None else int(max_gaussians)
    if cap is not None:
        if cap < 1:
            raise ValueError("max_gaussians must be positive or null")
        order = order[:cap]
    return np.sort(order.astype(np.int64, copy=False))


def gaussian_model_size_bytes(gaussian_model) -> int:
    total = 0
    for name in (
        "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation",
        "_opacity", "max_radii2D", "xyz_gradient_accum", "denom",
    ):
        value = getattr(gaussian_model, name, None)
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return int(total)


def compact_gaussian_model(gaussian_model, options=None):
    """Apply opacity, voxel redundancy, and total-count gates in-place."""
    options = options or {}
    enabled = bool(options.get("enabled", False))
    before = int(gaussian_model.get_xyz().shape[0])
    stats = {
        "enabled": enabled,
        "gaussians_before_compaction": before,
        "model_bytes_before_compaction": gaussian_model_size_bytes(gaussian_model),
        "min_opacity": float(options.get("min_opacity", 0.05)),
        "voxel_size_m": float(options.get("voxel_size_m", 0.005)),
        "max_gaussians": options.get("max_gaussians"),
    }
    if not enabled or before == 0:
        stats.update({
            "gaussians_after_compaction": before,
            "removed_gaussians": 0,
            "compaction_runtime_s": 0.0,
            "model_bytes_after_compaction": stats["model_bytes_before_compaction"],
        })
        gaussian_model.compaction_stats = stats
        return gaussian_model

    started = time.perf_counter()
    xyz = gaussian_model.get_xyz().detach().cpu().numpy()
    opacity = gaussian_model.get_opacity().detach().reshape(-1).cpu().numpy()
    keep_indices = select_compaction_indices(
        xyz,
        opacity,
        min_opacity=stats["min_opacity"],
        voxel_size_m=stats["voxel_size_m"],
        max_gaussians=stats["max_gaussians"],
    )
    keep = torch.zeros(before, dtype=torch.bool, device=gaussian_model.get_xyz().device)
    keep[torch.as_tensor(keep_indices, device=keep.device)] = True
    removed = int((~keep).sum().item())
    if removed:
        gaussian_model.prune_points(~keep)
    after = int(gaussian_model.get_xyz().shape[0])
    stats.update({
        "gaussians_after_opacity_filter": int((opacity >= stats["min_opacity"]).sum()),
        "gaussians_after_compaction": after,
        "removed_gaussians": removed,
        "removed_percent": 100.0 * removed / max(before, 1),
        "compaction_runtime_s": float(time.perf_counter() - started),
        "model_bytes_after_compaction": gaussian_model_size_bytes(gaussian_model),
    })
    gaussian_model.compaction_stats = stats
    return gaussian_model
