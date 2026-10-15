"""Point-to-plane ICP and loop-admission helpers."""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d


def _pose(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def submap_frame_c2w(submap, frame_id):
    offset = int(frame_id) - int(submap["submap_start_frame_id"])
    poses = submap["submap_c2ws"]
    if offset < 0 or offset >= len(poses):
        raise IndexError(f"frame {frame_id} is outside its submap")
    return _pose(poses[offset])


def loop_to_submap_start_transform(
    loop_transform, source_submap, target_submap,
    source_frame_id, target_frame_id,
):
    """Convert source-loop->target-loop into source-start->target-start."""
    source_start = _pose(source_submap["submap_c2ws"][0])
    target_start = _pose(target_submap["submap_c2ws"][0])
    source_loop = submap_frame_c2w(source_submap, source_frame_id)
    target_loop = submap_frame_c2w(target_submap, target_frame_id)
    return (
        np.linalg.inv(target_start)
        @ target_loop
        @ _pose(loop_transform)
        @ np.linalg.inv(source_loop)
        @ source_start
    )


def submap_start_to_loop_transform(
    start_transform, source_submap, target_submap,
    source_frame_id, target_frame_id,
):
    """Convert source-start->target-start into source-loop->target-loop."""
    source_start = _pose(source_submap["submap_c2ws"][0])
    target_start = _pose(target_submap["submap_c2ws"][0])
    source_loop = submap_frame_c2w(source_submap, source_frame_id)
    target_loop = submap_frame_c2w(target_submap, target_frame_id)
    source_loop_to_start = np.linalg.inv(source_loop) @ source_start
    target_loop_to_start = np.linalg.inv(target_loop) @ target_start
    return (
        target_loop_to_start
        @ _pose(start_transform)
        @ np.linalg.inv(source_loop_to_start)
    )


def is_rigid_transform(transform, atol: float = 1e-4) -> bool:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    rotation = transform[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
        and np.allclose(transform[3], [0, 0, 0, 1], atol=atol)
    )


def refine_point_to_plane_icp(
    source_cloud,
    target_cloud,
    initial_transform,
    distance_threshold_m: float = 0.005,
    max_iterations: int = 50,
):
    """Refine a source-to-target transform and never leak ICP exceptions."""
    start = time.perf_counter()
    result = {
        "success": False,
        "transformation": None,
        "initial_fitness": None,
        "initial_rmse": None,
        "fitness": 0.0,
        "rmse": float("inf"),
        "fitness_gain": None,
        "rmse_improvement_m": None,
        "runtime_s": 0.0,
        "failure_reason": None,
    }
    if not is_rigid_transform(initial_transform):
        result["failure_reason"] = "invalid_initial_transform"
        return result
    try:
        if len(source_cloud.points) < 3 or len(target_cloud.points) < 3:
            raise ValueError("point cloud has fewer than three points")
        if not source_cloud.has_normals():
            source_cloud.estimate_normals()
        if not target_cloud.has_normals():
            target_cloud.estimate_normals()
        initial = o3d.pipelines.registration.evaluate_registration(
            source_cloud,
            target_cloud,
            float(distance_threshold_m),
            np.asarray(initial_transform, dtype=np.float64),
        )
        result["initial_fitness"] = float(initial.fitness)
        result["initial_rmse"] = float(initial.inlier_rmse)
        alignment = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            float(distance_threshold_m),
            np.asarray(initial_transform, dtype=np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=int(max_iterations),
                relative_fitness=1e-9,
                relative_rmse=1e-9,
            ),
        )
        transformation = np.asarray(alignment.transformation, dtype=np.float64)
        if not is_rigid_transform(transformation, atol=1e-3):
            raise ValueError("ICP returned a non-rigid transformation")
        result.update({
            "success": True,
            "transformation": transformation,
            "fitness": float(alignment.fitness),
            "rmse": float(alignment.inlier_rmse),
            "fitness_gain": float(alignment.fitness - initial.fitness),
            "rmse_improvement_m": float(
                initial.inlier_rmse - alignment.inlier_rmse),
        })
    except Exception as error:  # one bad loop must not terminate full SLAM
        result["failure_reason"] = f"icp_failed:{type(error).__name__}:{error}"
    result["runtime_s"] = float(time.perf_counter() - start)
    return result


def loop_admission(fitness, rmse_m, min_fitness=0.30, max_rmse_m=0.005):
    if not np.isfinite(fitness) or not np.isfinite(rmse_m):
        return False, "non_finite_icp_metrics"
    if float(fitness) < float(min_fitness):
        return False, "fitness_below_threshold"
    if float(rmse_m) > float(max_rmse_m):
        return False, "rmse_above_threshold"
    return True, None
