"""Lightweight pose utilities shared by pose-graph backends."""

import numpy as np


def optimized_pose_for_keyframe(keyframe_ids, optimized_poses, frame_id, label=""):
    """Return the optimized pose for one saved keyframe with strict ID checks."""
    ids = np.asarray(keyframe_ids).reshape(-1).astype(np.int64, copy=False)
    poses = np.asarray(optimized_poses, dtype=np.float64)
    prefix = f"{label}: " if label else ""
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{prefix}optimized poses must have shape [N, 4, 4]")
    if len(ids) != len(poses):
        raise ValueError(
            f"{prefix}{len(ids)} keyframe IDs != {len(poses)} optimized poses")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{prefix}optimized keyframe IDs contain duplicates")
    matches = np.flatnonzero(ids == int(frame_id))
    if len(matches) != 1:
        raise KeyError(
            f"{prefix}keyframe {int(frame_id)} has {len(matches)} optimized poses")
    pose = poses[int(matches[0])]
    if not np.isfinite(pose).all():
        raise ValueError(f"{prefix}optimized pose for keyframe {frame_id} is non-finite")
    return pose


def apply_pose_correction(
        submap_optimized_poses: dict, agents_submaps: dict) -> dict:
    """Apply each submap's start-pose correction to its keyframe poses."""
    agents_corrected_kf_poses = {}
    for agent_id in sorted(agents_submaps.keys()):
        opt_kf_poses = []
        for i, submap in enumerate(agents_submaps[agent_id]):
            submap_kf_ids = (
                submap["keyframe_ids"] - submap["submap_start_frame_id"]
            )
            delta = (
                submap_optimized_poses[agent_id][i]
                @ np.linalg.inv(submap["submap_c2ws"][0])
            )
            for pose in submap["submap_c2ws"][submap_kf_ids]:
                opt_kf_poses.append(delta @ pose)
        agents_corrected_kf_poses[agent_id] = np.array(opt_kf_poses)
    return agents_corrected_kf_poses
