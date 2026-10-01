"""Lightweight pose utilities shared by pose-graph backends."""

import numpy as np


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
