import unittest

import numpy as np

from src.entities.gtsam_pose_graph import PoseGraphAdapter_gtsam
from src.utils.pose_utils import apply_pose_correction


def translated_pose(x, y=0.0, z=0.0):
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


class GtsamNoLoopTest(unittest.TestCase):

    def test_initial_poses_produce_identity_correction_without_optimize(self):
        agents_submaps = {
            0: [
                {
                    "submap_start_frame_id": 0,
                    "submap_c2ws": np.array([
                        translated_pose(0.0), translated_pose(0.1),
                    ]),
                    "keyframe_ids": np.array([0, 1]),
                },
                {
                    "submap_start_frame_id": 2,
                    "submap_c2ws": np.array([
                        translated_pose(0.2), translated_pose(0.3),
                    ]),
                    "keyframe_ids": np.array([2, 3]),
                },
            ],
            1: [
                {
                    "submap_start_frame_id": 0,
                    "submap_c2ws": np.array([
                        translated_pose(1.0), translated_pose(1.1),
                    ]),
                    "keyframe_ids": np.array([0, 1]),
                },
            ],
        }

        graph = PoseGraphAdapter_gtsam(agents_submaps, loops=[])
        start_poses = graph.get_poses()

        for agent_id, submaps in agents_submaps.items():
            expected_starts = np.array([
                submap["submap_c2ws"][0] for submap in submaps
            ])
            np.testing.assert_allclose(
                start_poses[agent_id], expected_starts, atol=1e-9)

        corrected = apply_pose_correction(start_poses, agents_submaps)
        for agent_id, submaps in agents_submaps.items():
            expected_keyframes = np.concatenate([
                submap["submap_c2ws"] for submap in submaps
            ])
            np.testing.assert_allclose(
                corrected[agent_id], expected_keyframes, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
