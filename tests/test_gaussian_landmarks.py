import unittest

import numpy as np
import torch

from src.utils.gaussian_landmarks import build_single_view_gaussian_landmarks


class FakeGaussianModel:

    def __init__(self, xyz, opacity):
        self._xyz = torch.tensor(xyz, dtype=torch.float32)
        self._opacity = torch.tensor(opacity, dtype=torch.float32)

    def get_xyz(self):
        return self._xyz

    def get_opacity(self):
        return self._opacity


class GaussianLandmarkTest(unittest.TestCase):

    def test_coordinate_filters_one_per_patch_and_determinism(self):
        camera_points = np.array([
            [-0.5, -0.5, 1.00],  # valid, patch 0
            [-0.5, -0.5, 1.02],  # same patch, worse depth residual
            [0.5, -0.5, 1.00],   # valid, patch 1
            [0.5, 0.5, 1.00],    # rejected by opacity
            [-0.5, 0.5, 2.00],   # rejected by depth consistency
            [0.0, 0.0, -1.00],   # rejected because it is behind the camera
        ], dtype=np.float32)
        reference_c2w = np.eye(4, dtype=np.float32)
        reference_c2w[0, 3] = 1.0
        world_points = camera_points.copy()
        world_points[:, 0] += reference_c2w[0, 3]
        model = FakeGaussianModel(
            world_points,
            [0.8, 0.95, 0.9, 0.01, 0.9, 0.9],
        )
        intrinsics = np.array([
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        depth = np.ones((4, 4), dtype=np.float32)
        patch_tokens = torch.eye(4, dtype=torch.float32)

        first = build_single_view_gaussian_landmarks(
            model, depth, intrinsics, reference_c2w,
            patch_tokens, (2, 2), min_opacity=0.05,
            depth_tolerance_m=0.05, max_landmarks=4)
        second = build_single_view_gaussian_landmarks(
            model, depth, intrinsics, reference_c2w,
            patch_tokens, (2, 2), min_opacity=0.05,
            depth_tolerance_m=0.05, max_landmarks=4)

        descriptors, points, patch_ids, diagnostics = first
        self.assertEqual(diagnostics["num_gaussian_landmarks"], 2)
        self.assertEqual(len(np.unique(patch_ids)), len(patch_ids))
        selected = {int(patch_id): point for patch_id, point in zip(patch_ids, points)}
        np.testing.assert_allclose(selected[0], camera_points[0], atol=1e-6)
        np.testing.assert_allclose(selected[1], camera_points[2], atol=1e-6)
        np.testing.assert_allclose(
            descriptors.detach().cpu().numpy(), patch_tokens[patch_ids].numpy())

        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_allclose(first[1], second[1], atol=0.0)
        torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
