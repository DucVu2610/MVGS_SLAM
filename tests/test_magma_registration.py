import numpy as np
import open3d as o3d
import torch

from src.registration.coarse_registration import (
    estimate_rigid_transform,
    ransac_rigid_transform,
    transform_points,
)
from src.registration.gaussian_matching import mutual_nearest_neighbors
from src.registration.geometric_consistency import pairwise_rigidity_consistency
from src.registration.icp_refinement import refine_point_to_plane_icp
from src.registration.landmark_selection import select_spatially_balanced_landmarks
from src.registration.multiview_descriptor import (
    aggregate_multiview_descriptors,
    build_gaussian_landmark_descriptors,
    depth_consistency_weights,
    project_gaussians,
)


class _Dataset:
    intrinsics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def __getitem__(self, index):
        image = np.full((4, 4, 3), 32 + index, dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        return index, image, depth, np.eye(4)


class _Extractor:
    device = "cpu"

    def extract_patch_tokens(self, image, return_metadata=False):
        tokens = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        tokens = torch.nn.functional.normalize(tokens, dim=1)
        base = (tokens, (2, 2), (2.0, 2.0))
        if not return_metadata:
            return base
        return (*base, {
            "dino_preprocess_mode": "test",
            "patch_input_height": 4,
            "patch_input_width": 4,
            "patch_grid_height": 2,
            "patch_grid_width": 2,
        })


def _submap():
    points = torch.tensor([
        [0.25, 0.25, 1.0],
        [1.25, 0.25, 1.0],
        [0.25, 1.25, 1.0],
        [1.25, 1.25, 1.0],
    ])
    return {
        "submap_start_frame_id": 0,
        "submap_end_frame_id": 2,
        "keyframe_ids": torch.tensor([0, 1]),
        "submap_c2ws": torch.eye(4).repeat(2, 1, 1),
        "intrinsics": _Dataset.intrinsics,
        "gaussian_model_params": {
            "xyz": points,
            "opacity": torch.full((4, 1), 4.0),
            "scaling": torch.zeros((4, 3)),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        },
    }


def test_projection_and_depth_weights():
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    camera, u, v, visible = project_gaussians(
        points, _Dataset.intrinsics, np.eye(4), (4, 4))
    assert visible.tolist() == [True, False]
    assert np.allclose([u[0].item(), v[0].item(), camera[0, 2].item()], [0, 0, 1])
    weights = depth_consistency_weights(torch.tensor([0.0, 0.05]), 0.05)
    assert torch.allclose(
        weights, weights.new_tensor([1.0, np.exp(-1)]), atol=1e-6)


def test_multiview_descriptor_normalization_and_minimum_observations():
    landmarks = build_gaussian_landmark_descriptors(
        _submap(), _Dataset(), _Extractor(), "multiview", {
            "max_landmarks": 4,
            "min_observations": 2,
            "landmark_voxel_size_m": 0.01,
        })
    assert len(landmarks["xyz"]) == 4
    assert torch.all(landmarks["observation_count"] == 2)
    assert torch.allclose(
        torch.linalg.norm(landmarks["descriptors"], dim=1),
        torch.ones(4), atol=1e-6)
    descriptors, stability = aggregate_multiview_descriptors(
        torch.tensor([[2.0, 0.0], [1.0, 1.0]]), torch.tensor([2.0, 2.0]))
    assert torch.allclose(torch.linalg.norm(descriptors, dim=1), torch.ones(2))
    assert torch.all((stability >= 0) & (stability <= 1))


def test_spatial_selection_honors_k_and_voxels():
    points = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    chosen = select_spatially_balanced_landmarks(
        points, [1, 0.9, 0.8, 0.7], max_landmarks=3, voxel_size_m=0.1)
    assert len(chosen) == 3
    assert 0 in chosen and 2 in chosen and 3 in chosen


def test_mutual_nn_ambiguity_margin():
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    all_matches = mutual_nearest_neighbors(source, target, ambiguity_margin=0.0)
    strict = mutual_nearest_neighbors(source, target, ambiguity_margin=0.3)
    assert all_matches.num_mutual == 2
    assert len(strict.source_indices) < len(all_matches.source_indices)


def test_pairwise_rigidity_rejects_inconsistent_match():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    target = source.copy()
    target[-1] += [2, 0, 0]
    keep, support = pairwise_rigidity_consistency(
        source, target, distance_threshold_m=0.05, min_support=2)
    assert keep.tolist() == [True, True, True, False]
    assert support[-1] < 2


def test_kabsch_and_ransac_with_outliers():
    generator = np.random.default_rng(4)
    source = generator.normal(size=(40, 3))
    angle = 0.3
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = [0.2, -0.3, 0.1]
    target = transform_points(source, expected)
    assert np.allclose(estimate_rigid_transform(source, target), expected, atol=1e-8)
    target[-10:] = generator.normal(size=(10, 3)) * 4
    estimated, inliers = ransac_rigid_transform(
        source, target, distance_threshold_m=0.01, max_iterations=1000, seed=5)
    assert inliers.sum() >= 30
    assert np.allclose(estimated, expected, atol=1e-6)


def test_icp_failure_is_reported_not_raised():
    empty = o3d.geometry.PointCloud()
    result = refine_point_to_plane_icp(empty, empty, np.eye(4))
    assert not result["success"]
    assert result["failure_reason"].startswith("icp_failed")
