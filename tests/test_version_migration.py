from pathlib import Path

import numpy as np
import pytest
import torch

from run_slam import validate_registration_config
from src.registration.coarse_registration import (
    is_valid_rigid_transform,
    register_landmark_sets,
    transform_points,
)
from src.registration.landmark_selection import (
    select_spatially_covered_correspondences,
)
from src.registration.multiview_descriptor import build_gaussian_landmark_descriptors
from src.utils.gpu_utils import build_agent_gpu_map
from src.utils.io_utils import load_config
from src.utils.pose_utils import optimized_pose_for_keyframe
from src.utils.submap_utils import find_submap
from tools.benchmark.protocol import (
    find_submap_index,
    landmark_view,
    resolve_agent_input,
    summarize_trials,
    validate_finalized_pose_ids,
)


class _Dataset:
    intrinsics = np.eye(3)

    def __getitem__(self, index):
        return (
            index,
            np.full((4, 4, 3), 40 + index, dtype=np.uint8),
            np.ones((4, 4), dtype=np.float32),
            np.eye(4),
        )


class _Extractor:
    device = "cpu"

    def extract_patch_tokens(self, image, return_metadata=False):
        values = torch.nn.functional.normalize(torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        ]), dim=1)
        result = (values, (2, 2), (2.0, 2.0))
        return (*result, {}) if return_metadata else result


def _submap():
    return {
        "submap_start_frame_id": 4,
        "submap_end_frame_id": 5,
        "keyframe_ids": torch.tensor([4, 5]),
        "submap_c2ws": torch.eye(4).repeat(2, 1, 1),
        "intrinsics": np.eye(3),
        "gaussian_model_params": {
            "xyz": torch.tensor([
                [0.25, 0.25, 1.0], [1.25, 0.25, 1.0],
                [0.25, 1.25, 1.0], [1.25, 1.25, 1.0],
            ]),
            "opacity": torch.full((4, 1), 4.0),
            "scaling": torch.zeros((4, 3)),
            "rotation": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        },
    }


def test_single_and_multiview_share_landmark_source():
    options = {
        "max_landmarks": 4,
        "min_observations": 2,
        "landmark_voxel_size_m": 0.01,
    }
    single = build_gaussian_landmark_descriptors(
        _submap(), _Dataset(), _Extractor(), "single", options)
    multi = build_gaussian_landmark_descriptors(
        _submap(), _Dataset(), _Extractor(), "multiview", options)
    assert torch.equal(single["gaussian_indices"], multi["gaussian_indices"])
    assert torch.all(single["observation_count"] >= 2)
    assert {
        "descriptor_single_best_view", "descriptor_multi_view"
    }.issubset(single)
    sliced_single = landmark_view(single, "single", 4)
    sliced_multi = landmark_view(single, "multiview", 4)
    assert torch.equal(
        sliced_single["gaussian_indices"], sliced_multi["gaussian_indices"])


def test_coverage_fix_runs_below_pair_cap():
    source = np.array([[0.00, 0, 0], [0.01, 0, 0], [1, 0, 0], [2, 0, 0]])
    target = source.copy()
    selected, info = select_spatially_covered_correspondences(
        source, target, [1.0, 0.9, 0.8, 0.7],
        max_correspondences=96, voxel_size_m=0.1,
        max_per_source_voxel=1, max_per_target_voxel=1, min_keep=3)
    assert info["input_pairs"] == 4
    assert info["strict_coverage_pairs"] == 3
    assert selected.tolist() == [0, 2, 3]


def test_hard_recovery_and_rigid_validation():
    generator = np.random.default_rng(7)
    source = generator.normal(size=(12, 3))
    expected = np.eye(4)
    expected[:3, 3] = [0.2, -0.1, 0.05]
    target = transform_points(source, expected)
    descriptors = torch.eye(12)

    def landmarks(points):
        return {
            "xyz": torch.tensor(points, dtype=torch.float32),
            "descriptors": descriptors,
            "reliability": torch.ones(12),
            "scales": torch.ones((12, 3)),
            "axes": torch.eye(3).repeat(12, 1, 1),
        }

    transform, diagnostics = register_landmark_sets(
        landmarks(source), landmarks(target), np.eye(4), np.eye(4), {
            "geometric_consistency": "none",
            "min_correspondences": 3,
            "min_ransac_inliers": 3,
            "ransac_max_iterations": 200,
            "hard_case_min_inlier_ratio": 1.1,
            "hard_case_top_k": 3,
            "coverage_voxel_size_m": 0.01,
        })
    assert diagnostics["fallback_triggered"]
    assert diagnostics["fallback_selected"]
    assert diagnostics["selected_stage"] == "hard_recovery"
    assert is_valid_rigid_transform(transform)
    assert np.allclose(transform, expected, atol=1e-6)
    invalid = expected.copy()
    invalid[0, 0] = 2.0
    assert not is_valid_rigid_transform(invalid)


def test_failure_never_returns_a_transform():
    landmarks = {
        "xyz": torch.zeros((2, 3)),
        "descriptors": torch.eye(2),
        "reliability": torch.ones(2),
        "scales": torch.ones((2, 3)),
        "axes": torch.eye(3).repeat(2, 1, 1),
    }
    transform, diagnostics = register_landmark_sets(
        landmarks, landmarks, np.eye(4), np.eye(4), {
            "geometric_consistency": "none", "min_correspondences": 3,
        })
    assert transform is None
    assert diagnostics["failure_reason"] == "insufficient_descriptor_matches"


def test_inclusive_submap_boundaries_and_unique_mapping():
    submaps = [
        {"submap_start_frame_id": 0, "submap_end_frame_id": 4},
        {"submap_start_frame_id": 5, "submap_end_frame_id": 9},
    ]
    assert find_submap(4, submaps) is submaps[0]
    assert find_submap(9, submaps) is submaps[1]
    assert find_submap_index(submaps, 5) == 1


def test_optimized_pose_ids_are_not_positionally_guessed():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 3.0
    assert optimized_pose_for_keyframe([4, 9], poses, 9)[0, 3] == 3.0
    submaps = [{"keyframe_ids": np.array([4, 9])}]
    validate_finalized_pose_ids(submaps, [4, 9], poses, label="agent 0")
    with pytest.raises(ValueError, match="do not match"):
        validate_finalized_pose_ids(submaps, [9, 4], poses, label="agent 0")


def test_agent_folder_resolution_and_gpu_shortage(tmp_path: Path):
    (tmp_path / "camera_a").mkdir()
    (tmp_path / "camera_b").mkdir()
    assert resolve_agent_input(tmp_path, 20, [10, 20]).name == "camera_b"
    with pytest.raises(RuntimeError, match="Insufficient CUDA devices"):
        build_agent_gpu_map([0, 1, 2], 2)


def test_benchmark_summary_denominators_and_conditional_errors():
    summary = summarize_trials([
        {"valid_estimate": True, "passed": True,
         "rotation_error_deg": 1.0, "translation_error_cm": 2.0},
        {"valid_estimate": False, "passed": False,
         "failure_reason": "ransac_failed"},
    ])
    assert summary["attempts"] == 2
    assert summary["pass_rate_percent"] == 50.0
    assert summary["conditional_rotation_error_deg_mean"] == 1.0
    assert summary["failure_reasons"] == {"ransac_failed": 1}


@pytest.mark.parametrize(
    "path,mode",
    [
        ("configs/magma/baseline_magic.yaml", "baseline"),
        ("configs/magma/magma_single.yaml", "single"),
        ("configs/magma/magma_multiview.yaml", "multiview"),
    ],
)
def test_all_three_descriptor_configs(path, mode):
    config = validate_registration_config(load_config(path))
    assert config["registration"]["descriptor_mode"] == mode
