from pathlib import Path

import numpy as np
import open3d as o3d

from run_slam import validate_registration_config
from src.mapping.gaussian_compaction import select_compaction_indices
from src.utils.io_utils import load_config
from tools.export_tsdf_mesh import (
    fuse_rgbd_frames,
    optimized_submap_correction,
)
from src.registration.icp_refinement import (
    loop_to_submap_start_transform,
    submap_start_to_loop_transform,
)


def test_optimized_submap_pose_correction():
    local_start = np.eye(4)
    local_start[:3, 3] = [1, 0, 0]
    optimized_start = np.eye(4)
    optimized_start[:3, 3] = [3, 1, 0]
    local_frame = local_start.copy()
    local_frame[:3, 3] += [0, 2, 0]
    correction = optimized_submap_correction(optimized_start, local_start)
    corrected = correction @ local_frame
    assert np.allclose(corrected[:3, 3], [3, 3, 0])


def test_loop_and_submap_start_transform_roundtrip():
    source_poses = np.repeat(np.eye(4)[None], 3, axis=0)
    target_poses = np.repeat(np.eye(4)[None], 3, axis=0)
    source_poses[:, 0, 3] = [1.0, 1.2, 1.4]
    target_poses[:, 1, 3] = [2.0, 2.3, 2.6]
    source = {"submap_start_frame_id": 10, "submap_c2ws": source_poses}
    target = {"submap_start_frame_id": 20, "submap_c2ws": target_poses}
    loop_transform = np.eye(4)
    loop_transform[:3, 3] = [0.3, -0.2, 0.1]
    at_start = loop_to_submap_start_transform(
        loop_transform, source, target, 12, 22)
    recovered = submap_start_to_loop_transform(
        at_start, source, target, 12, 22)
    assert np.allclose(recovered, loop_transform)


def test_compaction_opacity_voxel_and_cap():
    xyz = np.array([
        [0.000, 0, 0], [0.001, 0, 0], [0.02, 0, 0], [0.04, 0, 0],
    ])
    opacity = np.array([0.8, 0.9, 0.01, 0.7])
    kept = select_compaction_indices(
        xyz, opacity, min_opacity=0.05, voxel_size_m=0.005, max_gaussians=2)
    assert kept.tolist() == [1, 3]


def test_baseline_config_keeps_magma_features_disabled():
    config = load_config("configs/magma/baseline_magic.yaml")
    config = validate_registration_config(config)
    assert config["registration"]["descriptor_mode"] == "baseline"
    assert config["global_compaction"]["enabled"] is False
    assert config["submap"]["anchor_data"] == "pcd"


def test_tiny_colored_tsdf_fusion():
    height, width = 32, 32
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    depth = np.ones((height, width), dtype=np.float32)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, 30.0, 30.0, width / 2, height / 2)
    mesh, integrated = fuse_rgbd_frames(
        [(rgb, depth, np.eye(4))], intrinsic,
        voxel_size_m=0.04, sdf_trunc_m=0.12, depth_trunc_m=2.0)
    assert integrated == 1
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
