#!/usr/bin/env python3
"""Export a colored TSDF mesh from completed MAGMA/MAGiC-SLAM outputs.

Only optimized SLAM poses are used for integration.  Dataset ground-truth
poses may be loaded by the repository dataset class, but are never read by
this exporter or used to select frames, correct poses, or fuse geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import open3d as o3d
import torch

from src.utils.dataset_utils import resolve_agent_input


def as_array(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def rgb_uint8(value):
    image = as_array(value)
    while image.ndim > 3 and 1 in image.shape:
        image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"unexpected RGB shape: {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating) and np.nanmax(image) <= 1.5:
        image = image * 255.0
    return np.ascontiguousarray(
        np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8))


def depth_float32(value):
    depth = as_array(value, np.float32)
    while depth.ndim > 2 and 1 in depth.shape:
        depth = np.squeeze(depth)
    if depth.ndim == 3:
        depth = depth[0] if depth.shape[0] <= 4 else depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"unexpected depth shape: {depth.shape}")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth <= 0] = 0
    return np.ascontiguousarray(depth)


def optimized_submap_correction(optimized_start_c2w, local_start_c2w):
    """Return ``T_opt,start @ inv(T_local,start)``."""
    optimized = as_array(optimized_start_c2w, np.float64)
    local = as_array(local_start_c2w, np.float64)
    if optimized.shape != (4, 4) or local.shape != (4, 4):
        raise ValueError("optimized and local poses must be 4x4")
    return optimized @ np.linalg.inv(local)


def clean_mesh(mesh, min_component_triangles: int):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) and int(min_component_triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        mesh.remove_triangles_by_mask(
            counts[labels] < int(min_component_triangles))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def fuse_rgbd_frames(
    frames,
    intrinsic,
    voxel_size_m=0.01,
    sdf_trunc_m=0.04,
    depth_trunc_m=5.0,
):
    """Fuse ``(rgb, depth_m, c2w)`` frames; useful for smoke tests too."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size_m),
        sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for rgb, depth, c2w in frames:
        color = rgb_uint8(rgb)
        depth = depth_float32(depth)
        pose = as_array(c2w, np.float64)
        if color.shape[:2] != depth.shape:
            raise ValueError("RGB and depth dimensions differ")
        if not np.isfinite(pose).all() or not np.isclose(
                np.linalg.det(pose[:3, :3]), 1.0, atol=0.05):
            raise ValueError("invalid rigid c2w pose")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc_m),
            convert_rgb_to_intensity=False,
        )
        # Open3D TSDF integration expects world-to-camera extrinsics.
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        integrated += 1
    return volume.extract_triangle_mesh(), integrated


def _camera_intrinsic(dataset, camera_config, width, height):
    nominal_width = float(getattr(dataset, "width", camera_config.get("W", width)))
    nominal_height = float(getattr(dataset, "height", camera_config.get("H", height)))
    fx = float(getattr(dataset, "fx", camera_config["fx"]))
    fy = float(getattr(dataset, "fy", camera_config["fy"]))
    cx = float(getattr(dataset, "cx", camera_config.get("cx", width / 2)))
    cy = float(getattr(dataset, "cy", camera_config.get("cy", height / 2)))
    scale_x, scale_y = width / nominal_width, height / nominal_height
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)


def _resolve_data_root(slam_output, config):
    configured = Path(config["data"]["input_path"]).expanduser()
    if configured.is_absolute():
        return configured
    candidates = [
        Path.cwd() / configured,
        REPO_ROOT / configured,
        slam_output / configured,
        slam_output.parent / configured,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def export_tsdf(args):
    slam_output = Path(args.slam_output).expanduser().resolve()
    config_path = slam_output / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    # Imported lazily so geometry helpers remain usable without CUDA extensions.
    from src.entities.datasets import get_dataset
    from src.utils.io_utils import load_config

    config = load_config(str(config_path))
    scene = str(config["data"].get("scene_name", "scene"))
    method = str(args.method or config.get(
        "registration", {}).get("descriptor_mode", "magic"))
    output = slam_output / "tsdf_mesh"
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = output / f"{scene}_tsdf_mesh.ply"
    preview_path = output / f"{scene}_tsdf_mesh_preview.ply"
    records_path = output / "tsdf_integrated_frames.csv"
    stats_path = output / "tsdf_stats.json"
    params_path = output / "tsdf_params.json"
    data_root = _resolve_data_root(slam_output, config)
    params = {
        "slam_output": str(slam_output),
        "dataset": config["dataset_name"],
        "scene": scene,
        "method": method,
        "voxel_size_m": float(args.voxel_size),
        "sdf_trunc_m": float(args.sdf_trunc),
        "depth_trunc_m": float(args.depth_trunc),
        "frame_stride": int(args.frame_stride),
        "min_component_triangles": int(args.min_component_triangles),
        "preview_triangles": int(args.preview_triangles),
        "pose_source": "agent_X/optimized_kf_c2w.ckpt",
        "uses_ground_truth_pose": False,
    }
    params_path.write_text(json.dumps(params, indent=2))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_size),
        sdf_trunc=float(args.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    records, seen = [], set()
    started = time.perf_counter()
    for agent_id_value in config["data"]["agent_ids"]:
        agent_id = int(agent_id_value)
        agent_output = slam_output / f"agent_{agent_id}"
        submap_paths = sorted(
            (agent_output / "submaps").glob("*.ckpt"), key=natural_key)
        optimized_path = agent_output / "optimized_kf_c2w.ckpt"
        keyframe_ids_path = agent_output / "kf_ids.ckpt"
        if (not submap_paths or not optimized_path.is_file()
                or not keyframe_ids_path.is_file()):
            raise FileNotFoundError(
                f"agent {agent_id} is missing submaps, keyframe IDs, or optimized poses")
        submaps = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in submap_paths
        ]
        optimized = as_array(torch.load(
            optimized_path, map_location="cpu", weights_only=False), np.float64)
        optimized_ids = as_array(torch.load(
            keyframe_ids_path, map_location="cpu", weights_only=False)
        ).reshape(-1).astype(np.int64, copy=False)
        saved_ids = np.concatenate([
            as_array(submap["keyframe_ids"]).reshape(-1).astype(
                np.int64, copy=False)
            for submap in submaps
        ])
        if len(optimized) != len(optimized_ids):
            raise RuntimeError(
                f"agent {agent_id}: {len(optimized)} optimized poses != "
                f"{len(optimized_ids)} optimized keyframe IDs")
        if not np.array_equal(optimized_ids, saved_ids):
            raise RuntimeError(
                f"agent {agent_id}: optimized pose keyframe IDs do not match "
                "the finalized submap keyframe IDs")
        if len(np.unique(optimized_ids)) != len(optimized_ids):
            raise RuntimeError(
                f"agent {agent_id}: optimized keyframe IDs contain duplicates")
        optimized_by_id = {
            int(frame_id): optimized[index]
            for index, frame_id in enumerate(optimized_ids)
        }
        # Agent-local config is authoritative when present because Agent saves
        # the already-resolved per-agent input path there. This also supports
        # TUM pseudo-agent layouts without assuming directory names.
        agent_config_path = agent_output / "config.yaml"
        agent_config = (
            load_config(str(agent_config_path))
            if agent_config_path.is_file() else config)
        dataset_config = {
            **agent_config["data"],
            **agent_config["cam"],
            **agent_config["submap"],
        }
        if not Path(dataset_config["input_path"]).exists():
            dataset_config["input_path"] = str(resolve_agent_input(
                data_root, agent_id, config["data"]["agent_ids"]))
        dataset = get_dataset(config["dataset_name"])(dataset_config)
        for submap_index, submap in enumerate(submaps):
            local_poses = as_array(submap["submap_c2ws"], np.float64)
            keyframes = as_array(submap["keyframe_ids"]).reshape(-1)
            if local_poses.ndim != 3 or local_poses.shape[1:] != (4, 4):
                raise ValueError(
                    f"agent {agent_id}, submap {submap_index}: invalid local poses")
            frame_start = int(submap["submap_start_frame_id"])
            if frame_start not in optimized_by_id:
                raise RuntimeError(
                    f"agent {agent_id}, submap {submap_index}: start frame "
                    f"{frame_start} is not an optimized keyframe")
            correction = optimized_submap_correction(
                optimized_by_id[frame_start], local_poses[0])
            offsets = list(range(0, len(local_poses), int(args.frame_stride)))
            if offsets and offsets[-1] != len(local_poses) - 1:
                offsets.append(len(local_poses) - 1)
            for local_index in offsets:
                frame_id = frame_start + local_index
                identity = (agent_id, frame_id)
                record = {
                    "agent": agent_id,
                    "submap": submap_index,
                    "frame": frame_id,
                    "status": "skipped",
                    "reason": None,
                    "valid_depth_pixels": 0,
                }
                if identity in seen:
                    record["reason"] = "duplicate_agent_frame"
                    records.append(record)
                    continue
                if not 0 <= frame_id < len(dataset):
                    record["reason"] = "frame_out_of_dataset_range"
                    records.append(record)
                    continue
                seen.add(identity)
                try:
                    _, rgb, depth, _unused_dataset_pose = dataset[frame_id]
                    color, depth_m = rgb_uint8(rgb), depth_float32(depth)
                    c2w = correction @ local_poses[local_index]
                    if (not np.isfinite(c2w).all()
                            or not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-5)
                            or not np.allclose(
                                c2w[:3, :3].T @ c2w[:3, :3],
                                np.eye(3), atol=1e-3)
                            or np.linalg.det(c2w[:3, :3]) <= 0):
                        raise ValueError("invalid optimized rigid c2w pose")
                    valid_depth = int(np.count_nonzero(
                        (depth_m > 0) & (depth_m <= float(args.depth_trunc))))
                    record["valid_depth_pixels"] = valid_depth
                    if valid_depth == 0:
                        record["reason"] = "no_valid_depth"
                        records.append(record)
                        continue
                    intrinsic = _camera_intrinsic(
                        dataset, agent_config["cam"],
                        color.shape[1], color.shape[0])
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(color),
                        o3d.geometry.Image(depth_m),
                        depth_scale=1.0,
                        depth_trunc=float(args.depth_trunc),
                        convert_rgb_to_intensity=False,
                    )
                    volume.integrate(rgbd, intrinsic, np.linalg.inv(c2w))
                    record["status"] = "integrated"
                    records.append(record)
                except Exception as error:
                    record["reason"] = f"{type(error).__name__}:{error}"
                    records.append(record)

    mesh = clean_mesh(
        volume.extract_triangle_mesh(), int(args.min_component_triangles))
    if not len(mesh.vertices) or not len(mesh.triangles):
        with records_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[
                    "agent", "submap", "frame", "status", "reason",
                    "valid_depth_pixels",
                ])
            writer.writeheader()
            writer.writerows(records)
        skipped = [record for record in records if record["status"] == "skipped"]
        stats_path.write_text(json.dumps({
            **params,
            "status": "failed",
            "failure_reason": "empty_tsdf_mesh",
            "attempted_frames": len(records),
            "integrated_frames": sum(
                record["status"] == "integrated" for record in records),
            "skipped_frames": len(skipped),
            "skip_reasons": dict(sorted(Counter(
                record["reason"] or "unspecified" for record in skipped
            ).items())),
        }, indent=2))
        raise RuntimeError("TSDF produced an empty mesh")
    if not o3d.io.write_triangle_mesh(
            str(mesh_path), mesh, write_ascii=False, compressed=True):
        raise RuntimeError(f"could not write {mesh_path}")
    preview = mesh
    if len(mesh.triangles) > int(args.preview_triangles):
        preview = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(args.preview_triangles))
        preview.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(
            str(preview_path), preview, write_ascii=False, compressed=True):
        raise RuntimeError(f"could not write {preview_path}")

    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[
                "agent", "submap", "frame", "status", "reason",
                "valid_depth_pixels",
            ])
        writer.writeheader()
        writer.writerows(records)
    bounds = mesh.get_axis_aligned_bounding_box()
    integrated_records = [
        record for record in records if record["status"] == "integrated"]
    skipped_records = [
        record for record in records if record["status"] == "skipped"]
    stats = {
        **params,
        "status": "completed",
        "input_root": str(data_root),
        "attempted_frames": len(records),
        "integrated_frames": len(integrated_records),
        "skipped_frames": len(skipped_records),
        "skip_reasons": dict(sorted(Counter(
            record["reason"] or "unspecified" for record in skipped_records
        ).items())),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "preview_vertices": len(preview.vertices),
        "preview_triangles": len(preview.triangles),
        "bounds_min_m": np.asarray(bounds.min_bound).tolist(),
        "bounds_max_m": np.asarray(bounds.max_bound).tolist(),
        "runtime_s": float(time.perf_counter() - started),
        "mesh_bytes": mesh_path.stat().st_size,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    if args.zip:
        archive_path = output / f"{method}_{scene}_tsdf_mesh.zip"
        with zipfile.ZipFile(
                archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(output.iterdir()):
                if path.is_file() and path != archive_path:
                    archive.write(path, Path("tsdf_mesh") / path.name)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slam-output", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--sdf-trunc", type=float, default=0.04)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-component-triangles", type=int, default=1000)
    parser.add_argument("--preview-triangles", type=int, default=300000)
    parser.add_argument("--method")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    if min(args.voxel_size, args.sdf_trunc, args.depth_trunc) <= 0:
        parser.error("voxel size and truncation values must be positive")
    if args.frame_stride < 1:
        parser.error("frame stride must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(export_tsdf(parse_args()), indent=2))
