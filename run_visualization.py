"""Visualize a completed MVGS-SLAM run with Rerun.

The default point-cloud mode reads Gaussian centers directly from checkpoints and
therefore does not require the CUDA rasterizer.  Mesh mode retains the original
Gaussian-to-mesh path and requires CUDA plus the project extensions.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import json
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SH_C0 = 0.28209479177387814
COLOR_PALETTE = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize saved MVGS-SLAM checkpoints in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        required=True,
        type=Path,
        help="Completed SLAM output directory containing config.yaml and agent_<id>/.",
    )
    parser.add_argument(
        "--artifacts-path",
        "--artifacts_path",
        type=Path,
        default=None,
        help="Directory for the RRD file, manifest, and optional mesh cache.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional dataset root override when paths saved in config are unavailable.",
    )
    parser.add_argument(
        "--geometry",
        choices=("points", "mesh"),
        default="points",
        help="Map representation. 'points' is portable; 'mesh' requires CUDA extensions.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Maximum Gaussian points logged for each map entity; <=0 disables sampling.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.003,
        help="Radius used by Rerun for Gaussian-center points.",
    )
    parser.add_argument(
        "--rrd",
        type=Path,
        default=None,
        help="Output Rerun recording. Relative paths are resolved below artifacts-path.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Also launch the Rerun viewer (disabled by default for Kaggle/headless runs).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip RGB/depth logging, so the original dataset is not required.",
    )
    return parser.parse_args(argv)


def _require_rerun():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Rerun is not installed. Install requirements.txt (rerun-sdk) before "
            "running visualization. The --help command works without it."
        ) from exc
    return rr, rrb


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _load_torch(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_pickle(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("rb") as handle:
        return pickle.load(handle)


def _agent_color(agent_id: int, order: int) -> np.ndarray:
    if order < len(COLOR_PALETTE):
        return np.asarray(COLOR_PALETTE[order], dtype=np.uint8)
    hue = (order * 0.618033988749895) % 1.0
    return np.rint(np.asarray(colorsys.hsv_to_rgb(hue, 0.75, 0.9)) * 255).astype(
        np.uint8
    )


def _validate_pose_array(name: str, poses: Any) -> np.ndarray:
    array = _as_numpy(poses, np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N,4,4], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    rotations = array[:, :3, :3]
    orthogonal = np.matmul(np.transpose(rotations, (0, 2, 1)), rotations)
    errors = np.max(np.abs(orthogonal - np.eye(3)), axis=(1, 2))
    determinants = np.linalg.det(rotations)
    bottom_errors = np.max(
        np.abs(array[:, 3, :] - np.asarray([0.0, 0.0, 0.0, 1.0])), axis=1
    )
    bad = np.flatnonzero(
        (errors > 1e-3)
        | (np.abs(determinants - 1.0) > 1e-3)
        | (bottom_errors > 1e-6)
    )
    if bad.size:
        index = int(bad[0])
        raise ValueError(
            f"Invalid rigid rotation in {name}[{index}]: "
            f"orthogonality_error={errors[index]:.3g}, det={determinants[index]:.6g}, "
            f"bottom_row_error={bottom_errors[index]:.3g}"
        )
    return array


def _resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    artifacts_path = (
        args.artifacts_path.expanduser().resolve()
        if args.artifacts_path is not None
        else checkpoint_path / "visualization"
    )
    artifacts_path.mkdir(parents=True, exist_ok=True)
    if args.rrd is None:
        rrd_path = artifacts_path / "mvgs_slam.rrd"
    elif args.rrd.is_absolute():
        rrd_path = args.rrd
    else:
        rrd_path = artifacts_path / args.rrd
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    return artifacts_path, rrd_path


def _find_submap_files(agent_path: Path) -> list[Path]:
    submap_dir = agent_path / "submaps"
    if not submap_dir.is_dir():
        raise FileNotFoundError(f"Missing submap directory: {submap_dir}")
    files = list(submap_dir.glob("*.ckpt"))
    if not files:
        raise FileNotFoundError(f"No submap checkpoints found in {submap_dir}")

    def sort_key(path: Path) -> tuple[int, str]:
        digits = "".join(character for character in path.stem if character.isdigit())
        return (int(digits) if digits else sys.maxsize, path.name)

    return sorted(files, key=sort_key)


def _submap_id(submap: dict[str, Any], fallback: int) -> int:
    return int(submap.get("submap_id", fallback))


def _submap_bounds(submap: dict[str, Any]) -> tuple[int, int]:
    start = int(submap["submap_start_frame_id"])
    end = int(submap["submap_end_frame_id"])
    if end < start:
        raise ValueError(f"Invalid submap range [{start}, {end}]")
    return start, end


def _validate_agent_state(agent_id: int, state: dict[str, Any]) -> None:
    keyframe_ids = state["keyframe_ids"]
    if len(set(keyframe_ids)) != len(keyframe_ids):
        raise ValueError(f"Agent {agent_id} contains duplicate keyframe IDs")
    if keyframe_ids != sorted(keyframe_ids):
        raise ValueError(f"Agent {agent_id} keyframe IDs are not ordered")
    if len(state["estimated_kf_c2w"]) != len(keyframe_ids):
        raise ValueError(
            f"Agent {agent_id}: estimated keyframe poses ({len(state['estimated_kf_c2w'])}) "
            f"do not match keyframe IDs ({len(keyframe_ids)})"
        )
    if len(state["optimized_kf_c2w"]) != len(keyframe_ids):
        raise ValueError(
            f"Agent {agent_id}: optimized keyframe poses ({len(state['optimized_kf_c2w'])}) "
            f"do not match keyframe IDs ({len(keyframe_ids)})"
        )

    submap_keyframes: list[int] = []
    submap_ids: set[int] = set()
    previous_end = -1
    for index, submap in enumerate(state["submaps"]):
        saved_agent_id = int(submap.get("agent_id", agent_id))
        if saved_agent_id != agent_id:
            raise ValueError(
                f"Agent {agent_id} directory contains submap owned by agent {saved_agent_id}"
            )
        current_submap_id = _submap_id(submap, index)
        if current_submap_id in submap_ids:
            raise ValueError(f"Agent {agent_id} contains duplicate submap ID {current_submap_id}")
        submap_ids.add(current_submap_id)
        start, end = _submap_bounds(submap)
        if start <= previous_end:
            raise ValueError(
                f"Agent {agent_id}: overlapping or unordered submap starting at frame {start}"
            )
        previous_end = end
        local_poses = _validate_pose_array(
            f"agent_{agent_id}.submap_{_submap_id(submap, index)}.submap_c2ws",
            submap["submap_c2ws"],
        )
        expected = end - start + 1
        if len(local_poses) != expected:
            raise ValueError(
                f"Agent {agent_id}, submap {_submap_id(submap, index)}: "
                f"{len(local_poses)} local poses for {expected} inclusive frames"
            )
        ids = [int(value) for value in _as_numpy(submap.get("keyframe_ids", []))]
        if any(frame_id < start or frame_id > end for frame_id in ids):
            raise ValueError(
                f"Agent {agent_id}, submap {_submap_id(submap, index)} has keyframe "
                f"outside [{start}, {end}]"
            )
        submap_keyframes.extend(ids)

    if submap_keyframes != keyframe_ids:
        raise ValueError(
            f"Agent {agent_id}: submap keyframe IDs do not match saved kf_ids.ckpt"
        )
    if keyframe_ids and max(keyframe_ids) >= len(state["estimated_c2w"]):
        raise ValueError(
            f"Agent {agent_id}: keyframe {max(keyframe_ids)} is outside estimated trajectory"
        )


def load_run(checkpoint_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[Any]]:
    from src.utils.io_utils import load_config

    config_path = checkpoint_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = load_config(str(config_path))
    agent_ids = [int(value) for value in config["data"]["agent_ids"]]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError(f"Duplicate IDs in data.agent_ids: {agent_ids}")

    states: dict[int, dict[str, Any]] = {}
    for agent_id in agent_ids:
        agent_path = checkpoint_path / f"agent_{agent_id}"
        if not agent_path.is_dir():
            raise FileNotFoundError(
                f"Missing output for agent {agent_id}: expected {agent_path}"
            )
        submaps = [_load_torch(path) for path in _find_submap_files(agent_path)]
        submaps.sort(key=lambda item: int(item["submap_start_frame_id"]))
        keyframe_ids = [int(value) for value in _as_numpy(_load_torch(agent_path / "kf_ids.ckpt"))]
        state = {
            "path": agent_path,
            "submaps": submaps,
            "keyframe_ids": keyframe_ids,
            "keyframe_index": {frame_id: index for index, frame_id in enumerate(keyframe_ids)},
            "estimated_c2w": _validate_pose_array(
                f"agent_{agent_id}.estimated_c2w", _load_torch(agent_path / "estimated_c2w.ckpt")
            ),
            "estimated_kf_c2w": _validate_pose_array(
                f"agent_{agent_id}.estimated_kf_c2w",
                _load_torch(agent_path / "estimated_kf_c2w.ckpt"),
            ),
            "optimized_kf_c2w": _validate_pose_array(
                f"agent_{agent_id}.optimized_kf_c2w",
                _load_torch(agent_path / "optimized_kf_c2w.ckpt"),
            ),
        }
        _validate_agent_state(agent_id, state)
        states[agent_id] = state

    loops = _load_pickle(checkpoint_path / "filtered_loops.pkl", [])
    return config, states, loops


def _path_candidates(path: Path, config_path: Path, checkpoint_path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    return [
        (Path.cwd() / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (config_path.parent / path).resolve(),
        (checkpoint_path / path).resolve(),
    ]


def _existing_path(
    path: str | Path | None, config_path: Path, checkpoint_path: Path
) -> Path | None:
    if path is None:
        return None
    for candidate in _path_candidates(Path(path).expanduser(), config_path, checkpoint_path):
        if candidate.exists():
            return candidate
    return None


def load_agent_datasets(
    checkpoint_path: Path,
    run_config: dict[str, Any],
    states: dict[int, dict[str, Any]],
    input_override: Path | None,
) -> dict[int, Any]:
    from src.entities.datasets import get_dataset
    from src.utils.io_utils import load_config
    from src.utils.dataset_utils import resolve_agent_input

    agent_ids = list(states)
    datasets: dict[int, Any] = {}
    override = input_override.expanduser().resolve() if input_override is not None else None

    for agent_id, state in states.items():
        agent_config_path = state["path"] / "config.yaml"
        config = (
            load_config(str(agent_config_path))
            if agent_config_path.is_file()
            else copy.deepcopy(run_config)
        )
        if override is not None:
            agent_input = resolve_agent_input(override, agent_id, agent_ids)
        else:
            configured_path = config["data"].get("input_path")
            agent_input = _existing_path(configured_path, agent_config_path, checkpoint_path)
            if agent_input is None:
                root_path = run_config["data"].get("input_path")
                root = _existing_path(root_path, checkpoint_path / "config.yaml", checkpoint_path)
                if root is not None:
                    try:
                        agent_input = resolve_agent_input(root, agent_id, agent_ids)
                    except FileNotFoundError:
                        agent_input = None
            if agent_input is None:
                raise FileNotFoundError(
                    f"Cannot resolve dataset directory for agent {agent_id}. "
                    "Pass --input-path <dataset-root> or use --no-images."
                )

        config["data"] = dict(config["data"])
        config["data"]["input_path"] = str(agent_input)
        dataset_config = {
            **config["data"],
            **config["cam"],
            **config.get("submap", {}),
        }
        datasets[agent_id] = get_dataset(config["dataset_name"])(dataset_config)
    return datasets


def _sample_indices(count: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def gaussian_params_to_points(
    params: dict[str, Any], max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    xyz = _as_numpy(params["xyz"], np.float32).reshape(-1, 3)
    features_dc = _as_numpy(params["features_dc"], np.float32)
    if features_dc.ndim == 3:
        features_dc = features_dc[:, 0, :]
    features_dc = features_dc.reshape(-1, 3)
    if len(xyz) != len(features_dc):
        raise ValueError(
            f"Gaussian xyz/features_dc size mismatch: {len(xyz)} vs {len(features_dc)}"
        )
    colors = np.clip(0.5 + SH_C0 * features_dc, 0.0, 1.0)
    colors = np.rint(colors * 255.0).astype(np.uint8)
    indices = _sample_indices(len(xyz), max_points)
    return xyz[indices], colors[indices]


def ply_to_points(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ModuleNotFoundError as exc:
        raise RuntimeError("plyfile is required to read the merged Gaussian PLY") from exc
    data = PlyData.read(str(path))["vertex"].data
    names = set(data.dtype.names or ())
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack((data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]))
        colors = np.rint(np.clip(0.5 + SH_C0 * sh, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    indices = _sample_indices(len(points), max_points)
    return points[indices], colors[indices]


def _cuda_gaussian_model(params: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("--geometry mesh requires an available CUDA device")
    try:
        from src.entities.arguments import OptimizationParams
        from src.entities.gaussian_model import GaussianModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Mesh mode requires the compiled Gaussian rasterizer and simple-knn extensions"
        ) from exc
    cuda_params = {
        key: value.detach().to("cuda") if torch.is_tensor(value) else value
        for key, value in params.items()
    }
    optimization = OptimizationParams(ArgumentParser(add_help=False))
    model = GaussianModel(0)
    model.restore_from_params(cuda_params, optimization)
    return model


def submap_to_mesh(submap: dict[str, Any], dataset: Any, mesh_path: Path):
    try:
        import trimesh
        from src.utils.vis_utils import gaussian_model_to_mesh
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Mesh mode dependencies are unavailable") from exc

    start, _ = _submap_bounds(submap)
    keyframes = [int(value) for value in _as_numpy(submap.get("keyframe_ids", []))]
    frame_offsets = [frame_id - start for frame_id in keyframes]
    local_poses = _as_numpy(submap["submap_c2ws"], np.float64)
    c2ws = [local_poses[offset] for offset in frame_offsets]
    model = _cuda_gaussian_model(submap["gaussian_model_params"])
    mesh_data = gaussian_model_to_mesh(
        model,
        c2ws,
        dataset.height,
        dataset.width,
        dataset.intrinsics,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_data.vertices),
        faces=np.asarray(mesh_data.triangles),
        vertex_colors=np.asarray(mesh_data.vertex_colors),
        process=False,
    )
    mesh.export(mesh_path)
    del model
    torch.cuda.empty_cache()
    return mesh


def merged_map_to_mesh(
    merged_map_path: Path,
    states: dict[int, dict[str, Any]],
    datasets: dict[int, Any],
    mesh_path: Path,
):
    try:
        import trimesh
        from src.entities.arguments import OptimizationParams
        from src.entities.gaussian_model import GaussianModel
        from src.utils.vis_utils import gaussian_model_to_mesh
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Mesh mode dependencies are unavailable") from exc
    if mesh_path.is_file():
        return trimesh.load(mesh_path, force="mesh", process=False)
    if not torch.cuda.is_available():
        raise RuntimeError("--geometry mesh requires an available CUDA device")

    reference_id = next(iter(datasets))
    reference = datasets[reference_id]
    reference_camera = _camera_parameters(reference)
    for agent_id, dataset in datasets.items():
        intrinsic, width, height = _camera_parameters(dataset)
        if (
            (width, height) != (reference_camera[1], reference_camera[2])
            or not np.allclose(intrinsic, reference_camera[0])
        ):
            raise ValueError(
                "Merged mesh generation requires identical cameras across agents; "
                f"agent {agent_id} differs from agent {reference_id}. Use "
                "--geometry points or the TSDF exporter instead."
            )

    model = GaussianModel(0)
    model.training_setup(OptimizationParams(ArgumentParser(add_help=False)))
    model.load_ply(str(merged_map_path))
    poses = [
        pose
        for agent_id in states
        for pose in states[agent_id]["optimized_kf_c2w"]
    ]
    mesh_data = gaussian_model_to_mesh(
        model,
        poses,
        reference.height,
        reference.width,
        reference.intrinsics,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_data.vertices),
        faces=np.asarray(mesh_data.triangles),
        vertex_colors=np.asarray(mesh_data.vertex_colors),
        process=False,
    )
    mesh.export(mesh_path)
    del model
    torch.cuda.empty_cache()
    return mesh


def _load_or_create_submap_geometry(
    submap: dict[str, Any],
    dataset: Any | None,
    geometry: str,
    artifacts_path: Path,
    agent_id: int,
    index: int,
    max_points: int,
) -> tuple[str, Any]:
    if geometry == "points":
        return "points", gaussian_params_to_points(submap["gaussian_model_params"], max_points)

    if dataset is None:
        raise RuntimeError("Mesh mode needs the dataset; remove --no-images and provide its path")
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError("trimesh is required for --geometry mesh") from exc
    submap_id = _submap_id(submap, index)
    mesh_path = artifacts_path / f"agent_{agent_id}_submap_{submap_id}.ply"
    mesh = (
        trimesh.load(mesh_path, force="mesh", process=False)
        if mesh_path.is_file()
        else submap_to_mesh(submap, dataset, mesh_path)
    )
    return "mesh", mesh


def _log_geometry(rr: Any, entity_path: str, geometry_data: tuple[str, Any], radius: float) -> None:
    kind, data = geometry_data
    if kind == "points":
        points, colors = data
        rr.log(entity_path, rr.Points3D(points, colors=colors, radii=radius))
        return
    colors = np.asarray(data.visual.vertex_colors)
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(data.vertices),
            triangle_indices=np.asarray(data.faces),
            vertex_colors=colors[:, :3] if colors.ndim == 2 else colors,
        ),
    )


def _log_transform(rr: Any, entity_path: str, pose: np.ndarray) -> None:
    rr.log(
        entity_path,
        rr.Transform3D(
            mat3x3=pose[:3, :3],
            translation=pose[:3, 3],
            relation=rr.TransformRelation.ParentFromChild,
        ),
    )


def _camera_parameters(dataset: Any) -> tuple[np.ndarray, int, int]:
    matrix = np.asarray(
        [[dataset.fx, 0.0, dataset.cx], [0.0, dataset.fy, dataset.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return matrix, int(dataset.width), int(dataset.height)


def _log_frame_images(rr: Any, agent_id: int, dataset: Any, frame_id: int) -> None:
    _, rgb, depth, _ = dataset[frame_id]
    rgb = _as_numpy(rgb)
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    depth = _as_numpy(depth, np.float32).squeeze()
    rr.log(f"map/cameras/agent_{agent_id}/image", rr.Image(rgb))
    rr.log(f"agents/agent_{agent_id}/depth", rr.DepthImage(depth, meter=1.0))


def _log_camera(
    rr: Any,
    agent_id: int,
    pose: np.ndarray,
    color: np.ndarray,
    dataset: Any | None,
) -> None:
    camera_path = f"map/cameras/agent_{agent_id}"
    _log_transform(rr, camera_path, pose)
    if dataset is not None:
        intrinsic, width, height = _camera_parameters(dataset)
        rr.log(
            f"{camera_path}/image",
            rr.Pinhole(
                image_from_camera=intrinsic,
                resolution=[width, height],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
    rr.log(
        f"{camera_path}/center",
        rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.025),
    )


def _optimized_submap_transform(state: dict[str, Any], submap: dict[str, Any]) -> np.ndarray:
    start, _ = _submap_bounds(submap)
    try:
        index = state["keyframe_index"][start]
    except KeyError as exc:
        raise ValueError(f"Submap start frame {start} is not a saved keyframe") from exc
    optimized = state["optimized_kf_c2w"][index]
    local_origin = _as_numpy(submap["submap_c2ws"], np.float64)[0]
    return optimized @ np.linalg.inv(local_origin)


def _loop_values(loop: Any) -> tuple[int, int, int, int]:
    if hasattr(loop, "source_agent_id"):
        return (
            int(loop.source_agent_id),
            int(loop.source_frame_id),
            int(loop.target_agent_id),
            int(loop.target_frame_id),
        )
    if hasattr(loop, "agent0"):
        return int(loop.agent0), int(loop.frame0), int(loop.agent1), int(loop.frame1)
    if isinstance(loop, dict):
        if "source_agent_id" in loop:
            return (
                int(loop["source_agent_id"]),
                int(loop["source_frame_id"]),
                int(loop["target_agent_id"]),
                int(loop["target_frame_id"]),
            )
        return (
            int(loop["agent0"]),
            int(loop["frame0"]),
            int(loop["agent1"]),
            int(loop["frame1"]),
        )
    if isinstance(loop, (tuple, list)) and len(loop) >= 4:
        return tuple(int(value) for value in loop[:4])
    raise ValueError(f"Unsupported loop record: {loop!r}")


def _build_blueprint(rrb: Any, agent_ids: list[int], include_images: bool) -> Any:
    contents: list[Any] = [rrb.Spatial3DView(name="Map", origin="map")]
    if include_images:
        for agent_id in agent_ids:
            contents.append(
                rrb.Horizontal(
                    name=f"Agent {agent_id}",
                    contents=[
                        rrb.Spatial2DView(origin=f"map/cameras/agent_{agent_id}/image"),
                        rrb.Spatial2DView(origin=f"agents/agent_{agent_id}/depth"),
                    ],
                )
            )
    return rrb.Vertical(contents=contents)


def visualize(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    if args.point_radius <= 0:
        raise ValueError("--point-radius must be positive")
    rr, rrb = _require_rerun()
    artifacts_path, rrd_path = _resolve_artifact_paths(args)
    config, states, loops = load_run(checkpoint_path)
    datasets = (
        {}
        if args.no_images
        else load_agent_datasets(checkpoint_path, config, states, args.input_path)
    )
    if args.geometry == "mesh" and not datasets:
        raise RuntimeError("--geometry mesh cannot be combined with --no-images")

    agent_ids = list(states)
    colors = {
        agent_id: _agent_color(agent_id, order)
        for order, agent_id in enumerate(agent_ids)
    }
    submap_geometry: dict[tuple[int, int], tuple[str, Any]] = {}
    for agent_id, state in states.items():
        for index, submap in enumerate(state["submaps"]):
            submap_geometry[(agent_id, index)] = _load_or_create_submap_geometry(
                submap,
                datasets.get(agent_id),
                args.geometry,
                artifacts_path,
                agent_id,
                index,
                args.max_points,
            )

    rr.init("MVGS-SLAM", spawn=args.spawn)
    rr.save(str(rrd_path))
    rr.send_blueprint(_build_blueprint(rrb, agent_ids, not args.no_images))
    rr.log("map", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    all_keyframes = sorted(
        {frame_id for state in states.values() for frame_id in state["keyframe_ids"]}
    )
    trajectories: dict[int, list[np.ndarray]] = {agent_id: [] for agent_id in agent_ids}
    for frame_id in all_keyframes:
        rr.set_time_sequence("frame", frame_id)
        for agent_id, state in states.items():
            index = state["keyframe_index"].get(frame_id)
            if index is None:
                continue
            pose = state["estimated_kf_c2w"][index]
            trajectories[agent_id].append(pose[:3, 3])
            _log_camera(rr, agent_id, pose, colors[agent_id], datasets.get(agent_id))
            rr.log(
                f"map/trajectories/agent_{agent_id}",
                rr.LineStrips3D([np.asarray(trajectories[agent_id])], colors=[colors[agent_id]]),
            )
            if agent_id in datasets:
                if frame_id >= len(datasets[agent_id]):
                    raise IndexError(
                        f"Agent {agent_id}: keyframe {frame_id} exceeds dataset length "
                        f"{len(datasets[agent_id])}"
                    )
                _log_frame_images(rr, agent_id, datasets[agent_id], frame_id)

            for submap_index, submap in enumerate(state["submaps"]):
                start, _ = _submap_bounds(submap)
                if start == frame_id:
                    submap_id = _submap_id(submap, submap_index)
                    _log_geometry(
                        rr,
                        f"map/submaps/agent_{agent_id}/submap_{submap_id}",
                        submap_geometry[(agent_id, submap_index)],
                        args.point_radius,
                    )

    loop_time = (all_keyframes[-1] if all_keyframes else 0) + 1
    rr.set_time_sequence("frame", loop_time)
    valid_loop_count = 0
    for loop_index, loop in enumerate(loops):
        agent0, frame0, agent1, frame1 = _loop_values(loop)
        if agent0 not in states or agent1 not in states:
            raise ValueError(
                f"Loop {loop_index} references unknown agent ({agent0}, {agent1})"
            )
        poses0 = states[agent0]["estimated_c2w"]
        poses1 = states[agent1]["estimated_c2w"]
        if not (0 <= frame0 < len(poses0) and 0 <= frame1 < len(poses1)):
            raise IndexError(
                f"Loop {loop_index} frame is outside trajectory: "
                f"agent {agent0}/{frame0}, agent {agent1}/{frame1}"
            )
        rr.log(
            f"map/loop_closures/loop_{loop_index}",
            rr.LineStrips3D(
                [[poses0[frame0, :3, 3], poses1[frame1, :3, 3]]],
                colors=[[255, 0, 255]],
                radii=0.01,
            ),
        )
        valid_loop_count += 1

    correction_time = loop_time + 1
    rr.set_time_sequence("frame", correction_time)
    rr.log("map/loop_closures", rr.Clear(recursive=True))
    for agent_id, state in states.items():
        for index, submap in enumerate(state["submaps"]):
            submap_id = _submap_id(submap, index)
            _log_transform(
                rr,
                f"map/submaps/agent_{agent_id}/submap_{submap_id}",
                _optimized_submap_transform(state, submap),
            )
        optimized = state["optimized_kf_c2w"]
        if len(optimized):
            _log_camera(
                rr,
                agent_id,
                optimized[-1],
                colors[agent_id],
                datasets.get(agent_id),
            )
            rr.log(
                f"map/optimized_trajectories/agent_{agent_id}",
                rr.LineStrips3D([optimized[:, :3, 3]], colors=[colors[agent_id]], radii=0.012),
            )

    merged_path = next(
        (
            path
            for path in (
                checkpoint_path / "merged_refined.ply",
                checkpoint_path / "merged_coarse.ply",
            )
            if path.is_file()
        ),
        None,
    )
    if merged_path is not None:
        rr.set_time_sequence("frame", correction_time + 1)
        for entity_path in (
            "map/submaps",
            "map/cameras",
            "map/trajectories",
            "map/optimized_trajectories",
        ):
            rr.log(entity_path, rr.Clear(recursive=True))
        if args.geometry == "points":
            final_geometry = ("points", ply_to_points(merged_path, args.max_points))
        else:
            final_geometry = (
                "mesh",
                merged_map_to_mesh(
                    merged_path,
                    states,
                    datasets,
                    artifacts_path / "merged_map_mesh.ply",
                ),
            )
        _log_geometry(rr, "map/final_merged", final_geometry, args.point_radius)

    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush(blocking=True)
    rr.disconnect()

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "rrd_path": str(rrd_path),
        "geometry": args.geometry,
        "agent_ids": agent_ids,
        "keyframes_per_agent": {
            str(agent_id): len(state["keyframe_ids"]) for agent_id, state in states.items()
        },
        "submaps_per_agent": {
            str(agent_id): len(state["submaps"]) for agent_id, state in states.items()
        },
        "filtered_loop_count": valid_loop_count,
        "images_logged": not args.no_images,
        "used_ground_truth_poses": False,
        "final_map_path": str(merged_path) if merged_path is not None else None,
    }
    manifest_path = artifacts_path / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = visualize(args)
    except (FileNotFoundError, IndexError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Visualization failed: {exc}") from exc
    print(f"Rerun recording: {result['rrd_path']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
