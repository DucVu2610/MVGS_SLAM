#!/usr/bin/env python3
"""Benchmark Single/Multi Gaussian registration on one completed SLAM run.

The tool consumes fixed detected loops and finalized submap checkpoints.  It
does not instantiate LoopDetector, does not run DINO retrieval, and never
writes into full-pipeline wall-time files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.registration.coarse_registration import (
    is_valid_rigid_transform,
    register_landmark_sets,
)
from src.registration.icp_refinement import (
    loop_to_submap_start_transform,
    refine_point_to_plane_icp,
    submap_frame_c2w,
    submap_start_to_loop_transform,
)
from src.registration.multiview_descriptor import build_gaussian_landmark_descriptors
from src.utils.benchmark_utils import load_candidate_identities
from src.utils.io_utils import load_config
from tools.benchmark.protocol import (
    find_submap_index,
    landmark_view,
    resolve_agent_input,
    summarize_trials,
    validate_finalized_pose_ids,
)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_data_root(run_dir, config):
    configured = Path(config["data"]["input_path"]).expanduser()
    candidates = [configured] if configured.is_absolute() else [
        Path.cwd() / configured,
        run_dir / configured,
        run_dir.parent / configured,
        REPO_ROOT / "data" / configured,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve data.input_path={configured} for {run_dir}")


def load_completed_run(run_dir, allow_legacy_loops=False):
    config = load_config(str(run_dir / "config.yaml"))
    candidates, candidate_path = load_candidate_identities(run_dir)
    if candidate_path.name != "detected_loops.pkl" and not allow_legacy_loops:
        raise FileNotFoundError(
            "detected_loops.pkl is required for a complete fixed-retrieval "
            "benchmark; pass --allow-legacy-loops only for an explicitly "
            "labelled legacy run")
    candidate_keys = [(
        int(candidate.source_agent_id), int(candidate.source_frame_id),
        int(candidate.target_agent_id), int(candidate.target_frame_id),
    ) for candidate in candidates]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("fixed retrieved loops contain duplicate identities")
    submaps, submap_paths = {}, {}
    for agent_value in config["data"]["agent_ids"]:
        agent = int(agent_value)
        agent_dir = run_dir / f"agent_{agent}"
        paths = sorted(
            (agent_dir / "submaps").glob("*.ckpt"), key=_natural_key)
        if not paths:
            raise FileNotFoundError(f"no finalized submaps in {agent_dir}")
        keyframe_path = agent_dir / "kf_ids.ckpt"
        optimized_path = agent_dir / "optimized_kf_c2w.ckpt"
        if not keyframe_path.is_file() or not optimized_path.is_file():
            raise FileNotFoundError(
                f"{agent_dir} lacks kf_ids.ckpt or optimized_kf_c2w.ckpt")
        submaps[agent] = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in paths
        ]
        submap_paths[agent] = paths
        keyframe_ids = torch.load(
            keyframe_path, map_location="cpu", weights_only=False)
        optimized = torch.load(
            optimized_path, map_location="cpu", weights_only=False)
        validate_finalized_pose_ids(
            submaps[agent], keyframe_ids, optimized, label=f"agent {agent}")

    fixed = []
    required = set()
    for index, candidate in enumerate(candidates):
        source_agent = int(candidate.source_agent_id)
        target_agent = int(candidate.target_agent_id)
        source_submap = find_submap_index(
            submaps[source_agent], candidate.source_frame_id)
        target_submap = find_submap_index(
            submaps[target_agent], candidate.target_frame_id)
        required.add((source_agent, source_submap))
        required.add((target_agent, target_submap))
        fixed.append({
            "candidate_id": index,
            "source_agent_id": source_agent,
            "source_frame_id": int(candidate.source_frame_id),
            "target_agent_id": target_agent,
            "target_frame_id": int(candidate.target_frame_id),
            "source_submap_index": source_submap,
            "target_submap_index": target_submap,
        })
    return config, submaps, submap_paths, fixed, required, candidate_path


def load_agent_datasets(run_dir, config):
    from src.entities.datasets import get_dataset

    data_root = _resolve_data_root(run_dir, config)
    datasets = {}
    dataset_type = get_dataset(config["dataset_name"])
    for agent_value in config["data"]["agent_ids"]:
        agent = int(agent_value)
        agent_config_path = run_dir / f"agent_{agent}" / "config.yaml"
        agent_config = (
            load_config(str(agent_config_path))
            if agent_config_path.is_file() else config)
        dataset_config = {
            **agent_config["data"],
            **agent_config["cam"],
            **agent_config["submap"],
        }
        if not Path(dataset_config["input_path"]).expanduser().exists():
            dataset_config["input_path"] = str(resolve_agent_input(
                data_root, agent, config["data"]["agent_ids"]))
        datasets[agent] = dataset_type(dataset_config)
    return datasets, data_root


def build_shared_caches(required, submaps, datasets, extractor, options):
    caches, rows = {}, []
    for agent, submap_index in sorted(required):
        _sync()
        started = time.perf_counter()
        cache = build_gaussian_landmark_descriptors(
            submaps[agent][submap_index],
            datasets[agent],
            extractor,
            descriptor_mode="multiview",
            options=options,
        )
        _sync()
        elapsed = float(time.perf_counter() - started)
        required_keys = {
            "descriptor_single_best_view", "descriptor_multi_view",
            "gaussian_indices", "xyz",
        }
        if not required_keys.issubset(cache):
            raise RuntimeError(
                f"shared cache lacks {sorted(required_keys - set(cache))}")
        caches[(agent, submap_index)] = cache
        rows.append({
            "agent_id": agent,
            "submap_index": submap_index,
            "landmarks": len(cache["xyz"]),
            # Exactly one measured field; historical/stored time is separate.
            "cache_build_time_s_measured": elapsed,
            "cache_build_time_s_historical": None,
        })
    return caches, rows


def _dense_cloud(submap):
    from src.utils import utils

    color, depth = utils.get_rgbd(submap)
    return utils.rgbd2ptcloud(
        color, depth, submap["intrinsics"], np.eye(4))


def _pose_error(transform, reference):
    rotation = transform[:3, :3].T @ reference[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip(
        (np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    translation = float(np.linalg.norm(
        transform[:3, 3] - reference[:3, 3]) * 100.0)
    return angle, translation


def _ground_truth(datasets, fixed):
    source = datasets[fixed["source_agent_id"]][
        fixed["source_frame_id"]][-1]
    target = datasets[fixed["target_agent_id"]][
        fixed["target_frame_id"]][-1]
    source = np.asarray(
        source.detach().cpu().numpy() if hasattr(source, "detach") else source,
        dtype=np.float64)
    target = np.asarray(
        target.detach().cpu().numpy() if hasattr(target, "detach") else target,
        dtype=np.float64)
    return np.linalg.inv(target) @ source


def run_trial(fixed, method, mode, k, seed, context,
              ambiguity_margin=None, geometry=None):
    submaps = context["submaps"]
    source_submap = submaps[fixed["source_agent_id"]][
        fixed["source_submap_index"]]
    target_submap = submaps[fixed["target_agent_id"]][
        fixed["target_submap_index"]]
    source_c2w = submap_frame_c2w(
        source_submap, fixed["source_frame_id"])
    target_c2w = submap_frame_c2w(
        target_submap, fixed["target_frame_id"])
    options = dict(context["options"])
    options["ransac_seed"] = int(seed)
    if ambiguity_margin is not None:
        options["ambiguity_margin"] = float(ambiguity_margin)
    if geometry is not None:
        options["geometric_consistency"] = str(geometry)
    row = {
        **fixed,
        "method": method,
        "descriptor_mode": mode,
        "k": k,
        "seed": int(seed),
        "ambiguity_margin": ambiguity_margin,
        "geometric_consistency": geometry,
        "coarse_valid": False,
        "icp_evaluated": False,
        "valid_estimate": False,
        "passed": False,
        "failure_reason": None,
        "rotation_error_deg": None,
        "translation_error_cm": None,
        "coarse_rotation_error_deg": None,
        "coarse_translation_error_cm": None,
        "matcher_runtime_s": 0.0,
        "icp_runtime_s": 0.0,
        "online_runtime_s": 0.0,
    }
    coarse = None
    try:
        _sync()
        started = time.perf_counter()
        if method == "fpfh":
            from src.utils import utils

            source_cloud = context["dense"](
                fixed["source_agent_id"], fixed["source_submap_index"])
            target_cloud = context["dense"](
                fixed["target_agent_id"], fixed["target_submap_index"])
            try:
                o3d.utility.random.seed(int(seed))
            except AttributeError:
                pass
            coarse_start = np.asarray(
                utils.coarse_registration(source_cloud, target_cloud),
                dtype=np.float64)
            coarse = submap_start_to_loop_transform(
                coarse_start, source_submap, target_submap,
                fixed["source_frame_id"], fixed["target_frame_id"])
            diagnostics = {}
        else:
            source_shared = context["caches"][(
                fixed["source_agent_id"], fixed["source_submap_index"])]
            target_shared = context["caches"][(
                fixed["target_agent_id"], fixed["target_submap_index"])]
            source_landmarks = landmark_view(source_shared, mode, k)
            target_landmarks = landmark_view(target_shared, mode, k)
            row["source_landmarks"] = len(source_landmarks["xyz"])
            row["target_landmarks"] = len(target_landmarks["xyz"])
            coarse, diagnostics = register_landmark_sets(
                source_landmarks, target_landmarks,
                source_c2w, target_c2w, options)
        _sync()
        row["matcher_runtime_s"] = float(time.perf_counter() - started)
        for name, value in diagnostics.items():
            if value is None or isinstance(value, (str, bool, int, float, np.generic)):
                row[name] = _json_value(value) if isinstance(value, np.generic) else value
        if coarse is None:
            row["failure_reason"] = (
                diagnostics.get("failure_reason") or "coarse_failed")
            return _finish_trial(row, coarse, None, fixed, context)
        if not is_valid_rigid_transform(coarse, atol=1e-3):
            row["failure_reason"] = "invalid_rigid_rotation"
            return _finish_trial(row, None, None, fixed, context)
        row["coarse_valid"] = True
        row["T_coarse"] = coarse.tolist()
        source_cloud = context["dense"](
            fixed["source_agent_id"], fixed["source_submap_index"])
        target_cloud = context["dense"](
            fixed["target_agent_id"], fixed["target_submap_index"])
        start_transform = loop_to_submap_start_transform(
            coarse, source_submap, target_submap,
            fixed["source_frame_id"], fixed["target_frame_id"])
        roundtrip = submap_start_to_loop_transform(
            start_transform, source_submap, target_submap,
            fixed["source_frame_id"], fixed["target_frame_id"])
        if not np.allclose(roundtrip, coarse, atol=1e-8):
            raise ValueError("transformation direction/frame round-trip failed")
        row["icp_evaluated"] = True
        icp = refine_point_to_plane_icp(
            source_cloud,
            target_cloud,
            start_transform,
            distance_threshold_m=float(options.get(
                "icp_distance_threshold_m", 0.005)),
            max_iterations=int(options.get("icp_max_iterations", 500)),
        )
        row["icp_runtime_s"] = float(icp["runtime_s"])
        row["online_runtime_s"] = (
            row["matcher_runtime_s"] + row["icp_runtime_s"])
        row["initial_fitness"] = icp.get("initial_fitness")
        row["initial_rmse_m"] = icp.get("initial_rmse")
        row["final_fitness"] = icp.get("fitness")
        row["final_rmse_m"] = icp.get("rmse")
        if not icp["success"]:
            row["failure_reason"] = icp["failure_reason"]
            return _finish_trial(row, coarse, None, fixed, context)
        final = submap_start_to_loop_transform(
            icp["transformation"], source_submap, target_submap,
            fixed["source_frame_id"], fixed["target_frame_id"])
        if not is_valid_rigid_transform(final, atol=1e-3):
            row["failure_reason"] = "invalid_refined_transform"
            return _finish_trial(row, coarse, None, fixed, context)
        row["valid_estimate"] = True
        row["T_final"] = final.tolist()
        row["passed"] = bool(
            float(icp["fitness"]) > context["fitness_threshold"]
            and float(icp["rmse"]) < context["rmse_threshold"])
        if not row["passed"]:
            row["failure_reason"] = "icp_admission_failed"
        return _finish_trial(row, coarse, final, fixed, context)
    except Exception as error:
        row["failure_reason"] = f"{type(error).__name__}:{error}"
        return _finish_trial(row, coarse, None, fixed, context)


def _finish_trial(row, coarse, final, fixed, context):
    # Ground truth is deliberately queried only after registration decisions.
    try:
        reference = _ground_truth(context["datasets"], fixed)
        row["gt_available"] = True
        if coarse is not None and is_valid_rigid_transform(coarse, atol=1e-3):
            (row["coarse_rotation_error_deg"],
             row["coarse_translation_error_cm"]) = _pose_error(coarse, reference)
        if final is not None and row["valid_estimate"]:
            row["rotation_error_deg"], row["translation_error_cm"] = _pose_error(
                final, reference)
    except Exception as error:
        row["gt_available"] = False
        row["gt_failure_reason"] = f"{type(error).__name__}:{error}"
    return row


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    from src.entities.loop_detection.feature_extractors import (
        get_patch_feature_extractor,
    )

    started = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve()
    output = (args.output_dir or run_dir / "benchmark_single_multi").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (config, submaps, submap_paths, fixed, required,
     candidate_path) = load_completed_run(
         run_dir, allow_legacy_loops=args.allow_legacy_loops)
    datasets, data_root = load_agent_datasets(run_dir, config)
    options = dict(config["submap"].get("registration_options", {}))
    options.update(config.get("registration", {}))
    options["max_landmarks"] = max(args.k)
    margins = (
        args.ambiguity_margins
        if args.ambiguity_margins is not None
        else [float(options.get("ambiguity_margin", 0.0))]
    )
    geometries = (
        args.geometries
        if args.geometries is not None
        else [str(options.get("geometric_consistency", "distance"))]
    )
    registration_extractor_options = dict(config["submap"])
    registration_extractor_options.update(config.get("registration", {}))
    extractor_started = time.perf_counter()
    extractor = get_patch_feature_extractor(
        config["loop_detection"], registration_extractor_options)
    model_init_time = float(time.perf_counter() - extractor_started)
    caches, cache_rows = build_shared_caches(
        required, submaps, datasets, extractor, options)
    _write_csv(output / "descriptor_cache_runtime.csv", cache_rows)

    dense_cache, dense_rows = {}, []

    def dense(agent, submap_index):
        key = (int(agent), int(submap_index))
        if key not in dense_cache:
            begin = time.perf_counter()
            dense_cache[key] = _dense_cloud(submaps[key[0]][key[1]])
            dense_rows.append({
                "agent_id": key[0],
                "submap_index": key[1],
                "dense_cache_build_time_s_measured": float(
                    time.perf_counter() - begin),
                "points": len(dense_cache[key].points),
            })
        return dense_cache[key]

    context = {
        "submaps": submaps,
        "datasets": datasets,
        "caches": caches,
        "dense": dense,
        "options": options,
        "fitness_threshold": float(
            config["loop_detection"]["fitness_threshold"]),
        "rmse_threshold": float(
            config["loop_detection"]["inlier_rmse_threshold"]),
    }
    # Dense point-cloud construction is a separate cache-build phase, so its
    # cost cannot leak into either matcher or ICP runtime.
    for agent, submap_index in sorted(required):
        dense(agent, submap_index)
    _write_csv(output / "dense_cache_runtime.csv", dense_rows)
    protocol = {
        "candidate_source": str(candidate_path),
        "candidate_source_kind": candidate_path.name,
        "retrieval_rerun": False,
        "finalized_submaps": {
            str(agent): [str(path) for path in paths]
            for agent, paths in submap_paths.items()
        },
        "fixed_loops": fixed,
        "seeds": args.seeds,
        "k": args.k,
        "descriptor_modes": ["single", "multiview"],
        "ambiguity_margins": margins,
        "geometric_consistency": geometries,
        "same_landmark_source": True,
        "icp": {
            "distance_threshold_m": float(options.get(
                "icp_distance_threshold_m", 0.005)),
            "max_iterations": int(options.get("icp_max_iterations", 500)),
        },
        "data_root": str(data_root),
        "model_init_time_s": model_init_time,
        "cache_time_in_matcher_runtime": False,
        "benchmark_time_added_to_full_pipeline_runtime": False,
    }
    (output / "protocol.json").write_text(json.dumps(
        protocol, indent=2, default=_json_value))

    rows = []
    for fixed_loop in fixed:
        for seed in args.seeds:
            for k in args.k:
                for margin in margins:
                    for geometry in geometries:
                        for mode in ("single", "multiview"):
                            rows.append(run_trial(
                                fixed_loop, "gaussian", mode, k, seed, context,
                                ambiguity_margin=margin, geometry=geometry))
            if args.include_fpfh:
                rows.append(run_trial(
                    fixed_loop, "fpfh", "fpfh", None, seed, context))
            _write_csv(output / "trials.partial.csv", rows)
    _write_csv(output / "trials.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["method"], row["descriptor_mode"], row["k"],
            row["ambiguity_margin"], row["geometric_consistency"],
        )].append(row)
    summaries = []
    for (method, mode, k, margin, geometry), group in sorted(
            grouped.items(), key=lambda item: str(item[0])):
        summaries.append({
            "method": method,
            "descriptor_mode": mode,
            "k": k,
            "ambiguity_margin": margin,
            "geometric_consistency": geometry,
            **summarize_trials(group),
        })
    _write_csv(output / "summary.csv", summaries)
    result = {
        "benchmark_runtime_s": float(time.perf_counter() - started),
        "full_pipeline_runtime_s": None,
        "full_pipeline_runtime_includes_benchmark": False,
        "summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(
        result, indent=2, default=_json_value))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--k", nargs="+", type=int, choices=(256, 512, 1024),
        default=[256, 512, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--allow-legacy-loops", action="store_true")
    parser.add_argument("--ambiguity-margins", nargs="+", type=float)
    parser.add_argument(
        "--geometries", nargs="+",
        choices=("none", "distance", "distance_scale",
                 "distance_scale_relative_axis"))
    parser.add_argument(
        "--include-fpfh", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=_json_value))
