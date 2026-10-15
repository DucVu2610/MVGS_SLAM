""" This module contains utility functions for key components of MAGiC-SLAM pipeline. """
import random

import numpy as np
import open3d as o3d
import roma
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.entities.gaussian_model import GaussianModel
from src.entities.losses import l1_loss, ssim
from src.mapping.gaussian_compaction import compact_gaussian_model, gaussian_model_size_bytes
from src.registration.coarse_registration import register_gaussian_landmarks
from src.registration.icp_refinement import (
    is_rigid_transform,
    loop_admission,
    loop_to_submap_start_transform,
    refine_point_to_plane_icp,
    submap_start_to_loop_transform,
)
from src.utils import utils
from src.utils.submap_utils import find_submap
from src.utils.utils import np2ptcloud, coarse_registration

from argparse import ArgumentParser
from src.entities.arguments import OptimizationParams
from src.utils.pose_utils import apply_pose_correction, optimized_pose_for_keyframe

import copy
import time

class Registration(object):
    """ A class to store the registration information between two submaps. """
    def __init__(self, source_agent_id: int, source_frame_id: int, target_agent_id: int, target_frame_id: int) -> None:
        self.source_agent_id = source_agent_id
        self.source_frame_id = source_frame_id
        self.target_agent_id = target_agent_id
        self.target_frame_id = target_frame_id
        self.source_transformation = np.eye(4)
        self.target_transformation = np.eye(4)
        self.init_transformation = np.eye(4)
        self.transformation = np.eye(4)
        self.inlier_rmse = 100.0
        self.fitness = 0.0
        self.registration_method_requested = None
        self.coarse_method_used = None
        self.fallback_used = False
        self.coarse_registration_failed = False
        self.num_source_features = None
        self.num_target_features = None
        self.num_feature_matches = None
        self.num_mutual_matches = None
        self.num_confidence_matches = None
        self.num_depth_valid_correspondences = None
        self.num_geometry_consistent_correspondences = None
        self.num_correspondences = None
        self.num_ransac_inliers = None
        self.ransac_inlier_ratio = None
        self.ransac_score = None
        self.ransac_iterations = None
        self.correspondence_retention_ratio = None
        self.mean_match_similarity = None
        self.mean_match_margin = None
        self.match_similarity_p10 = None
        self.match_similarity_p50 = None
        self.match_similarity_p90 = None
        self.match_margin_p10 = None
        self.match_margin_p50 = None
        self.match_margin_p90 = None
        self.mean_compatibility_support = None
        self.coverage_input_pairs = None
        self.coverage_strict_pairs = None
        self.coverage_filled_pairs = None
        self.fallback_triggered = False
        self.fallback_selected = False
        self.selected_stage = None
        self.dino_min_similarity = None
        self.dino_min_margin = None
        self.dino_keep_top_fraction = None
        self.dino_depth_window_radius = None
        self.dino_depth_max_mad_m = None
        self.dino_compatibility_threshold_m = None
        self.dino_min_compatibility_support = None
        self.dino_preprocess_mode = None
        self.source_patch_input_height = None
        self.source_patch_input_width = None
        self.source_patch_grid_height = None
        self.source_patch_grid_width = None
        self.target_patch_input_height = None
        self.target_patch_input_width = None
        self.target_patch_grid_height = None
        self.target_patch_grid_width = None
        self.num_gaussians_total = None
        self.num_gaussians_opacity_valid = None
        self.num_gaussians_in_view = None
        self.num_gaussians_depth_consistent = None
        self.num_gaussian_landmarks = None
        self.mean_gaussian_landmark_opacity = None
        self.mean_gaussian_depth_residual_m = None
        self.gaussian_min_opacity = None
        self.gaussian_depth_tolerance_m = None
        self.gaussian_max_landmarks = None
        self.gaussian_refit_inliers = None
        self.coarse_registration_time = None
        self.icp_registration_time = None
        self.coarse_transformation = None
        self.descriptor_mode = None
        self.descriptor_runtime_s = None
        self.matching_runtime_s = None
        self.geometric_consistency_runtime_s = None
        self.ransac_runtime_s = None
        self.num_source_landmarks = None
        self.num_target_landmarks = None
        self.accepted = None
        self.failure_reason = None
        self.T_init = None
        self.T_refined = None
        self.initial_fitness = None
        self.initial_inlier_rmse = None
        self.icp_fitness_gain = None
        self.icp_rmse_improvement_m = None


def refine_map(gaussian_model, agents_datasets: dict, agents_keyframe_ids: dict,
               agents_c2ws: dict, iterations=3000, compaction_options=None):
    """ Refine a Gaussian model using keyframes from agents' datasets: Section 3.4
    Args:
        gaussian_model: The Gaussian model.
        agents_datasets: The agents' datasets (agent_id: str -> dataset : Dataset)
        agents_keyframe_ids: The agents' keyframe IDs (agent_id: str -> keyframe_ids : np.ndarray)
        agents_c2ws: The agents' camera-to-world matrices (agent_id: str -> c2ws : np.ndarray)
        iterations: The number of iterations to refine the map.
    Returns:
        gaussian_model: The refined Gaussian model.
    """
    options = compaction_options or {}
    compaction_enabled = bool(options.get("enabled", False))
    prune_enabled = compaction_enabled and bool(
        options.get("prune_during_refinement", True))
    prune_interval = max(1, int(options.get(
        "prune_interval", options.get("refine_prune_interval", 100))))
    prune_opacity = float(options.get(
        "prune_min_opacity", options.get("refine_prune_min_opacity", 0.01)))
    prune_start = float(options.get(
        "prune_start_fraction", options.get("refine_prune_start_fraction", 0.20)))
    prune_end = float(options.get(
        "prune_end_fraction", options.get("refine_prune_end_fraction", 0.80)))
    before = int(gaussian_model.get_xyz().shape[0])
    pruned_total = 0
    started = time.perf_counter()
    print("Refining map")
    for iteration in tqdm(range(iterations)):
        agent_id = random.choice(list(agents_datasets.keys()))
        sample_id = random.randint(0, len(agents_keyframe_ids[agent_id]) - 1)
        render_data = agents_datasets[agent_id].get_render_frame(
            agents_keyframe_ids[agent_id][sample_id],
            np.linalg.inv(agents_c2ws[agent_id][sample_id]))
        gt_color, gt_depth = render_data["gt_color"], render_data["gt_depth"]

        render_dict = utils.render_gaussian_model(gaussian_model, render_data["render_settings"])
        color_loss = 0.8 * l1_loss(render_dict["color"], gt_color) + 0.2 * (1.0 - ssim(render_dict["color"], gt_color))

        depth_lss = l1_loss(render_dict["depth"], gt_depth)

        loss = color_loss + depth_lss

        loss.backward()

        with torch.no_grad():
            radii = render_dict["radii"]
            visibility_filter = radii > 0
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])

            # Disabled compaction preserves the original MAGiC-SLAM pruning
            # condition exactly.  The configurable periodic policy is active
            # only in MAGMA compaction mode.
            if not compaction_enabled:
                should_prune = iteration > iterations * 0.2 and iteration < iterations * 0.8
                threshold = 0.005
            else:
                should_prune = (
                    prune_enabled
                    and iteration > iterations * prune_start
                    and iteration < iterations * prune_end
                    and iteration % prune_interval == 0)
                threshold = prune_opacity
            if should_prune:
                prune_mask = (gaussian_model.get_opacity() < threshold).squeeze()
                num_pruned = int(prune_mask.sum().item())
                if num_pruned:
                    gaussian_model.prune_points(prune_mask)
                    pruned_total += num_pruned

            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            # gaussian_model.update_learning_rate(iteration)
    gaussian_model.refinement_stats = {
        "enabled": compaction_enabled,
        "iterations": int(iterations),
        "gaussians_before_refinement": before,
        "gaussians_pruned_during_refinement": pruned_total,
        "gaussians_after_refinement": int(gaussian_model.get_xyz().shape[0]),
        "model_bytes_after_refinement": gaussian_model_size_bytes(gaussian_model),
        "refinement_runtime_s": float(time.perf_counter() - started),
        "prune_during_refinement": prune_enabled,
        "prune_interval": prune_interval,
        "prune_min_opacity": prune_opacity if compaction_enabled else 0.005,
    }
    return gaussian_model


def merge_submaps(agents_submaps: dict, agents_kf_ids: dict,
                  agents_opt_kf_c2ws: dict, opt_args,
                  compaction_options=None):
    """ Merge submaps from agents in a coarse manner: Section 3.4
    Args:
        agents_submaps: A dictionary of agent submaps.
        agents_kf_ids: A dictionary of agent keyframe IDs.
        agents_opt_kf_c2ws: A dictionary of agent optimized camera-to-world matrices.
        opt_args: The optimization arguments.
    Returns:
        merged_map: The merged Gaussian model.
    """
    merged_map = GaussianModel(0)
    merged_map.training_setup(opt_args)
    device = "cuda"

    merge_started = time.perf_counter()
    before_merge = sum(
        int(submap["gaussian_model_params"]["xyz"].shape[0])
        for submaps in agents_submaps.values() for submap in submaps)
    print("Merging submaps")
    for agent_id in tqdm(sorted(agents_submaps.keys())):

        for _, submap in enumerate(agents_submaps[agent_id][::-1]):

            xyz = submap["gaussian_model_params"]["xyz"].to(device)
            rotations = submap["gaussian_model_params"]["rotation"].to(device)
            features_dc = submap["gaussian_model_params"]["features_dc"].to(device)
            features_rest = submap["gaussian_model_params"]["features_rest"].to(device)
            opacity = submap["gaussian_model_params"]["opacity"].to(device)
            scaling = submap["gaussian_model_params"]["scaling"].to(device)

            submap_opt_c2w = optimized_pose_for_keyframe(
                agents_kf_ids[agent_id],
                agents_opt_kf_c2ws[agent_id],
                submap["submap_start_frame_id"],
                label=f"agent {agent_id}",
            )
            submap_c2w = submap["submap_c2ws"][0]
            delta = utils.np2torch(submap_opt_c2w @ np.linalg.inv(submap_c2w), device=device)

            opt_xyz = xyz @ delta[:3, :3].T + delta[:3, 3]

            rotation_matrices = roma.unitquat_to_rotmat(rotations)
            opt_rotations = delta[:3, :3][None] @ rotation_matrices
            opt_rotations = roma.rotmat_to_unitquat(opt_rotations)

            merged_map.densification_postfix(
                opt_xyz,
                features_dc,
                features_rest,
                opacity,
                scaling,
                opt_rotations)

    after_merge = int(merged_map.get_xyz().shape[0])
    merge_runtime = float(time.perf_counter() - merge_started)
    merged_map = compact_gaussian_model(merged_map, compaction_options)
    merged_map.compaction_stats.update({
        "gaussians_before_merge": before_merge,
        "gaussians_after_merge": after_merge,
        "merge_runtime_s": merge_runtime,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
    })
    return merged_map


def register_submaps(agents_submaps: dict, registration: Registration):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    source_cloud = agents_submaps[registration.source_agent_id]
    source_submap = find_submap(registration.source_frame_id, agents_submaps[registration.source_agent_id])
    target_submap = find_submap(registration.target_frame_id, agents_submaps[registration.target_agent_id])

    source_cloud = np2ptcloud(source_submap["point_cloud"][:, :3], source_submap["point_cloud"][:, 3:] / 255.0)
    target_cloud = np2ptcloud(target_submap["point_cloud"][:, :3], target_submap["point_cloud"][:, 3:] / 255.0)

    source_cloud.estimate_normals()
    target_cloud.estimate_normals()

    distance_threshold = 0.005
    fine_alignment = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, registration.init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500,
                                                          relative_fitness=1e-9, relative_rmse=1e-9))

    registration.transformation = fine_alignment.transformation
    registration.fitness = fine_alignment.fitness
    registration.inlier_rmse = fine_alignment.inlier_rmse

    return registration


def register_agents_submaps(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 5) -> list:
    """ Register the submaps of the agents in parallel.
    Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        max_threads: The maximum number of threads to use.
    """
    registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
        agents_submaps, registration) for registration in registrations)
    return registrations

def register_submaps_depth(agents_submaps: dict, registration: Registration,
                           initial_transformation_unknown: bool = True,
                           registration_method: str = "fpfh",
                           feature_extractor=None,
                           fallback_to_fpfh: bool = True,
                           registration_options: dict = None,
                           agents_datasets: dict = None,
                           landmark_cache: dict = None):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
        initial_transformation_unknown: If True, use a coarse registration as the
            initial guess for inter-agent loops. If False, the odometry-based init_transformation
            set by detect_loops() is used directly (assumes a known relative pose between agents).
        registration_method: "fpfh", "dinov2", "gaussian_landmark", "sift",
            "orb", or "akaze"
            for the coarse registration used when initial_transformation_unknown is True.
        feature_extractor: DINOv2 feature extractor, required by "dinov2" and
            "gaussian_landmark".
        fallback_to_fpfh: If True, use FPFH when a visual method produces too
            few valid correspondences. Disable this for pure extractor benchmarks.
        registration_options: Shared RANSAC/correspondence settings plus sparse
            local-feature settings. Defaults preserve the original DINOv2 path.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    options = dict(registration_options or {})
    registration_method = registration_method.lower()
    supported_methods = {
        "fpfh", "dinov2", "gaussian_landmark", "sift", "orb", "akaze"
    }
    if registration_method not in supported_methods:
        raise ValueError(
            f"Unknown registration_method={registration_method!r}; "
            f"expected one of {sorted(supported_methods)}")
    registration.registration_method_requested = registration_method
    registration.descriptor_mode = options.get("descriptor_mode", "baseline")
    source_submap = find_submap(
        registration.source_frame_id,
        agents_submaps[registration.source_agent_id])
    target_submap = find_submap(
        registration.target_frame_id,
        agents_submaps[registration.target_agent_id])
    if source_submap is None or target_submap is None:
        registration.failure_reason = "submap_not_found"
        registration.coarse_registration_failed = True
        registration.accepted = False
        registration.transformation = None
        return registration

    try:
        source_color, source_depth = utils.get_rgbd(source_submap)
        target_color, target_depth = utils.get_rgbd(target_submap)
        source_cloud = utils.rgbd2ptcloud(
            source_color, source_depth, source_submap["intrinsics"], np.eye(4))
        target_cloud = utils.rgbd2ptcloud(
            target_color, target_depth, target_submap["intrinsics"], np.eye(4))
    except Exception as error:
        registration.failure_reason = f"dense_geometry_failed:{type(error).__name__}:{error}"
        registration.accepted = False
        registration.transformation = None
        return registration

    min_correspondences = int(options.get("min_correspondences", 8))
    ransac_distance_threshold = float(options.get(
        "ransac_distance_threshold_m",
        options.get("ransac_distance_threshold", 0.05)))
    is_inter_agent = source_submap["agent_id"] != target_submap["agent_id"]
    dense_initial_transform = None
    if is_inter_agent and initial_transformation_unknown:
        coarse_start = time.perf_counter()
        transform, diagnostics = None, {}
        transform_is_loop_frame = False
        try:
            descriptor_mode = str(options.get("descriptor_mode", "baseline")).lower()
            if (registration_method == "gaussian_landmark"
                    and descriptor_mode in {"single", "multiview"}):
                if agents_datasets is None:
                    raise ValueError("single/multiview descriptors require agent datasets")
                magma_options = dict(options)
                magma_options.setdefault(
                    "similarity_threshold", options.get("dino_min_similarity", -1.0))
                magma_options.setdefault(
                    "ambiguity_margin", max(0.0, options.get("dino_min_margin", 0.0)))
                magma_options.setdefault(
                    "compatibility_threshold_m",
                    options.get("dino_compatibility_threshold_m", 0.03))
                magma_options.setdefault(
                    "min_compatibility_support",
                    options.get("dino_min_compatibility_support", 16))
                magma_options.setdefault("ransac_distance_threshold_m", ransac_distance_threshold)
                transform, diagnostics = register_gaussian_landmarks(
                    source_submap=source_submap,
                    target_submap=target_submap,
                    source_frame_id=registration.source_frame_id,
                    target_frame_id=registration.target_frame_id,
                    source_dataset=agents_datasets[registration.source_agent_id],
                    target_dataset=agents_datasets[registration.target_agent_id],
                    feature_extractor=feature_extractor,
                    descriptor_mode=descriptor_mode,
                    options=magma_options,
                    cache=landmark_cache,
                )
                transform_is_loop_frame = True
            elif registration_method == "dinov2":
                if feature_extractor is None:
                    raise ValueError("DINOv2 registration requires a patch extractor")
                transform, diagnostics = utils.coarse_registration_dinov2(
                    source_color, source_depth, target_color, target_depth,
                    source_submap["intrinsics"], target_submap["intrinsics"],
                    feature_extractor,
                    distance_threshold=ransac_distance_threshold,
                    min_correspondences=min_correspondences,
                    return_diagnostics=True,
                    min_similarity=options.get("dino_min_similarity", -1.0),
                    min_margin=options.get("dino_min_margin", -1.0),
                    keep_top_fraction=options.get("dino_keep_top_fraction", 1.0),
                    depth_window_radius=options.get("dino_depth_window_radius", 0),
                    depth_max_mad_m=options.get("dino_depth_max_mad_m"),
                    compatibility_threshold_m=options.get(
                        "dino_compatibility_threshold_m", 0.0),
                    min_compatibility_support=options.get(
                        "dino_min_compatibility_support", 0))
            elif registration_method == "gaussian_landmark":
                # Legacy single-reference image-to-Gaussian path retained for
                # old configs whose descriptor_mode is absent/baseline.
                target_gaussian = utils.load_3dgs(target_submap)
                try:
                    transform, diagnostics = utils.coarse_registration_gaussian_landmark(
                        source_color, source_depth, target_color, target_depth,
                        source_submap["intrinsics"], target_submap["intrinsics"],
                        target_gaussian, np.asarray(target_submap["submap_c2ws"][0]),
                        feature_extractor,
                        distance_threshold=ransac_distance_threshold,
                        min_correspondences=min_correspondences,
                        return_diagnostics=True,
                        min_similarity=options.get("dino_min_similarity", -1.0),
                        min_margin=options.get("dino_min_margin", -1.0),
                        keep_top_fraction=options.get("dino_keep_top_fraction", 1.0),
                        depth_window_radius=options.get("dino_depth_window_radius", 0),
                        depth_max_mad_m=options.get("dino_depth_max_mad_m"),
                        compatibility_threshold_m=options.get(
                            "dino_compatibility_threshold_m", 0.0),
                        min_compatibility_support=options.get(
                            "dino_min_compatibility_support", 0),
                        gaussian_min_opacity=options.get("gaussian_min_opacity", 0.05),
                        gaussian_depth_tolerance_m=options.get(
                            "gaussian_depth_tolerance_m", 0.05),
                        gaussian_max_landmarks=options.get("gaussian_max_landmarks", 2048),
                        gaussian_refit_inliers=options.get("gaussian_refit_inliers", True))
                finally:
                    del target_gaussian
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif registration_method in {"sift", "orb", "akaze"}:
                transform, diagnostics = utils.coarse_registration_local_features(
                    source_color, source_depth, target_color, target_depth,
                    source_submap["intrinsics"], target_submap["intrinsics"],
                    method=registration_method,
                    distance_threshold=ransac_distance_threshold,
                    min_correspondences=min_correspondences,
                    max_keypoints=options.get("max_keypoints", 4096),
                    ratio_threshold=options.get("ratio_threshold", 0.8))
        except Exception as error:
            registration.failure_reason = (
                f"coarse_failed:{type(error).__name__}:{error}")
            transform = None

        for field, value in diagnostics.items():
            setattr(registration, field, value)
        if transform is None and diagnostics.get("failure_reason"):
            registration.failure_reason = diagnostics["failure_reason"]
        if (transform is not None and diagnostics
                and int(diagnostics.get("num_ransac_inliers", 3))
                < int(options.get("min_ransac_inliers", 3))):
            transform = None
            registration.failure_reason = "insufficient_ransac_inliers"
        if transform is not None and not is_rigid_transform(transform, atol=1e-3):
            transform = None
            registration.failure_reason = "invalid_rigid_rotation"
        if transform is None:
            if registration_method == "fpfh" or fallback_to_fpfh:
                try:
                    transform = utils.coarse_registration(source_cloud, target_cloud)
                    transform_is_loop_frame = False
                    registration.coarse_method_used = "fpfh"
                    registration.fallback_used = registration_method != "fpfh"
                    registration.failure_reason = None
                except Exception as error:
                    registration.failure_reason = (
                        f"fpfh_failed:{type(error).__name__}:{error}")
            if transform is None:
                registration.coarse_registration_failed = True
                registration.coarse_registration_time = time.perf_counter() - coarse_start
                registration.accepted = False
                registration.transformation = None
                return registration
        else:
            registration.coarse_method_used = registration_method
        registration.coarse_registration_time = time.perf_counter() - coarse_start
        try:
            if transform_is_loop_frame:
                loop_initial_transform = np.asarray(transform, dtype=np.float64)
                dense_initial_transform = loop_to_submap_start_transform(
                    loop_initial_transform,
                    source_submap, target_submap,
                    registration.source_frame_id, registration.target_frame_id)
            else:
                dense_initial_transform = np.asarray(transform, dtype=np.float64)
                loop_initial_transform = submap_start_to_loop_transform(
                    dense_initial_transform,
                    source_submap, target_submap,
                    registration.source_frame_id, registration.target_frame_id)
            if (not is_rigid_transform(loop_initial_transform, atol=1e-3)
                    or not is_rigid_transform(dense_initial_transform, atol=1e-3)):
                raise ValueError("invalid rigid rotation after frame conversion")
        except Exception as error:
            registration.failure_reason = (
                f"frame_conversion_failed:{type(error).__name__}:{error}")
            registration.coarse_registration_failed = True
            registration.accepted = False
            registration.transformation = None
            return registration
        registration.init_transformation = loop_initial_transform
        registration.coarse_transformation = loop_initial_transform.copy()

    if dense_initial_transform is None:
        # LoopDetector's odometry initialization is expressed between the two
        # submap-start camera frames, matching the dense RGB-D clouds.
        dense_initial_transform = np.asarray(
            registration.init_transformation, dtype=np.float64)
        if not is_rigid_transform(dense_initial_transform, atol=1e-3):
            registration.failure_reason = "invalid_initial_transform"
            registration.accepted = False
            registration.transformation = None
            return registration
        registration.init_transformation = submap_start_to_loop_transform(
            dense_initial_transform,
            source_submap, target_submap,
            registration.source_frame_id, registration.target_frame_id)

    registration.T_init = np.asarray(registration.init_transformation).copy()
    icp = refine_point_to_plane_icp(
        source_cloud,
        target_cloud,
        dense_initial_transform,
        distance_threshold_m=float(options.get("icp_distance_threshold_m", 0.005)),
        max_iterations=int(options.get(
            "icp_max_iterations",
            500)),
    )
    registration.icp_registration_time = icp["runtime_s"]
    registration.initial_fitness = icp.get("initial_fitness")
    registration.initial_inlier_rmse = icp.get("initial_rmse")
    registration.icp_fitness_gain = icp.get("fitness_gain")
    registration.icp_rmse_improvement_m = icp.get("rmse_improvement_m")
    if not icp["success"]:
        registration.failure_reason = icp["failure_reason"]
        registration.accepted = False
        registration.transformation = None
        return registration
    registration.transformation = submap_start_to_loop_transform(
        icp["transformation"],
        source_submap, target_submap,
        registration.source_frame_id, registration.target_frame_id)
    registration.T_refined = np.asarray(registration.transformation).copy()
    if not is_rigid_transform(registration.transformation, atol=1e-3):
        registration.failure_reason = "invalid_refined_transform"
        registration.accepted = False
        registration.transformation = None
        registration.T_refined = None
        return registration
    registration.fitness = icp["fitness"]
    registration.inlier_rmse = icp["rmse"]
    if registration.descriptor_mode == "baseline" and "admission" not in options:
        # Preserve the legacy LoopDetector's strict >/< filtering for old
        # configs. New modes use the explicit MAGMA admission gate below.
        registration.accepted = None
        registration.failure_reason = None
    else:
        admission = options.get("admission", {})
        registration.accepted, reason = loop_admission(
            registration.fitness,
            registration.inlier_rmse,
            min_fitness=float(admission.get("min_fitness", 0.30)),
            max_rmse_m=float(admission.get("max_rmse_m", 0.005)),
        )
        registration.failure_reason = reason
    return registration

def move_to_device(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    elif isinstance(obj, set):
        return {move_to_device(v, device) for v in obj}
    else:
        return obj # Leave non-tensor types unchanged

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def get_information_matrix(source_cloud, target_cloud, distance_threshold, transformation, fitness, threshold=0.5):
    """ Compute the information matrix for the registration.
    Args:
        source_cloud: The source point cloud.
        target_cloud: The target point cloud.
        transformation: The transformation matrix.
    Returns:
        information_matrix: The information matrix.
    """
    if not isinstance(source_cloud, o3d.t.geometry.PointCloud):
        source_cloud_t = o3d.t.geometry.PointCloud.from_legacy(source_cloud)
    else:
        source_cloud_t = source_cloud

    if not isinstance(target_cloud, o3d.t.geometry.PointCloud):
        target_cloud_t = o3d.t.geometry.PointCloud.from_legacy(target_cloud)
    else:
        target_cloud_t = target_cloud
    # Now get_information_matrix will work
    if fitness > threshold:
        info_matrix = o3d.t.pipelines.registration.get_information_matrix(
            source_cloud_t, target_cloud_t, distance_threshold, transformation)
    else:
        info_matrix = o3d.core.Tensor.zeros((6, 6), dtype=o3d.core.float64)
    
    # Reorder from Open3D [tx,ty,tz,rx,ry,rz] to GTSAM [rx,ry,rz,tx,ty,tz]
    reorder_indices = [3, 4, 5, 0, 1, 2]
    info_matrix = info_matrix[np.ix_(reorder_indices, reorder_indices)].numpy()

    return info_matrix

def register_agents_submaps_depth(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 1)-> list:
    """ Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        intrinsics: The camera intrinsics to use for depth registration.
    """
    if max_threads < 2:
        registrations = [registration_method(agents_submaps, registration) for registration in registrations]
    else:
       registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
           agents_submaps, registration) for registration in registrations)
    return registrations
