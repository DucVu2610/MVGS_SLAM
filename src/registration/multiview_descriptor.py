"""Single- and multi-view DINO descriptors for 3D Gaussian landmarks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .landmark_selection import (
    landmark_reliability,
    select_spatially_balanced_landmarks,
)


def _rgb_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"RGB image must be three-dimensional, received {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.5:
        array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _depth_tensor(value, device) -> torch.Tensor:
    depth = torch.as_tensor(value, device=device, dtype=torch.float32).squeeze()
    if depth.ndim == 3:
        depth = depth[0] if depth.shape[0] <= 4 else depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must reduce to HxW, received {tuple(depth.shape)}")
    return depth


def project_gaussians(
    xyz_world,
    intrinsics,
    c2w,
    image_shape: Tuple[int, int],
):
    """Project world-frame Gaussian centres into an RGB-D camera."""
    xyz = torch.as_tensor(xyz_world)
    device, dtype = xyz.device, xyz.dtype
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_world must have shape [N, 3]")
    camera_to_world = torch.as_tensor(c2w, device=device, dtype=dtype)
    intrinsic = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    if camera_to_world.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("c2w and intrinsics must be 4x4 and 3x3")
    height, width = map(int, image_shape)
    world_to_camera = torch.linalg.inv(camera_to_world)
    camera = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    safe_z = z.clamp_min(1e-8)
    u = intrinsic[0, 0] * camera[:, 0] / safe_z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / safe_z + intrinsic[1, 2]
    in_image = (
        torch.isfinite(camera).all(dim=1)
        & (z > 0)
        & (u >= 0)
        & (u <= width - 1)
        & (v >= 0)
        & (v <= height - 1)
    )
    return camera, u, v, in_image


def depth_consistent_projection(
    xyz_world,
    opacity,
    depth,
    intrinsics,
    c2w,
    min_opacity: float = 0.05,
    depth_tolerance_m: float = 0.05,
):
    """Project and retain visible, depth-consistent Gaussian centres."""
    depth = _depth_tensor(depth, torch.as_tensor(xyz_world).device)
    camera, u, v, in_image = project_gaussians(
        xyz_world, intrinsics, c2w, tuple(depth.shape))
    opacity = torch.as_tensor(
        opacity, device=camera.device, dtype=camera.dtype).reshape(-1)
    if len(opacity) != len(camera):
        raise ValueError("opacity and xyz lengths differ")
    in_image &= torch.isfinite(opacity) & (opacity >= float(min_opacity))
    candidate = torch.nonzero(in_image).squeeze(1)
    if candidate.numel() == 0:
        empty = torch.empty(0, device=camera.device, dtype=torch.long)
        return empty, u, v, torch.empty(0, device=camera.device, dtype=camera.dtype)
    ui = torch.round(u[candidate]).long().clamp(0, depth.shape[1] - 1)
    vi = torch.round(v[candidate]).long().clamp(0, depth.shape[0] - 1)
    observed = depth[vi, ui]
    residual = torch.abs(camera[candidate, 2] - observed)
    consistent = (
        torch.isfinite(observed)
        & torch.isfinite(residual)
        & (observed > 0)
        & (residual <= float(depth_tolerance_m))
    )
    return candidate[consistent], u, v, residual[consistent]


def depth_consistency_weights(residual_m, temperature_m: float) -> torch.Tensor:
    """Compute :math:`exp(-r/tau)` observation weights."""
    if float(temperature_m) <= 0:
        raise ValueError("temperature_m must be positive")
    residual = torch.as_tensor(residual_m)
    return torch.exp(-residual / float(temperature_m))


def bilinear_sample_descriptors(
    patch_tokens,
    grid_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    u,
    v,
):
    """Bilinearly sample a DINO patch grid at original image coordinates."""
    tokens = torch.as_tensor(patch_tokens).float()
    grid_height, grid_width = map(int, grid_shape)
    image_height, image_width = map(int, image_shape)
    if tokens.ndim != 2 or len(tokens) != grid_height * grid_width:
        raise ValueError("patch token count does not match grid_shape")
    u = torch.as_tensor(u, device=tokens.device, dtype=tokens.dtype)
    v = torch.as_tensor(v, device=tokens.device, dtype=tokens.dtype)
    gx = u * grid_width / image_width - 0.5
    gy = v * grid_height / image_height - 0.5
    x_normalized = (
        2.0 * gx / (grid_width - 1) - 1.0
        if grid_width > 1 else torch.zeros_like(gx)
    )
    y_normalized = (
        2.0 * gy / (grid_height - 1) - 1.0
        if grid_height > 1 else torch.zeros_like(gy)
    )
    sample_grid = torch.stack((x_normalized, y_normalized), dim=-1).view(1, -1, 1, 2)
    feature_map = tokens.reshape(grid_height, grid_width, -1).permute(2, 0, 1)[None]
    sampled = F.grid_sample(
        feature_map,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, :, 0].T
    return F.normalize(sampled, dim=1, eps=1e-12)


def aggregate_multiview_descriptors(
    descriptor_sum,
    weight_sum,
):
    """Normalize weighted descriptor sums and return view stability."""
    descriptor_sum = torch.as_tensor(descriptor_sum).float()
    weight_sum = torch.as_tensor(
        weight_sum, device=descriptor_sum.device, dtype=descriptor_sum.dtype).reshape(-1)
    if descriptor_sum.ndim != 2 or len(descriptor_sum) != len(weight_sum):
        raise ValueError("descriptor_sum and weight_sum shapes are incompatible")
    descriptors = F.normalize(descriptor_sum, dim=1, eps=1e-12)
    stability = torch.linalg.norm(descriptor_sum, dim=1) / weight_sum.clamp_min(1e-12)
    return descriptors, stability.clamp(0.0, 1.0)


def _checkpoint_gaussians(submap, device):
    params = submap["gaussian_model_params"]
    xyz = torch.as_tensor(params["xyz"], device=device, dtype=torch.float32).detach()
    raw_opacity = torch.as_tensor(
        params["opacity"], device=device, dtype=torch.float32).detach().reshape(-1)
    opacity = torch.sigmoid(raw_opacity)
    scaling = torch.exp(torch.as_tensor(
        params.get("scaling", torch.zeros((len(xyz), 3))),
        device=device, dtype=torch.float32).detach())
    rotation = torch.as_tensor(
        params.get("rotation", torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(xyz), 1)),
        device=device, dtype=torch.float32).detach()
    return xyz, opacity, scaling, rotation


def _rotation_axes_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion.float(), dim=1, eps=1e-12)
    w, x, y, z = quaternion.unbind(dim=1)
    rows = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=1)
    return rows.reshape(-1, 3, 3)


def _frame_pose(submap, frame_id: int):
    start = int(submap["submap_start_frame_id"])
    offset = int(frame_id) - start
    poses = submap["submap_c2ws"]
    if offset < 0 or offset >= len(poses):
        raise IndexError(
            f"frame {frame_id} maps to pose offset {offset}, but submap has {len(poses)} poses")
    return poses[offset]


@torch.inference_mode()
def build_gaussian_landmark_descriptors(
    submap: Dict[str, Any],
    dataset,
    feature_extractor,
    descriptor_mode: str = "multiview",
    options: Dict[str, Any] | None = None,
):
    """Build selected Gaussian landmarks from all valid submap keyframes.

    Returned ``xyz`` remains in the submap/agent world frame.  The coarse
    registration stage explicitly converts it to the source and target loop
    camera frames before estimating a source-to-target transform.
    """
    options = options or {}
    mode = str(descriptor_mode).lower()
    if mode not in {"single", "multiview"}:
        raise ValueError("descriptor_mode must be 'single' or 'multiview'")
    if feature_extractor is None or not hasattr(feature_extractor, "extract_patch_tokens"):
        raise ValueError("a patch-token feature extractor is required")

    device = torch.device(getattr(feature_extractor, "device", "cpu"))
    xyz, opacity, scaling, rotation = _checkpoint_gaussians(submap, device)
    total = len(xyz)
    keyframe_ids = torch.as_tensor(submap["keyframe_ids"]).reshape(-1).tolist()
    intrinsics = getattr(dataset, "intrinsics", submap.get("intrinsics"))
    min_opacity = float(options.get("gaussian_min_opacity", 0.05))
    depth_tolerance = float(options.get("gaussian_depth_tolerance_m", 0.05))
    temperature = float(options.get("depth_weight_temperature_m", 0.05))
    max_landmarks = int(options.get(
        "max_landmarks", options.get("gaussian_max_landmarks", 512)))
    # V3.3.1 compares descriptors, not landmark availability: single-best and
    # multi-view must therefore use the same eligible Gaussian identities.
    min_observations = int(options.get("min_observations", 2))

    descriptor_sum = None
    best_descriptor = None
    best_weight = torch.full((total,), -1.0, device=device)
    weight_sum = torch.zeros(total, device=device)
    residual_sum = torch.zeros(total, device=device)
    observation_count = torch.zeros(total, device=device, dtype=torch.int32)
    views_used = 0

    for frame_id_value in keyframe_ids:
        frame_id = int(frame_id_value)
        _, image, depth, _ = dataset[frame_id]
        pil_image = _rgb_image(image)
        width, height = pil_image.size
        tokens, grid_shape, _, _ = feature_extractor.extract_patch_tokens(
            pil_image, return_metadata=True)
        tokens = F.normalize(tokens.float(), dim=1, eps=1e-12)
        if tokens.device != device:
            tokens = tokens.to(device)
        if descriptor_sum is None:
            descriptor_sum = torch.zeros(
                (total, tokens.shape[1]), device=device, dtype=torch.float32)
            best_descriptor = torch.zeros_like(descriptor_sum)
        elif tokens.shape[1] != descriptor_sum.shape[1]:
            raise ValueError("feature extractor changed descriptor dimension between views")

        depth_tensor = _depth_tensor(depth, device)
        if tuple(depth_tensor.shape) != (height, width):
            depth_tensor = F.interpolate(
                depth_tensor[None, None], size=(height, width), mode="nearest")[0, 0]
        valid, u, v, residual = depth_consistent_projection(
            xyz,
            opacity,
            depth_tensor,
            intrinsics,
            _frame_pose(submap, frame_id),
            min_opacity=min_opacity,
            depth_tolerance_m=depth_tolerance,
        )
        if valid.numel() == 0:
            continue
        descriptors = bilinear_sample_descriptors(
            tokens, grid_shape, (height, width), u[valid], v[valid])
        weights = depth_consistency_weights(residual, temperature).float()
        descriptor_sum.index_add_(0, valid, descriptors * weights[:, None])
        weight_sum.index_add_(0, valid, weights)
        residual_sum.index_add_(0, valid, residual)
        observation_count.index_add_(
            0, valid, torch.ones_like(valid, dtype=torch.int32))
        better = weights > best_weight[valid]
        if bool(better.any()):
            update = valid[better]
            best_weight[update] = weights[better]
            best_descriptor[update] = descriptors[better]
        views_used += 1

    if descriptor_sum is None:
        raise RuntimeError("no keyframe descriptors could be extracted")
    eligible = (
        (observation_count >= min_observations)
        & (weight_sum > 0)
        & (opacity >= min_opacity)
    )
    eligible_indices = torch.nonzero(eligible).squeeze(1)
    if eligible_indices.numel() < 3:
        raise RuntimeError(
            f"only {eligible_indices.numel()} Gaussian landmarks have "
            f"{min_observations}+ valid observation(s)")

    multi_descriptor, stability_all = aggregate_multiview_descriptors(
        descriptor_sum[eligible_indices], weight_sum[eligible_indices])
    single_descriptor = F.normalize(
        best_descriptor[eligible_indices], dim=1, eps=1e-12)
    mean_residual = (
        residual_sum[eligible_indices]
        / observation_count[eligible_indices].float().clamp_min(1))
    reliability = landmark_reliability(
        stability_all.cpu().numpy(),
        observation_count[eligible_indices].cpu().numpy(),
        mean_residual.cpu().numpy(),
        opacity[eligible_indices].cpu().numpy(),
        temperature,
    )
    selected_local = select_spatially_balanced_landmarks(
        xyz[eligible_indices].cpu().numpy(),
        reliability,
        max_landmarks=max_landmarks,
        voxel_size_m=float(options.get("landmark_voxel_size_m", 0.05)),
    )
    selected_local_tensor = torch.as_tensor(selected_local, device=device)
    selected = eligible_indices[selected_local_tensor]
    descriptors = (
        single_descriptor[selected_local_tensor]
        if mode == "single" else multi_descriptor[selected_local_tensor]
    )
    axes = _rotation_axes_wxyz(rotation[selected])

    result = {
        "xyz": xyz[selected].detach().cpu(),
        "descriptor_single_best_view": single_descriptor[
            selected_local_tensor].detach().cpu(),
        "descriptor_multi_view": multi_descriptor[
            selected_local_tensor].detach().cpu(),
        "gaussian_indices": selected.detach().cpu(),
        "stability": stability_all[selected_local_tensor].detach().cpu(),
        "observation_count": observation_count[selected].detach().cpu(),
        "mean_depth_residual_m": mean_residual[selected_local_tensor].detach().cpu(),
        "opacity": opacity[selected].detach().cpu(),
        "reliability": torch.as_tensor(reliability[selected_local]),
        "scales": scaling[selected].detach().cpu(),
        "axes": axes.detach().cpu(),
        "descriptor_mode": mode,
        "diagnostics": {
            "num_gaussians_total": int(total),
            "num_gaussians_opacity_valid": int((opacity >= min_opacity).sum().item()),
            "num_gaussians_multiview": int(eligible_indices.numel()),
            "num_gaussian_landmarks": int(len(selected)),
            "num_views_used": int(views_used),
            "mean_gaussian_landmark_opacity": float(opacity[selected].mean().item()),
            "mean_gaussian_depth_residual_m": float(
                mean_residual[selected_local_tensor].mean().item()),
            "mean_descriptor_stability": float(
                stability_all[selected_local_tensor].mean().item()),
        },
    }
    result["descriptors"] = result[
        "descriptor_single_best_view" if mode == "single"
        else "descriptor_multi_view"
    ]
    return result


def points_in_camera_frame(points_world, c2w):
    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(c2w, dtype=np.float64)
    world_to_camera = np.linalg.inv(pose)
    return points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
