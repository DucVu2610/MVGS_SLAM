"""Registration building blocks for MAGMA-SLAM.

The package is intentionally independent from the SLAM orchestrator so the
geometry and matching stages can be unit-tested without a dataset or GPU.
"""

from .coarse_registration import (
    estimate_rigid_transform,
    is_valid_rigid_transform,
    ransac_rigid_transform,
    register_gaussian_landmarks,
)
from .gaussian_matching import mutual_nearest_neighbors, mutual_topk_candidates
from .geometric_consistency import pairwise_rigidity_consistency
from .landmark_selection import (
    select_spatially_balanced_landmarks,
    select_spatially_covered_correspondences,
)
from .multiview_descriptor import descriptor_cache_key

__all__ = [
    "estimate_rigid_transform",
    "is_valid_rigid_transform",
    "mutual_nearest_neighbors",
    "mutual_topk_candidates",
    "pairwise_rigidity_consistency",
    "ransac_rigid_transform",
    "register_gaussian_landmarks",
    "select_spatially_balanced_landmarks",
    "select_spatially_covered_correspondences",
    "descriptor_cache_key",
]
