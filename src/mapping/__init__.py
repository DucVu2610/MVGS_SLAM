"""Global-map post-processing for MAGMA-SLAM."""

from .gaussian_compaction import compact_gaussian_model, select_compaction_indices

__all__ = ["compact_gaussian_model", "select_compaction_indices"]
