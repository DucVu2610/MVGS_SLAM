"""Descriptor matching for Gaussian landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MatchResult:
    source_indices: torch.Tensor
    target_indices: torch.Tensor
    similarity: torch.Tensor
    ambiguity_margin: torch.Tensor
    num_mutual: int
    margin_quality: torch.Tensor | None = None
    rank_quality: torch.Tensor | None = None


def mutual_topk_candidates(
    source_descriptors,
    target_descriptors,
    top_k: int = 5,
    similarity_threshold: float = -1.0,
) -> MatchResult:
    """Build the V3.2 hard-case mutual Top-K candidate pool."""
    source = torch.as_tensor(source_descriptors).float()
    target = torch.as_tensor(target_descriptors).float().to(source.device)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("descriptors must have shape [N, D]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target descriptor dimensions differ")
    if source.shape[0] == 0 or target.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        values = torch.empty(0, dtype=source.dtype, device=source.device)
        return MatchResult(empty, empty, values, values, 0)

    source = F.normalize(source, dim=1, eps=1e-12)
    target = F.normalize(target, dim=1, eps=1e-12)
    similarity = source @ target.T
    source_k = min(max(1, int(top_k)), similarity.shape[1])
    target_k = min(max(1, int(top_k)), similarity.shape[0])
    source_values, source_neighbors = similarity.topk(source_k, dim=1)
    target_values, target_neighbors = similarity.topk(target_k, dim=0)
    target_membership = torch.zeros_like(similarity, dtype=torch.bool)
    target_rank = torch.full_like(similarity, int(top_k), dtype=torch.long)
    target_columns = torch.arange(similarity.shape[1], device=source.device)
    for rank in range(target_k):
        target_membership[target_neighbors[rank], target_columns] = True
        target_rank[target_neighbors[rank], target_columns] = rank

    source_ids = torch.arange(similarity.shape[0], device=source.device)
    source_ids = source_ids[:, None].expand(-1, source_k).reshape(-1)
    target_ids = source_neighbors.reshape(-1)
    source_ranks = torch.arange(
        source_k, device=source.device)[None].expand(
            similarity.shape[0], -1).reshape(-1)
    mutual = target_membership[source_ids, target_ids]
    source_ranks = source_ranks[mutual]
    source_ids, target_ids = source_ids[mutual], target_ids[mutual]
    values = similarity[source_ids, target_ids]

    source_margin = (
        source_values[:, 0] - source_values[:, 1]
        if source_k >= 2 else torch.full(
            (len(source),), float("inf"), device=source.device)
    )
    target_margin = (
        target_values[0] - target_values[1]
        if target_k >= 2 else torch.full(
            (len(target),), float("inf"), device=source.device)
    )
    margins = torch.minimum(source_margin[source_ids], target_margin[target_ids])
    source_margin_quality = torch.clamp(
        source_margin[source_ids] / 0.060, 0.0, 1.0)
    target_margin_quality = torch.clamp(
        target_margin[target_ids] / 0.060, 0.0, 1.0)
    margin_quality = torch.sqrt(
        source_margin_quality * target_margin_quality)
    denominator = max(2.0 * (int(top_k) - 1), 1.0)
    rank_quality = torch.clamp(
        1.0 - (
            source_ranks + target_rank[source_ids, target_ids]
        ).float() / denominator,
        0.0, 1.0,
    )
    keep = values >= float(similarity_threshold)
    return MatchResult(
        source_indices=source_ids[keep],
        target_indices=target_ids[keep],
        similarity=values[keep],
        ambiguity_margin=margins[keep],
        num_mutual=int(mutual.sum().item()),
        margin_quality=margin_quality[keep],
        rank_quality=rank_quality[keep],
    )


def mutual_nearest_neighbors(
    source_descriptors,
    target_descriptors,
    similarity_threshold: float = -1.0,
    ambiguity_margin: float = 0.0,
) -> MatchResult:
    """Cosine mutual-NN matching with a bidirectional Top-1/Top-2 margin."""
    source = torch.as_tensor(source_descriptors).float()
    target = torch.as_tensor(target_descriptors).float().to(source.device)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("descriptors must have shape [N, D]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target descriptor dimensions differ")
    if source.shape[0] == 0 or target.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        empty_value = torch.empty(0, dtype=source.dtype, device=source.device)
        return MatchResult(empty, empty, empty_value, empty_value, 0)

    source = F.normalize(source, dim=1, eps=1e-12)
    target = F.normalize(target, dim=1, eps=1e-12)
    similarity = source @ target.T
    source_k = min(2, similarity.shape[1])
    target_k = min(2, similarity.shape[0])
    source_values, source_neighbors = similarity.topk(source_k, dim=1)
    target_values, target_neighbors = similarity.topk(target_k, dim=0)

    source_to_target = source_neighbors[:, 0]
    source_ids = torch.arange(len(source), device=source.device)
    mutual = target_neighbors[0, source_to_target] == source_ids
    source_ids = source_ids[mutual]
    target_ids = source_to_target[mutual]
    best_similarity = source_values[source_ids, 0]

    source_margin = (
        source_values[source_ids, 0] - source_values[source_ids, 1]
        if source_k == 2 else torch.full_like(best_similarity, float("inf"))
    )
    target_margin = (
        target_values[0, target_ids] - target_values[1, target_ids]
        if target_k == 2 else torch.full_like(best_similarity, float("inf"))
    )
    margins = torch.minimum(source_margin, target_margin)
    keep = best_similarity >= float(similarity_threshold)
    if float(ambiguity_margin) > 0:
        keep &= margins >= float(ambiguity_margin)
    return MatchResult(
        source_indices=source_ids[keep],
        target_indices=target_ids[keep],
        similarity=best_similarity[keep],
        ambiguity_margin=margins[keep],
        num_mutual=int(mutual.sum().item()),
    )
