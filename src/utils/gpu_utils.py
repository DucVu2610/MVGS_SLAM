"""GPU placement helpers for multi-agent execution."""

from __future__ import annotations

from typing import Dict, Iterable


def build_agent_gpu_map(
        agent_ids: Iterable[int], gpu_count: int) -> Dict[int, int]:
    """Return the CUDA device assigned to each configured agent.

    GPU 0 remains dedicated to the server when there are at least
    ``num_agents + 1`` devices.  When there is exactly one GPU per agent, the
    first agent shares GPU 0 with the server.  Fewer devices cannot satisfy the
    multi-GPU execution contract and are rejected before workers are started.
    """
    normalized_ids = [int(agent_id) for agent_id in agent_ids]
    if not normalized_ids:
        raise ValueError("At least one agent is required for GPU assignment.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Agent IDs must be unique for GPU assignment.")

    gpu_count = int(gpu_count)
    num_agents = len(normalized_ids)
    if gpu_count < num_agents:
        raise RuntimeError(
            "Insufficient CUDA devices for multi-GPU execution: "
            f"found {gpu_count}, but {num_agents} agents require at least "
            f"{num_agents}. Disable multi_gpu or provide more GPUs."
        )

    first_agent_gpu = 1 if gpu_count >= num_agents + 1 else 0
    return {
        agent_id: first_agent_gpu + agent_index
        for agent_index, agent_id in enumerate(normalized_ids)
    }
