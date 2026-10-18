"""GPU placement helpers for multi-agent execution."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping
import warnings


def build_agent_gpu_map(
        agent_ids: Iterable[int], gpu_count: int, *, server_gpu: int = 0,
        explicit_map: Mapping[int, int] | None = None,
        strict: bool = False) -> Dict[int, int]:
    """Return the CUDA device assigned to each configured agent.

    GPU 0 remains dedicated to the server when there are at least
    ``num_agents + 1`` devices. Otherwise agents are assigned round-robin over
    valid CUDA IDs. ``strict=True`` retains the old one-GPU-per-agent check.
    """
    normalized_ids = [int(agent_id) for agent_id in agent_ids]
    if not normalized_ids:
        raise ValueError("At least one agent is required for GPU assignment.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Agent IDs must be unique for GPU assignment.")

    gpu_count = int(gpu_count)
    num_agents = len(normalized_ids)
    if gpu_count < 1:
        raise RuntimeError(
            "multi_gpu requires at least one visible CUDA device, but none "
            "were reported by torch.cuda.device_count().")
    server_gpu = int(server_gpu)
    if not 0 <= server_gpu < gpu_count:
        raise ValueError(
            f"runtime.server_gpu={server_gpu} is outside [0, {gpu_count - 1}]")
    if strict and gpu_count < num_agents:
        raise RuntimeError(
            "Insufficient CUDA devices for multi-GPU execution: "
            f"found {gpu_count}, but {num_agents} agents require at least "
            f"{num_agents} in strict mode. Disable runtime.strict_gpu_mapping "
            "or provide more GPUs."
        )

    if explicit_map is not None:
        normalized_map = {int(key): int(value) for key, value in explicit_map.items()}
        missing = [agent_id for agent_id in normalized_ids if agent_id not in normalized_map]
        extra = sorted(set(normalized_map) - set(normalized_ids))
        if missing or extra:
            raise ValueError(
                f"runtime.agent_gpu_map must match data.agent_ids; "
                f"missing={missing}, extra={extra}")
        invalid = {
            agent_id: device for agent_id, device in normalized_map.items()
            if not 0 <= device < gpu_count
        }
        if invalid:
            raise ValueError(
                f"runtime.agent_gpu_map contains unavailable CUDA IDs: {invalid}; "
                f"visible device count={gpu_count}")
        result = {agent_id: normalized_map[agent_id] for agent_id in normalized_ids}
    elif gpu_count >= num_agents + 1:
        available = [device for device in range(gpu_count) if device != server_gpu]
        result = {
            agent_id: available[index]
            for index, agent_id in enumerate(normalized_ids)
        }
    else:
        result = {
            agent_id: index % gpu_count
            for index, agent_id in enumerate(normalized_ids)
        }

    assignments = list(result.values())
    shared_agents = len(set(assignments)) < len(assignments)
    shares_server = server_gpu in assignments
    if shared_agents or shares_server:
        warnings.warn(
            "CUDA devices are shared by the server and/or multiple agents: "
            f"server={server_gpu}, agents={result}",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
