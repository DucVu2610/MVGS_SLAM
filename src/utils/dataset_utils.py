"""Dataset path resolution shared by SLAM and offline tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_agent_input(
    data_root,
    agent_id: int,
    configured_agent_ids: Iterable[int],
) -> Path:
    """Resolve an agent folder without assuming IDs are positional.

    Resolution order is part of the runtime protocol: ``agent_<id>``, then
    ``<id>``, then the configured ID's position in ``data.agent_ids``.
    """
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    normalized_id = int(agent_id)
    configured = [int(value) for value in configured_agent_ids]
    if len(configured) != len(set(configured)):
        raise ValueError("data.agent_ids must contain unique IDs")
    if normalized_id not in configured:
        raise KeyError(
            f"agent {normalized_id} is not listed in data.agent_ids={configured}")

    for name in (f"agent_{normalized_id}", str(normalized_id)):
        candidate = root / name
        if candidate.is_dir():
            return candidate

    children = sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    position = configured.index(normalized_id)
    if position >= len(children):
        names = [path.name for path in children]
        raise FileNotFoundError(
            f"cannot resolve agent {normalized_id} below {root}: tried "
            f"agent_{normalized_id}, {normalized_id}, and configured position "
            f"{position}; available directories={names}"
        )
    return children[position]
