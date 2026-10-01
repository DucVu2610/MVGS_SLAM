"""Dependency-light helpers for registration benchmark provenance."""

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Callable, List, Tuple


@dataclass(frozen=True)
class RegistrationCandidate:
    """Only the identity of a detected inter-agent registration query."""

    source_agent_id: int
    source_frame_id: int
    target_agent_id: int
    target_frame_id: int


def create_fresh_registration(candidate, registration_factory: Callable):
    """Create a registration object without copying prior results or metrics."""
    return registration_factory(
        candidate.source_agent_id,
        candidate.source_frame_id,
        candidate.target_agent_id,
        candidate.target_frame_id,
    )


def load_candidate_identities(
        run_dir: Path) -> Tuple[List[RegistrationCandidate], Path]:
    """Load inter-agent identities, preferring pre-registration detections."""
    run_dir = Path(run_dir)
    detected_path = run_dir / "detected_loops.pkl"
    candidate_path = (
        detected_path if detected_path.exists() else run_dir / "loops.pkl"
    )
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"No detected_loops.pkl or loops.pkl found in {run_dir}")

    with open(candidate_path, "rb") as input_file:
        saved_loops = pickle.load(input_file)
    candidates = [
        RegistrationCandidate(
            int(loop.source_agent_id),
            int(loop.source_frame_id),
            int(loop.target_agent_id),
            int(loop.target_frame_id),
        )
        for loop in saved_loops
        if loop.source_agent_id != loop.target_agent_id
    ]
    if not candidates:
        raise RuntimeError(f"No inter-agent loops found in {candidate_path}")
    return candidates, candidate_path
