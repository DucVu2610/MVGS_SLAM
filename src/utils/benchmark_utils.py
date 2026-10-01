"""Dependency-light helpers for registration benchmark provenance."""

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Callable, List, Tuple


GAUSSIAN_LANDMARK_DEFAULTS = {
    "dino_min_similarity": -1.0,
    "dino_min_margin": -1.0,
    "dino_keep_top_fraction": 1.0,
    "dino_depth_window_radius": 0,
    "dino_depth_max_mad_m": None,
    "dino_compatibility_threshold_m": 0.0,
    "dino_min_compatibility_support": 0,
    "gaussian_min_opacity": 0.05,
    "gaussian_depth_tolerance_m": 0.05,
    "gaussian_max_landmarks": 2048,
    "gaussian_refit_inliers": True,
}


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


def resolve_gaussian_registration_options(
        config_options=None, explicit_overrides=None):
    """Resolve Gaussian benchmark options with defaults < config < CLI."""
    options = dict(GAUSSIAN_LANDMARK_DEFAULTS)
    options.update(config_options or {})
    options.update(explicit_overrides or {})
    return options


def build_gaussian_cli_overrides(args):
    """Return only Gaussian options explicitly supplied on the command line."""
    overrides = {}
    if getattr(args, "no_gaussian_refit_inliers", False):
        overrides["gaussian_refit_inliers"] = False
    for argument_name, option_name in (
            ("gaussian_min_opacity", "gaussian_min_opacity"),
            ("gaussian_depth_tolerance_m", "gaussian_depth_tolerance_m"),
            ("gaussian_max_landmarks", "gaussian_max_landmarks")):
        value = getattr(args, argument_name, None)
        if value is not None:
            overrides[option_name] = value
    return overrides


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
