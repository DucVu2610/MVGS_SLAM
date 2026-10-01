import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.utils.benchmark_utils import (
    GAUSSIAN_LANDMARK_DEFAULTS,
    build_gaussian_cli_overrides,
    create_fresh_registration,
    load_candidate_identities,
    resolve_gaussian_registration_options,
)


def registration(source_agent, source_frame, target_agent, target_frame):
    return SimpleNamespace(
        source_agent_id=source_agent,
        source_frame_id=source_frame,
        target_agent_id=target_agent,
        target_frame_id=target_frame,
        transformation=np.eye(4),
        fitness=0.0,
        inlier_rmse=100.0,
    )


def save_pickle(path, value):
    with open(path, "wb") as output_file:
        pickle.dump(value, output_file)


class BenchmarkCandidateIntegrityTest(unittest.TestCase):

    def test_detected_candidates_take_precedence_and_are_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            detected = registration(0, 4, 1, 8)
            detected.transformation = np.full((4, 4), 7.0)
            detected.fitness = 0.99
            intra_agent = registration(0, 1, 0, 2)
            old_registered = registration(0, 40, 1, 80)
            save_pickle(run_dir / "detected_loops.pkl", [detected, intra_agent])
            save_pickle(run_dir / "loops.pkl", [old_registered])

            candidates, candidate_path = load_candidate_identities(run_dir)

            self.assertEqual(candidate_path.name, "detected_loops.pkl")
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertEqual(
                (candidate.source_agent_id, candidate.source_frame_id,
                 candidate.target_agent_id, candidate.target_frame_id),
                (0, 4, 1, 8))
            self.assertFalse(hasattr(candidate, "transformation"))
            self.assertFalse(hasattr(candidate, "fitness"))

    def test_old_runs_fall_back_to_loops_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            saved = registration(0, 3, 1, 6)
            save_pickle(run_dir / "loops.pkl", [saved])

            candidates, candidate_path = load_candidate_identities(run_dir)

            self.assertEqual(candidate_path.name, "loops.pkl")
            self.assertEqual(len(candidates), 1)

    def test_fresh_registration_never_copies_previous_metrics(self):
        saved = registration(0, 3, 1, 6)
        saved.transformation = np.full((4, 4), 2.0)
        saved.fitness = 0.8
        saved.inlier_rmse = 0.01

        fresh = create_fresh_registration(saved, registration)

        np.testing.assert_array_equal(fresh.transformation, np.eye(4))
        self.assertEqual(fresh.fitness, 0.0)
        self.assertEqual(fresh.inlier_rmse, 100.0)


class GaussianRegistrationOptionsTest(unittest.TestCase):

    def setUp(self):
        self.config_options = {
            "dino_min_margin": 0.0,
            "dino_depth_window_radius": 2,
            "dino_compatibility_threshold_m": 0.03,
            "dino_min_compatibility_support": 16,
            "gaussian_refit_inliers": True,
        }

    def test_config_wins_over_defaults_and_missing_value_uses_default(self):
        options = resolve_gaussian_registration_options(self.config_options)

        self.assertEqual(options["dino_min_margin"], 0.0)
        self.assertEqual(options["dino_depth_window_radius"], 2)
        self.assertEqual(options["dino_compatibility_threshold_m"], 0.03)
        self.assertEqual(options["dino_min_compatibility_support"], 16)
        self.assertEqual(
            options["gaussian_min_opacity"],
            GAUSSIAN_LANDMARK_DEFAULTS["gaussian_min_opacity"])

    def test_explicit_cli_override_wins_over_config(self):
        options = resolve_gaussian_registration_options(
            self.config_options,
            {"dino_depth_window_radius": 4},
        )

        self.assertEqual(options["dino_depth_window_radius"], 4)

    def test_absent_no_refit_flag_preserves_false_config_value(self):
        args = SimpleNamespace(no_gaussian_refit_inliers=False)
        cli_overrides = build_gaussian_cli_overrides(args)
        options = resolve_gaussian_registration_options(
            {"gaussian_refit_inliers": False}, cli_overrides)

        self.assertNotIn("gaussian_refit_inliers", cli_overrides)
        self.assertFalse(options["gaussian_refit_inliers"])

    def test_explicit_no_refit_flag_overrides_true_config_value(self):
        args = SimpleNamespace(no_gaussian_refit_inliers=True)
        cli_overrides = build_gaussian_cli_overrides(args)
        options = resolve_gaussian_registration_options(
            {"gaussian_refit_inliers": True}, cli_overrides)

        self.assertEqual(cli_overrides, {"gaussian_refit_inliers": False})
        self.assertFalse(options["gaussian_refit_inliers"])

    def test_apart_zero_experiment_options_are_preserved(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/ReplicaMultiagent/apart_0_gaussian_landmark_loop035.yaml"
        )
        with open(config_path, "r") as input_file:
            experiment_config = yaml.safe_load(input_file)
        options = resolve_gaussian_registration_options(
            experiment_config["submap"]["registration_options"])

        expected = {
            "min_correspondences": 8,
            "min_ransac_inliers": 3,
            "ransac_distance_threshold": 0.05,
            "dino_min_similarity": -1.0,
            "dino_min_margin": 0.0,
            "dino_keep_top_fraction": 1.0,
            "dino_depth_window_radius": 2,
            "dino_depth_max_mad_m": 0.05,
            "dino_compatibility_threshold_m": 0.03,
            "dino_min_compatibility_support": 16,
            "gaussian_min_opacity": 0.05,
            "gaussian_depth_tolerance_m": 0.05,
            "gaussian_max_landmarks": 2048,
            "gaussian_refit_inliers": True,
        }
        self.assertEqual(
            {key: options[key] for key in expected}, expected)


if __name__ == "__main__":
    unittest.main()
