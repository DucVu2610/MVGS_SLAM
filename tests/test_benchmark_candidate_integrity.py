import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.utils.benchmark_utils import (
    create_fresh_registration,
    load_candidate_identities,
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


if __name__ == "__main__":
    unittest.main()
