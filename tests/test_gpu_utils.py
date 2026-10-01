import unittest

from src.utils.gpu_utils import build_agent_gpu_map


class AgentGpuMapTest(unittest.TestCase):

    def test_server_keeps_gpu_zero_when_spare_gpu_exists(self):
        self.assertEqual(build_agent_gpu_map([0, 1], 3), {0: 1, 1: 2})

    def test_server_shares_gpu_zero_when_gpu_count_matches_agents(self):
        self.assertEqual(build_agent_gpu_map([0, 1], 2), {0: 0, 1: 1})

    def test_insufficient_gpu_count_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "Insufficient CUDA devices"):
            build_agent_gpu_map([0, 1], 1)


if __name__ == "__main__":
    unittest.main()
