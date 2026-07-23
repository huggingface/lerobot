from types import SimpleNamespace

import torch

from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset


def test_compute_quantile_stats_skips_language_features():
    class FakeDataset:
        num_episodes = 1
        features = {
            "action": {"dtype": "float32"},
            "observation.language": {"dtype": "language"},
        }
        meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 2}])
        rows = [
            {
                "action": torch.tensor([0.0]),
                "observation.language": [{"role": "user", "content": "pick"}],
            },
            {
                "action": torch.tensor([1.0]),
                "observation.language": [{"role": "assistant", "content": "done"}],
            },
        ]

        def __getitem__(self, index):
            return self.rows[index]

    stats = compute_quantile_stats_for_dataset(FakeDataset())

    assert set(stats) == {"action"}
