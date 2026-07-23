# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset


class FakeHFDataset:
    """Minimal stand-in exposing the column slicing used by the augment script."""

    def __init__(self, columns: dict[str, list]):
        self._columns = columns

    def select_columns(self, keys):
        return FakeHFDataset({key: self._columns[key] for key in keys})

    def __getitem__(self, index):
        return {key: values[index] for key, values in self._columns.items()}


def test_compute_quantile_stats_skips_language_features():
    class FakeDataset:
        num_episodes = 1
        features = {
            "action": {"dtype": "float32"},
            "observation.language": {"dtype": "language"},
        }
        meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 2}])
        hf_dataset = FakeHFDataset(
            {
                "action": [[0.0], [1.0]],
                "observation.language": [
                    [{"role": "user", "content": "pick"}],
                    [{"role": "assistant", "content": "done"}],
                ],
            }
        )

    stats = compute_quantile_stats_for_dataset(FakeDataset())

    assert set(stats) == {"action"}
