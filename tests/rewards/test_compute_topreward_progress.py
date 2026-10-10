# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Characterization tests for TOPReward's legacy progress conversion."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.topreward import compute_rabc_weights
from lerobot.rewards.topreward.compute_rabc_weights import (
    _apply_legacy_success_threshold,
    normalize_rewards,
)


def test_legacy_success_threshold_preserves_raw_log_probabilities_when_disabled():
    values = torch.tensor([-2.0, -0.5])

    result = _apply_legacy_success_threshold(values, float("-inf"))

    assert result is values


def test_legacy_success_threshold_preserves_binary_script_behavior():
    values = torch.tensor([-2.0, -0.5, 0.0])

    result = _apply_legacy_success_threshold(values, -1.0)

    assert torch.equal(result, torch.tensor([0.0, 1.0, 1.0]))


def test_normalize_rewards_preserves_constant_and_varying_episode_behavior():
    np.testing.assert_array_equal(normalize_rewards([0.0, 0.0]), np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(
        normalize_rewards([-2.0, -1.0, 0.0]),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_compute_topreward_progress_preserves_legacy_parquet_artifact(monkeypatch, tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    image_key = "observation.images.top"

    class FakeConfig:
        def __init__(self, **kwargs) -> None:
            self.device = kwargs["device"]
            self.vlm_name = kwargs.get("vlm_name", "fake/vlm")
            self.image_key = image_key
            self.task_key = "task"
            self.default_task = None
            self.fps = 2.0
            self.prompt_prefix = "prefix"
            self.prompt_suffix_template = "{instruction}"
            self.add_chat_template = False
            self.max_input_length = 128
            self.success_threshold = float("-inf")

    class FakeModel:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config

        def to(self, _device: str):
            return self

        def eval(self):
            return self

        def compute_log_probability(self, batch):
            return batch["prefix_length"].float()

    class FakeEncoder:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, transition):
            frames = transition[TransitionKey.OBSERVATION][image_key]
            return {
                TransitionKey.OBSERVATION: {
                    "prefix_length": torch.tensor([frames.shape[1]], dtype=torch.float32),
                }
            }

    class FakeDataset:
        def __init__(self, _repo_id: str, *, download_videos: bool) -> None:  # noqa: ARG002
            self.root = tmp_path
            self.num_episodes = 1
            self.num_frames = 3
            self.meta = SimpleNamespace(
                episodes=[{"dataset_from_index": 0, "dataset_to_index": self.num_frames}]
            )

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), float(index)),
                "task": "pick up the cube",
            }

    monkeypatch.setattr(compute_rabc_weights, "TOPRewardConfig", FakeConfig)
    monkeypatch.setattr(compute_rabc_weights, "TOPRewardModel", FakeModel)
    monkeypatch.setattr(compute_rabc_weights, "TOPRewardEncoderProcessorStep", FakeEncoder)
    monkeypatch.setattr(compute_rabc_weights, "LeRobotDataset", FakeDataset)

    output_path = compute_rabc_weights.compute_topreward_progress(
        dataset_repo_id="local/test",
        output_path=str(tmp_path / "topreward_progress.parquet"),
        device="cpu",
    )

    table = pq.read_table(output_path)
    assert table.schema.names == ["index", "episode_index", "frame_index", "progress_sparse"]
    assert table.schema.field("index").type == pa.int64()
    assert table.schema.field("episode_index").type == pa.int64()
    assert table.schema.field("frame_index").type == pa.int64()
    assert table.schema.field("progress_sparse").type == pa.float32()
    assert table.schema.metadata == {b"vlm_name": b"fake/vlm"}
    assert table.column("progress_sparse").to_pylist() == [0.0, 0.5, 1.0]
