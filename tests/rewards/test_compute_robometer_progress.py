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

"""Tests for RoboMeter's RA-BC progress script."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.robometer import compute_rabc_weights
from lerobot.rewards.robometer.modeling_robometer import ROBOMETER_FEATURE_PREFIX, RobometerPrediction
from lerobot.rewards.robometer.processor_robometer import _video_to_numpy


@pytest.mark.parametrize(
    ("num_frames", "num_subsampled_frames", "expected"),
    [
        (4, 1, [[0], [0], [0], [0]]),
        (3, 4, [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 2]]),
        (
            6,
            4,
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 3, 4], [0, 2, 3, 5]],
        ),
    ],
)
def test_build_subsample_indices(num_frames: int, num_subsampled_frames: int, expected: list) -> None:
    indices = compute_rabc_weights._build_subsample_indices(num_frames, num_subsampled_frames)

    assert [index.tolist() for index in indices] == expected


def test_video_to_numpy_without_max_frames_keeps_every_frame() -> None:
    selected_frames = torch.arange(4 * 3 * 2 * 2, dtype=torch.uint8).reshape(4, 3, 2, 2)

    capped = _video_to_numpy(selected_frames, max_frames=4)
    uncapped = _video_to_numpy(selected_frames, max_frames=None)

    np.testing.assert_array_equal(uncapped, capped)


@pytest.mark.parametrize("num_subsampled_frames", [0, -1])
def test_compute_progress_rejects_invalid_subsample_count_before_loading(
    num_subsampled_frames: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"num_subsampled_frames must be >= 1, got {num_subsampled_frames}",
    ):
        compute_rabc_weights.compute_robometer_progress(
            dataset_repo_id="unused/dataset",
            reward_model_path="unused/model",
            num_subsampled_frames=num_subsampled_frames,
        )


def test_scoring_encoder_preserves_already_selected_frames() -> None:
    encoder = object.__new__(compute_rabc_weights.RobometerEncoderProcessorStep)
    encoder.image_key = "observation.images.top"
    encoder.task_key = "task"
    encoder.default_task = None
    encoder.max_frames = None

    captured_samples: list[tuple[np.ndarray, str]] = []

    def encode_samples(samples: list[tuple[np.ndarray, str]]) -> dict[str, torch.Tensor]:
        captured_samples.extend(samples)
        return {"input_ids": torch.zeros(len(samples), 1, dtype=torch.long)}

    encoder.encode_samples = encode_samples
    batch_size = 2
    num_subsampled_frames = 4
    frames = torch.zeros(batch_size, num_subsampled_frames, 3, 2, 2)
    transition = {
        TransitionKey.OBSERVATION: {encoder.image_key: frames},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["task a", "task b"]},
    }

    encoder(transition)

    assert [sample_frames.shape[0] for sample_frames, _ in captured_samples] == [
        num_subsampled_frames,
        num_subsampled_frames,
    ]
    assert [task for _, task in captured_samples] == ["task a", "task b"]


@pytest.mark.dataset
def test_compute_robometer_progress_writes_progress_parquet(monkeypatch, tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    image_key = "observation.images.top"
    encoder_kwargs: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, pretrained_path: str, device: str) -> None:
            self.pretrained_path = pretrained_path
            self.device = device
            self.image_key = image_key
            self.task_key = "task"
            self.default_task = None
            self.base_model_id = "fake/backbone"
            self.use_multi_image = True
            self.use_per_frame_progress_token = True

    class FakeModel:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config

        @classmethod
        def from_pretrained(cls, _path: str, *, config: FakeConfig):
            return cls(config)

        def to(self, _device: str):
            return self

        def eval(self):
            return self

        def predict_progress(self, batch):
            progress = batch[f"{ROBOMETER_FEATURE_PREFIX}input_ids"].float()
            return RobometerPrediction(progress=progress, success_probability=torch.zeros_like(progress))

    class FakeEncoder:
        def __init__(self, **kwargs) -> None:
            encoder_kwargs.update(kwargs)

        def __call__(self, transition):
            frames = transition[TransitionKey.OBSERVATION][image_key]
            frame_ids = frames[:, :, 0, 0, 0]
            return {
                TransitionKey.OBSERVATION: {
                    f"{ROBOMETER_FEATURE_PREFIX}input_ids": frame_ids,
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

    monkeypatch.setattr(compute_rabc_weights, "RobometerConfig", FakeConfig)
    monkeypatch.setattr(compute_rabc_weights, "RobometerRewardModel", FakeModel)
    monkeypatch.setattr(compute_rabc_weights, "RobometerEncoderProcessorStep", FakeEncoder)
    monkeypatch.setattr(compute_rabc_weights, "LeRobotDataset", FakeDataset)

    output_path = compute_rabc_weights.compute_robometer_progress(
        dataset_repo_id="local/test",
        reward_model_path="local/robometer",
        output_path=str(tmp_path / "robometer_progress.parquet"),
        device="cpu",
        batch_size=2,
        num_subsampled_frames=2,
    )

    table = pq.read_table(output_path)
    assert table.schema.names == ["index", "episode_index", "frame_index", "progress_sparse"]
    assert table.schema.field("index").type == pa.int64()
    assert table.schema.field("episode_index").type == pa.int64()
    assert table.schema.field("frame_index").type == pa.int64()
    assert table.schema.field("progress_sparse").type == pa.float32()
    assert table.schema.metadata == {b"reward_model_path": b"local/robometer"}
    assert table.column("progress_sparse").to_pylist() == [0.0, 1.0, 2.0]
    assert encoder_kwargs["max_frames"] is None  # the sub-samples are already the right size
