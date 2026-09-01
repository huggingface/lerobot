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

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.robometer.modeling_robometer import RobometerPrediction
from lerobot.rewards.robometer.scoring_robometer import (
    PROGRESS_SIGNAL,
    SUCCESS_PROBABILITY_SIGNAL,
    RobometerFrameScorer,
    build_subsample_indices,
    make_robometer_scoring_encoder,
)
from lerobot.rewards.scoring import read_frame_signals, score_dataset


def test_build_subsample_indices_pins_characterized_legacy_rule():
    actual = build_subsample_indices(6, 4)

    assert [indices.tolist() for indices in actual] == [
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3, 4],
        [0, 2, 3, 5],
    ]


def test_build_subsample_indices_keeps_prefix_endpoints():
    for current_frame, indices in enumerate(build_subsample_indices(7, 4)):
        assert indices.shape == (4,)
        assert indices.dtype == np.int64
        assert indices[0] == 0
        assert indices[-1] == current_frame


def test_make_robometer_scoring_encoder_preserves_presampled_frames(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEncoder:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    import lerobot.rewards.robometer.scoring_robometer as scoring_robometer

    monkeypatch.setattr(scoring_robometer, "RobometerEncoderProcessorStep", FakeEncoder)
    config = SimpleNamespace(
        base_model_id="fake/backbone",
        image_key="observation.images.top",
        task_key="task",
        default_task="do the task",
        use_multi_image=True,
        use_per_frame_progress_token=True,
    )

    encoder = make_robometer_scoring_encoder(config)

    assert isinstance(encoder, FakeEncoder)
    assert captured["max_frames"] is None
    assert captured["image_key"] == config.image_key


def test_robometer_frame_scorer_returns_dense_namespaced_signals():
    image_key = "observation.images.top"

    class FakeDataset:
        absolute_to_relative_idx = None

        def __init__(self) -> None:
            self.meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}])

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), float(index)),
                "task": "pick up the cube",
            }

    class FakeEncoder:
        def __call__(self, transition):
            frames = transition[TransitionKey.OBSERVATION][image_key]
            frame_ids = frames[:, :, 0, 0, 0]
            return {TransitionKey.OBSERVATION: {"encoded": frame_ids}}

    class FakeModel:
        def predict_progress(self, batch):
            progress = batch["encoded"].float() / 2.0
            return RobometerPrediction(
                progress=progress,
                success_probability=progress / 2.0,
            )

    scorer = RobometerFrameScorer(
        model=FakeModel(),
        encoder=FakeEncoder(),
        image_key=image_key,
        batch_size=2,
        num_subsampled_frames=2,
    )

    result = scorer(FakeDataset(), episode_index=0)

    np.testing.assert_array_equal(result.frame_indices, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(result.signals[PROGRESS_SIGNAL], np.asarray([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(result.signals[SUCCESS_PROBABILITY_SIGNAL], np.asarray([0.0, 0.25, 0.5]))
    assert set(result.descriptors) == {PROGRESS_SIGNAL, SUCCESS_PROBABILITY_SIGNAL}
    assert result.descriptors[PROGRESS_SIGNAL].bounds == (0.0, 1.0)
    assert scorer.options == {
        "batch_size": 2,
        "base_model_id": None,
        "default_task": "perform the task",
        "image_key": image_key,
        "num_subsampled_frames": 2,
        "sampling": "linspace_round_fixed_prefix",
        "task_key": "task",
        "task_resolution": "first_frame_or_default",
        "use_multi_image": None,
        "use_per_frame_progress_token": None,
    }


def test_robometer_frame_scorer_supports_episode_filtered_dataset_views():
    image_key = "observation.images.top"

    class FakeDataset:
        absolute_to_relative_idx = {5: 0, 6: 1}

        def __init__(self) -> None:
            self.meta = SimpleNamespace(
                episodes=[
                    {"dataset_from_index": 0, "dataset_to_index": 5},
                    {"dataset_from_index": 5, "dataset_to_index": 7},
                ]
            )

        def __getitem__(self, relative_index: int):
            return {
                image_key: torch.full((3, 2, 2), float(relative_index + 5)),
                "task": "selected task",
            }

    class FakeEncoder:
        def __call__(self, transition):
            frames = transition[TransitionKey.OBSERVATION][image_key]
            return {
                TransitionKey.OBSERVATION: {
                    "encoded": frames[:, :, 0, 0, 0],
                }
            }

    class FakeModel:
        def predict_progress(self, batch):
            values = batch["encoded"].float() / 10.0
            return RobometerPrediction(progress=values, success_probability=values)

    scorer = RobometerFrameScorer(
        model=FakeModel(),
        encoder=FakeEncoder(),
        image_key=image_key,
        num_subsampled_frames=2,
    )

    result = scorer(FakeDataset(), episode_index=1)

    np.testing.assert_allclose(result.signals[PROGRESS_SIGNAL], [0.5, 0.6])


def test_robometer_vertical_slice_preserves_legacy_progress_values_and_adds_success(tmp_path):
    image_key = "observation.images.top"

    class FakeDataset:
        absolute_to_relative_idx = None
        episodes = None

        def __init__(self) -> None:
            self.meta = SimpleNamespace(
                total_episodes=1,
                episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}],
            )

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), float(index)),
                "task": "pick up the cube",
            }

    class FakeEncoder:
        def __call__(self, transition):
            frames = transition[TransitionKey.OBSERVATION][image_key]
            return {TransitionKey.OBSERVATION: {"encoded": frames[:, :, 0, 0, 0]}}

    class FakeModel:
        def predict_progress(self, batch):
            progress = batch["encoded"].float() / 2.0
            return RobometerPrediction(
                progress=progress,
                success_probability=progress / 2.0,
            )

    scorer = RobometerFrameScorer(
        model=FakeModel(),
        encoder=FakeEncoder(),
        image_key=image_key,
        batch_size=2,
        num_subsampled_frames=2,
    )
    output_path = tmp_path / "robometer.parquet"

    summary = score_dataset(
        FakeDataset(),
        scorer,
        output_path=output_path,
        provenance={"model": "fake/robometer"},
    )

    table = read_frame_signals(summary.artifact_path)
    # The old progress_sparse fixture produced [0.0, 0.5, 1.0] for these
    # processed frames. The new artifact preserves those values exactly.
    assert table[PROGRESS_SIGNAL].to_pylist() == [0.0, 0.5, 1.0]
    assert table[SUCCESS_PROBABILITY_SIGNAL].to_pylist() == [0.0, 0.25, 0.5]
    assert table.schema.field(PROGRESS_SIGNAL).type == pa.float32()
    assert table.schema.field(SUCCESS_PROBABILITY_SIGNAL).type == pa.float32()
    assert table.column_names == [
        "index",
        "episode_index",
        "frame_index",
        PROGRESS_SIGNAL,
        SUCCESS_PROBABILITY_SIGNAL,
    ]
