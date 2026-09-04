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
import pytest
import torch

from lerobot.rewards.rynnvalue.modeling_rynnvalue import RynnValuePrediction
from lerobot.rewards.rynnvalue.scoring_rynnvalue import (
    IS_INFERENCE_FRAME_SIGNAL,
    POTENTIAL_SIGNAL,
    PROGRESS_SIGNAL,
    REMAINING_TIME_SIGNAL,
    RynnValueFrameScorer,
    interpolate_anchor_values,
    make_rynnvalue_frame_scorer,
    remaining_time_to_progress,
    score_rynnvalue_dataset,
    select_anchor_indices,
    select_prefix_indices,
)


def test_anchor_selection_uses_inference_fps_and_keeps_boundaries():
    assert select_anchor_indices(61, dataset_fps=30, inference_fps=1).tolist() == [0, 30, 60]
    assert select_anchor_indices(1, dataset_fps=30, inference_fps=1).tolist() == [0]

    with pytest.raises(ValueError, match="inference_fps"):
        select_anchor_indices(10, dataset_fps=30, inference_fps=31)


def test_prefix_selection_is_causal_and_uniform():
    assert select_prefix_indices(10, max_frames=4).tolist() == [0, 3, 7, 10]
    assert select_prefix_indices(10, max_frames=1).tolist() == [10]
    assert select_prefix_indices(2, max_frames=8).tolist() == [0, 1, 2]
    assert select_prefix_indices(2, max_frames=None).tolist() == [0, 1, 2]


def test_anchor_predictions_are_interpolated_without_changing_anchors():
    anchors = np.asarray([0, 2, 4], dtype=np.int64)
    values = np.asarray([4.0, 2.0, 0.0], dtype=np.float32)
    dense = interpolate_anchor_values(5, anchors, values)

    np.testing.assert_allclose(dense, [4.0, 3.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(dense[anchors], values)


def test_horizon_normalization_is_bounded_and_not_episode_relative():
    remaining_time = np.asarray([12.0, 10.0, 5.0, 0.0, -1.0], dtype=np.float32)
    progress = remaining_time_to_progress(remaining_time, horizon_s=10.0)

    np.testing.assert_allclose(progress, [0.0, 0.0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        remaining_time_to_progress(remaining_time, horizon_s=0)


def test_make_rynnvalue_frame_scorer_preserves_adapter_selected_prefixes(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEncoder:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    import lerobot.rewards.rynnvalue.scoring_rynnvalue as scoring_rynnvalue

    monkeypatch.setattr(scoring_rynnvalue, "RynnValueEncoderProcessorStep", FakeEncoder)
    config = SimpleNamespace(
        pretrained_path="converted/rynnvalue",
        pretrained_revision="checkpoint-commit",
        model_id="original/rynnvalue",
        model_revision="original-commit",
        image_key="observation.images.top",
        task_key="task",
        default_task="perform the task",
        robot_description=None,
        camera_description=None,
        use_meta=False,
    )

    scorer = make_rynnvalue_frame_scorer(
        SimpleNamespace(),
        config,
        dataset_fps=30.0,
        max_frames=6,
    )

    assert captured["model_id"] == "converted/rynnvalue"
    assert captured["model_revision"] == "checkpoint-commit"
    assert captured["max_frames"] is None
    assert scorer.max_frames == 6


def test_rynnvalue_frame_scorer_emits_dense_native_and_derived_signals():
    image_key = "observation.images.top"

    class FakeDataset:
        absolute_to_relative_idx = None
        fps = 2.0

        def __init__(self) -> None:
            self.meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 5}])

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), index, dtype=torch.uint8),
                "task": "pick up the cube",
            }

    class FakeEncoder:
        def encode_samples(self, samples):
            final_frame_ids = [images[-1].getpixel((0, 0))[0] for images, _ in samples]
            return {"input_ids": torch.tensor(final_frame_ids).unsqueeze(1)}

    class FakeModel:
        def predict_remaining_time(self, batch):
            final_frame_ids = batch["observation.rynnvalue.input_ids"].squeeze(1)
            return RynnValuePrediction(remaining_time_s=4.0 - final_frame_ids.float())

    scorer = RynnValueFrameScorer(
        model=FakeModel(),
        encoder=FakeEncoder(),
        image_key=image_key,
        dataset_fps=2.0,
        batch_size=2,
        inference_fps=1.0,
        max_frames=3,
        horizon_s=4.0,
        use_meta=False,
    )

    result = scorer(FakeDataset(), episode_index=0)

    np.testing.assert_array_equal(result.frame_indices, np.arange(5))
    np.testing.assert_allclose(result.signals[REMAINING_TIME_SIGNAL], [4, 3, 2, 1, 0])
    np.testing.assert_allclose(result.signals[POTENTIAL_SIGNAL], [-4, -3, -2, -1, 0])
    np.testing.assert_array_equal(result.signals[IS_INFERENCE_FRAME_SIGNAL], [True, False, True, False, True])
    np.testing.assert_allclose(result.signals[PROGRESS_SIGNAL], [0, 0.25, 0.5, 0.75, 1])
    assert result.descriptors[REMAINING_TIME_SIGNAL].unit == "s"
    assert result.descriptors[PROGRESS_SIGNAL].bounds == (0.0, 1.0)
    assert scorer.options["inference_fps"] == 1.0
    assert scorer.options["horizon_s"] == 4.0

    scorer.horizon_s = None
    without_progress = scorer(FakeDataset(), episode_index=0)
    assert PROGRESS_SIGNAL not in without_progress.signals
    assert PROGRESS_SIGNAL not in without_progress.descriptors


def test_score_rynnvalue_dataset_builds_reproducible_provenance(monkeypatch, tmp_path):
    import lerobot.rewards.rynnvalue.scoring_rynnvalue as scoring_rynnvalue

    config = SimpleNamespace(
        type="rynnvalue",
        pretrained_path="model/default",
        pretrained_revision="config-revision",
    )
    model = SimpleNamespace(config=config)
    dataset = SimpleNamespace(repo_id="user/dataset", revision="dataset-commit", fps=30.0)
    fake_scorer = SimpleNamespace(options={"batch_size": 2, "inference_fps": 1.0})
    expected_summary = SimpleNamespace(output_path=tmp_path / "signals.parquet")
    captured: dict[str, object] = {}

    def fake_make_scorer(received_model, received_config, **kwargs):
        captured["make_scorer"] = (received_model, received_config, kwargs)
        return fake_scorer

    def fake_score_dataset(received_dataset, scorer, **kwargs):
        captured["score_dataset"] = (received_dataset, scorer, kwargs)
        return expected_summary

    monkeypatch.setattr(scoring_rynnvalue, "make_rynnvalue_frame_scorer", fake_make_scorer)
    monkeypatch.setattr(scoring_rynnvalue, "score_dataset", fake_score_dataset)

    output_path = tmp_path / "signals.parquet"
    summary = score_rynnvalue_dataset(
        dataset,
        model,
        output_path=output_path,
        model_id="user/rynnvalue",
        model_revision="model-commit",
        episode_indices=[2, 1],
        batch_size=2,
        inference_fps=1.0,
        max_frames=8,
        horizon_s=12.0,
    )

    assert summary is expected_summary
    assert captured["make_scorer"] == (
        model,
        config,
        {
            "dataset_fps": 30.0,
            "batch_size": 2,
            "inference_fps": 1.0,
            "max_frames": 8,
            "horizon_s": 12.0,
        },
    )
    assert captured["score_dataset"] == (
        dataset,
        fake_scorer,
        {
            "output_path": output_path,
            "provenance": {
                "lerobot_version": __import__("lerobot").__version__,
                "dataset": {"repo_id": "user/dataset", "revision": "dataset-commit"},
                "model": {"type": "rynnvalue", "id": "user/rynnvalue", "revision": "model-commit"},
                "adapter": {
                    "id": "lerobot.rynnvalue.causal_prefix",
                    "version": 1,
                    "options": fake_scorer.options,
                },
            },
            "episode_indices": [2, 1],
            "resume": True,
        },
    )
