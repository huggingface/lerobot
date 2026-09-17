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

"""Tests for SARM's typed progress capability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.rewards.sarm import SARMConfig, SARMPrediction, SARMRewardModel


class _FakeStageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, img_seq, lang_emb, state, lengths, *, scheme):  # noqa: ARG002
        batch_size, _, seq_len, _ = img_seq.shape
        num_classes = 2 if scheme == "sparse" else 3
        logits = torch.zeros(batch_size, seq_len, num_classes, device=img_seq.device)
        logits[..., 0] = 2.0
        return logits


class _FakeSubtaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, img_seq, lang_emb, state, lengths, stage_prior, *, scheme):  # noqa: ARG002
        batch_size, _, seq_len, _ = img_seq.shape
        return torch.linspace(0.1, 0.9, seq_len, device=img_seq.device).expand(batch_size, -1)


def _make_model(*, annotation_mode: str = "dual") -> SARMRewardModel:
    config = SARMConfig(
        annotation_mode=annotation_mode,
        n_obs_steps=2,
        max_rewind_steps=1,
        image_dim=4,
        text_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        max_state_dim=3,
        dropout=0.0,
        num_sparse_stages=2,
        sparse_subtask_names=["approach", "grasp"],
        sparse_temporal_proportions=[0.4, 0.6],
        num_dense_stages=3,
        dense_subtask_names=["approach", "grasp", "place"],
        dense_temporal_proportions=[0.2, 0.3, 0.5],
        device="cpu",
    )
    model = SARMRewardModel(config)
    model.stage_model = _FakeStageModel()
    model.subtask_model = _FakeSubtaskModel()
    return model


def _make_batch(batch_size: int = 2, seq_len: int = 4) -> dict[str, torch.Tensor]:
    return {
        "text_features": torch.zeros(batch_size, 4),
        "video_features": torch.zeros(batch_size, seq_len, 4),
        "state_features": torch.zeros(batch_size, seq_len, 3),
        "lengths": torch.full((batch_size,), seq_len, dtype=torch.int32),
    }


@pytest.mark.parametrize(("head_mode", "num_stages"), [("sparse", 2), ("dense", 3)])
def test_sarm_predict_progress_returns_typed_frame_signals(head_mode: str, num_stages: int):
    model = _make_model()
    batch = _make_batch()

    prediction = model.predict_progress(batch, head_mode=head_mode)

    assert isinstance(prediction, SARMPrediction)
    assert prediction.progress.shape == (2, 4)
    assert prediction.stage_probabilities.shape == (2, 4, num_stages)
    assert prediction.stage_confidence.shape == (2, 4)
    assert prediction.valid_mask.shape == (2, 4)
    assert prediction.progress.dtype == torch.float32
    assert prediction.stage_probabilities.dtype == torch.float32
    assert prediction.stage_confidence.dtype == torch.float32
    assert prediction.valid_mask.dtype == torch.bool
    assert prediction.progress.device == model.device
    assert prediction.valid_mask.device == model.device
    assert prediction.valid_mask.all()
    assert torch.all((prediction.progress >= 0.0) & (prediction.progress <= 1.0))


def test_sarm_predict_progress_marks_padded_frames_invalid():
    model = _make_model()
    batch = _make_batch()
    batch["lengths"] = torch.tensor([2, 4], dtype=torch.int32)

    prediction = model.predict_progress(batch)

    assert torch.equal(
        prediction.valid_mask,
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )


@pytest.mark.parametrize(
    "lengths",
    [
        torch.tensor([4]),
        torch.tensor([0, 4]),
        torch.tensor([5, 4]),
    ],
)
def test_sarm_predict_progress_rejects_invalid_lengths(lengths: torch.Tensor):
    model = _make_model()
    batch = _make_batch()
    batch["lengths"] = lengths

    with pytest.raises(ValueError, match="lengths"):
        model.predict_progress(batch)


def test_sarm_calculate_rewards_preserves_numpy_compatibility():
    model = _make_model()
    batch = _make_batch()
    prediction = model.predict_progress(batch, head_mode="sparse")

    progress, stage_probabilities, confidence = model.calculate_rewards(
        text_embeddings=batch["text_features"],
        video_embeddings=batch["video_features"],
        state_features=batch["state_features"],
        lengths=batch["lengths"],
        return_all_frames=True,
        return_stages=True,
        return_confidence=True,
        head_mode="sparse",
    )

    np.testing.assert_allclose(progress, prediction.progress.numpy())
    np.testing.assert_allclose(stage_probabilities, prediction.stage_probabilities.numpy())
    np.testing.assert_allclose(confidence, prediction.stage_confidence.numpy())


def test_sarm_calculate_rewards_preserves_unbatched_default_frame_behavior():
    model = _make_model()
    batch = _make_batch(batch_size=1)
    prediction = model.predict_progress(
        {
            "text_features": batch["text_features"][0],
            "video_features": batch["video_features"][0],
            "state_features": batch["state_features"][0],
        }
    )

    progress = model.calculate_rewards(
        text_embeddings=batch["text_features"][0],
        video_embeddings=batch["video_features"][0],
        state_features=batch["state_features"][0],
        head_mode=None,
    )

    assert isinstance(progress, np.floating)
    assert progress == pytest.approx(prediction.progress[0, model.config.n_obs_steps].item())


def test_sarm_predict_progress_uses_parameter_device_without_changing_mode():
    model = _make_model()
    model.train()
    model.device = torch.device("meta")
    batch = _make_batch()
    batch.pop("state_features")
    batch.pop("lengths")

    prediction = model.predict_progress(batch)

    assert prediction.progress.device.type == "cpu"
    assert model.training is True


def test_sarm_prediction_can_feed_a_differentiable_consumer():
    model = _make_model()
    prediction = model.predict_progress(_make_batch())
    consumer = torch.nn.Linear(prediction.progress.shape[-1], 1)

    loss = consumer(prediction.progress).sum()
    loss.backward()

    assert prediction.progress.requires_grad is False
    assert consumer.weight.grad is not None
    assert consumer.bias.grad is not None


def test_sarm_predict_progress_requires_processor_features():
    model = _make_model()

    with pytest.raises(KeyError, match="text_features"):
        model.predict_progress({"video_features": torch.zeros(1, 4, 4)})
    with pytest.raises(KeyError, match="video_features"):
        model.predict_progress({"text_features": torch.zeros(1, 4)})


def test_sarm_predict_progress_rejects_unavailable_dense_head():
    model = _make_model(annotation_mode="single_stage")

    with pytest.raises(ValueError, match="dense predictions require"):
        model.predict_progress(_make_batch(), head_mode="dense")
