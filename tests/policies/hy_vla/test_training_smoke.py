#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Hy-VLA action-training smoke tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="Hy-VLA requires the `hy_vla` extra (transformers)")

from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAPolicy
from lerobot.policies.pretrained import PreTrainedPolicy


class _CapturingTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self.received = None

    def __call__(self, tasks, **kwargs):
        self.received = list(tasks)
        batch = len(tasks)
        length = kwargs["max_length"]
        return {
            "input_ids": torch.zeros(batch, length, dtype=torch.long),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        return "decoded"


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(
        self, images, image_masks, tokens, language_masks, state, actions, noise, time, text_labels=None
    ):
        target = actions[..., :20]
        return (self.scale * torch.ones_like(target) - target).square(), None


class _IndexedLossFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, images, image_masks, tokens, language_masks, state, actions, noise, time, text_labels=None
    ):
        losses = torch.arange(actions.shape[-1], device=actions.device, dtype=actions.dtype)
        return losses.expand_as(actions) + self.anchor * 0, None


def _lightweight_policy() -> HyVLAPolicy:
    policy = object.__new__(HyVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = HyVLAConfig(device="cpu")
    policy.language_tokenizer = _CapturingTokenizer()
    policy.model = _FakeFlow()
    policy.reset()
    return policy


def test_exact_raw_task_reaches_tokenizer_before_chat_suffix():
    policy = _lightweight_policy()
    raw = "  keep_under_score\nand whitespace  "
    batch = {"observation.images.top_head": torch.zeros(1, 3, 8, 8), "task": [raw]}
    policy.prepare_language(batch)
    assert policy._last_raw_tasks == (raw,)
    assert policy.language_tokenizer.received == [raw + policy.config.task_suffix]


def test_missing_camera_keeps_its_configured_visual_slot():
    policy = _lightweight_policy()
    policy.config.empty_cameras = 1
    policy.config.resize_imgs_with_padding = (8, 8)
    images, masks = policy.prepare_images(
        {
            "observation.images.top_head": torch.full((1, 3, 8, 8), 0.25),
            "observation.images.hand_right": torch.full((1, 3, 8, 8), 0.75),
        }
    )
    assert len(images) == len(masks) == 3
    torch.testing.assert_close(images[0], torch.full_like(images[0], -0.5))
    torch.testing.assert_close(images[1], torch.full_like(images[1], -1.0))
    torch.testing.assert_close(images[2], torch.full_like(images[2], 0.5))
    assert [mask.item() for mask in masks] == [True, False, True]


def test_processor_tensors_are_cast_to_loaded_model_dtype():
    policy = _lightweight_policy()
    batch = {
        "observation.images.top_head": torch.zeros(1, 3, 8, 8, dtype=torch.float64),
        "observation.images.hand_left": torch.zeros(1, 3, 8, 8, dtype=torch.float64),
        "observation.images.hand_right": torch.zeros(1, 3, 8, 8, dtype=torch.float64),
        "observation.state": torch.zeros(1, 20, dtype=torch.float64),
        "action": torch.zeros(1, 50, 20, dtype=torch.float64),
    }
    images, _ = policy.prepare_images(batch)
    assert images[0].dtype == policy.model.scale.dtype
    assert policy.prepare_state(batch).dtype == policy.model.scale.dtype
    assert policy.prepare_action(batch).dtype == policy.model.scale.dtype


def test_finite_forward_backward_update_and_tiny_overfit():
    policy = _lightweight_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    batch = {
        "observation.images.top_head": torch.zeros(2, 3, 8, 8),
        "observation.images.hand_left": torch.zeros(2, 3, 8, 8),
        "observation.images.hand_right": torch.zeros(2, 3, 8, 8),
        "observation.state": torch.zeros(2, 32),
        "action": torch.zeros(2, 50, 32),
        "task": ["a", "b"],
    }
    initial, _ = policy(batch)
    for _ in range(8):
        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        assert policy.model.scale.grad is not None
        assert torch.isfinite(policy.model.scale.grad)
        assert policy.model.scale.grad.abs() > 0
        optimizer.step()
    final, _ = policy(batch)
    assert torch.isfinite(final)
    assert final < initial


def test_training_loss_uses_action_slot_mask():
    policy = _lightweight_policy()
    policy.model = _IndexedLossFlow()
    batch_size, horizon = 2, policy.config.chunk_size
    batch = {
        **{key: torch.zeros(batch_size, 3, 8, 8) for key in policy.config.image_features},
        "observation.state": torch.zeros(batch_size, 32),
        "action": torch.zeros(batch_size, horizon, 32),
        "action.mask": torch.zeros(batch_size, horizon, 32, dtype=torch.bool),
        "task": ["a", "b"],
    }
    slots = (0, 2, 4, 7, 10, 15, 19)
    batch["action.mask"][..., list(slots)] = True

    loss, _ = policy(batch)
    expected = torch.tensor(slots, dtype=torch.float32).mean()
    torch.testing.assert_close(loss, expected)


def test_mem_history_matches_author_cadence_and_zero_padding():
    policy = _lightweight_policy()
    policy.config = HyVLAConfig(
        device="cpu",
        chunk_size=40,
        n_action_steps=40,
        action_representation="relative_absolute",
        action_decode_mode="blend",
        embodiment="robotwin_dual_arm",
        native_quaternion_order="wxyz",
        use_video_encoder=True,
        img_history_size=3,
        img_history_interval=2,
        execution_horizon=2,
    )
    policy.reset()
    keys = list(policy.config.image_features)
    for value in range(5):
        batch = {key: torch.full((1, 3, 2, 2), value, dtype=torch.float32) for key in keys}
        policy._append_inference_history(batch)

    stacked = policy._with_inference_history(batch)
    for key in keys:
        assert stacked[key].shape == (1, 3, 3, 2, 2)
        torch.testing.assert_close(stacked[key][0, :, 0, 0, 0], torch.tensor([0.0, 2.0, 4.0]))

    policy.reset()
    batch = {key: torch.ones(1, 3, 2, 2) for key in keys}
    policy._append_inference_history(batch)
    stacked = policy._with_inference_history(batch)
    for key in keys:
        torch.testing.assert_close(stacked[key][0, :, 0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))


class _TextLossFlow(nn.Module):
    """Flow stub that also reports a fixed text loss, as co-training does."""

    def __init__(self, text_loss: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.text_loss = torch.tensor(text_loss)
        self.seen_labels = None

    def forward(
        self, images, image_masks, tokens, language_masks, state, actions, noise, time, text_labels=None
    ):
        self.seen_labels = text_labels
        target = actions[..., :20]
        flow = torch.ones_like(target) + self.anchor * 0
        return flow, (self.text_loss if text_labels is not None else None)


def _text_batch(batch_size: int = 1, *, labels: torch.Tensor | None = None) -> dict:
    policy_config = HyVLAConfig(device="cpu")
    horizon = policy_config.chunk_size
    batch = {
        **{key: torch.zeros(batch_size, 3, 8, 8) for key in policy_config.image_features},
        "observation.state": torch.zeros(batch_size, 32),
        "action": torch.zeros(batch_size, horizon, 32),
        "task": ["pick the cup"] * batch_size,
    }
    if labels is not None:
        batch["text_labels"] = labels
    return batch


def test_text_loss_joins_the_flow_objective_under_its_configured_weight():
    policy = _lightweight_policy()
    policy.config.flow_loss_weight = 2.0
    policy.config.text_loss_weight = 0.25
    policy.model = _TextLossFlow(text_loss=4.0)

    labels = torch.tensor([[-100, 7, 9]])
    loss, metrics = policy.forward(_text_batch(labels=labels))

    # flow is all-ones so its mean is 1.0: 2.0 * 1.0 + 0.25 * 4.0
    assert loss.item() == pytest.approx(3.0)
    assert metrics["flow_loss"].item() == pytest.approx(1.0)
    assert metrics["text_loss"].item() == pytest.approx(4.0)
    assert torch.equal(policy.model.seen_labels, labels)


def test_action_only_batch_reports_no_text_loss():
    policy = _lightweight_policy()
    policy.model = _TextLossFlow(text_loss=4.0)

    loss, metrics = policy.forward(_text_batch())

    assert "text_loss" not in metrics
    assert policy.model.seen_labels is None
    assert loss.item() == pytest.approx(1.0)


def test_text_supervision_rejects_per_sample_reduction():
    policy = _lightweight_policy()
    policy.model = _TextLossFlow(text_loss=4.0)

    with pytest.raises(ValueError, match="reduction='mean'"):
        policy.forward(_text_batch(labels=torch.tensor([[-100, 7, 9]])), reduction="none")


def test_text_cross_entropy_supervises_only_labelled_next_tokens():
    from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAFlowMatching

    model = object.__new__(HyVLAFlowMatching)
    nn.Module.__init__(model)
    head = nn.Linear(2, 5, bias=False)
    model.dual_tower = SimpleNamespace(vlm=SimpleNamespace(language_model=SimpleNamespace(lm_head=head)))
    # Prefix is [image token, lang0, lang1, lang2]; only the language tail is read.
    prefix_out = torch.tensor([[[9.0, 9.0], [0.5, -0.5], [1.0, 0.25], [-0.25, 0.75]]])
    lang_tokens = torch.zeros(1, 3, dtype=torch.long)
    labels = torch.tensor([[-100, 3, -100]])

    loss = model.text_cross_entropy(prefix_out, lang_tokens, labels)

    expected = torch.nn.functional.cross_entropy(
        head(prefix_out[:, -3:][:, :-1][torch.tensor([[True, False]])]), torch.tensor([3])
    )
    assert loss.item() == pytest.approx(expected.item())
    assert model.text_cross_entropy(prefix_out, lang_tokens, None) is None
    assert model.text_cross_entropy(prefix_out, lang_tokens, torch.full((1, 3), -100)) is None


def test_generate_text_uses_base_contract_and_reuses_the_hy_prefix():
    policy = _lightweight_policy()
    captured = {}

    def fake_generate_text_tokens(images, image_masks, tokens, language_masks, **kwargs):
        captured.update(kwargs)
        captured["tokens"] = tokens
        return torch.tensor([[5, 6]])

    policy.model.generate_text_tokens = fake_generate_text_tokens
    batch = {**_text_batch(), "messages": [[{"role": "user", "content": "pick the cup"}]]}

    assert HyVLAPolicy.generate_text is not PreTrainedPolicy.generate_text
    assert HyVLAPolicy.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert policy.supports_text_generation()
    subtask = policy.generate_text(batch)
    assert subtask == "decoded"
    assert captured["max_new_tokens"] == policy.config.text_max_new_tokens
    # The task went through Hy's own tokenizer path, so the chat suffix is applied.
    assert policy.language_tokenizer.received == ["pick the cup" + policy.config.task_suffix]

    policy.generate_text({**batch, "messages": [[{"role": "user", "content": "which cup is closest?"}]]})
    assert policy.language_tokenizer.received == ["which cup is closest?" + policy.config.task_suffix]

    with pytest.raises(ValueError, match="preprocessed `messages`"):
        policy.generate_text(_text_batch())
