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

import torch
from torch import nn

from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAPolicy


class _CapturingTokenizer:
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


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, images, image_masks, tokens, language_masks, state, actions, noise, time):
        target = actions[..., :20]
        return (self.scale * torch.ones_like(target) - target).square()


class _IndexedLossFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, image_masks, tokens, language_masks, state, actions, noise, time):
        losses = torch.arange(actions.shape[-1], device=actions.device, dtype=actions.dtype)
        return losses.expand_as(actions) + self.anchor * 0


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
