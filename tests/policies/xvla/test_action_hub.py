#!/usr/bin/env python

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

"""Regression tests for XVLA action spaces."""

import pytest
import torch

from lerobot.policies.xvla.action_hub import BimanualSO101ActionSpace


class TestBimanualSO101ActionSpace:
    # SO-101 grippers are continuous joint positions, not binary open/close.
    # preprocess must not zero them (the model needs them as conditioning) and
    # postprocess must not squash them through sigmoid (training uses plain MSE,
    # so a sigmoid at inference creates a train/inference mismatch that pins the
    # gripper output near 0.5). See discussion in fix/xvla_so_bimanual.

    @pytest.fixture
    def space(self):
        return BimanualSO101ActionSpace()

    def test_preprocess_preserves_gripper_values(self, space):
        proprio = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
        action = proprio.clone() + 100.0

        proprio_m, action_m = space.preprocess(proprio, action)

        # Padded to model dim
        assert proprio_m.shape == (1, 1, 20)
        assert action_m.shape == (1, 1, 20)

        # Gripper channels (idx 5, 11) must carry the original values through
        for idx in space.gripper_idx:
            assert proprio_m[..., idx].item() == proprio[..., idx].item()
            assert action_m[..., idx].item() == action[..., idx].item()

    def test_preprocess_accepts_none_action(self, space):
        proprio = torch.randn(2, 3, 12)
        proprio_m, action_m = space.preprocess(proprio, None)
        assert proprio_m.shape == (2, 3, 20)
        assert action_m is None

    def test_postprocess_does_not_apply_sigmoid(self, space):
        # Large logits — if sigmoid were applied, output would saturate near 1.
        action = torch.full((1, 1, 20), 5.0)
        out = space.postprocess(action)

        assert out.shape == (1, 1, 12)
        assert torch.allclose(out, torch.full_like(out, 5.0))

    def test_postprocess_preserves_gripper_sign_and_magnitude(self, space):
        action = torch.zeros(1, 1, 20)
        action[..., 5] = -2.5  # left gripper
        action[..., 11] = 3.5  # right gripper

        out = space.postprocess(action)

        assert out[..., 5].item() == pytest.approx(-2.5)
        assert out[..., 11].item() == pytest.approx(3.5)

    def test_postprocess_rejects_too_short_input(self, space):
        with pytest.raises(ValueError):
            space.postprocess(torch.zeros(1, 1, 8))
