#!/usr/bin/env python

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

"""Tests for the controllers' processor pipelines and frame history."""

import numpy as np
import torch

from lerobot.controllers.history import FrameHistory
from lerobot.controllers.processors import make_controller_processors
from lerobot.processor.converters import (
    observation_to_transition,
    policy_action_to_transition,
    transition_to_observation,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import DataProcessorPipeline

DEFAULT = np.array([0.1, -0.2, 0.3, 0.4], np.float32)
ORDER = [2, 0, 3, 1]


class TestControllerProcessors:
    def test_observation_is_scaled_in_robot_order_then_reordered(self):
        scales = np.full(4, 0.5)
        offsets = -DEFAULT * 0.5
        pre, _ = make_controller_processors(
            observation_affine={"q": (scales, offsets)},
            action_affine=(np.ones(4), np.zeros(4)),
            observation_order={"q": ORDER, "dq": ORDER},
        )
        q = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
        dq = np.array([10.0, 20.0, 30.0, 40.0], np.float32)

        obs = pre({"q": q, "dq": dq})

        np.testing.assert_allclose(np.asarray(obs["q"]), (scales * q + offsets)[ORDER], rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(obs["dq"]), dq[ORDER])

    def test_action_is_reordered_to_robot_order_then_scaled(self):
        """A residual network action becomes `scales * a + default`, in robot joint order."""
        scale = np.array([0.25, 0.5, 1.0, 2.0], np.float32)
        _, post = make_controller_processors(
            observation_affine={}, action_affine=(scale, DEFAULT), action_order=ORDER
        )
        action = np.array([1.0, 2.0, 3.0, 4.0], np.float32)

        target = post(torch.from_numpy(action)).numpy()

        np.testing.assert_allclose(target, scale * action[ORDER] + DEFAULT, rtol=1e-6)

    def test_scalar_scale_applies_to_every_joint(self):
        _, post = make_controller_processors(observation_affine={}, action_affine=(0.25, DEFAULT))

        target = post(torch.ones(4)).numpy()

        np.testing.assert_allclose(target, 0.25 + DEFAULT, rtol=1e-6)

    def test_selected_action_joints(self):
        _, post = make_controller_processors(
            observation_affine={}, action_affine=(1.0, DEFAULT[:2]), action_order=range(2)
        )
        assert post(torch.ones(4)).shape == (2,)

    def test_pipelines_round_trip_through_save_pretrained(self, tmp_path):
        """The coefficients and orders live in the saved processor, not in the controller's code."""
        pre, post = make_controller_processors(
            observation_affine={"q": (np.full(4, 0.5), -DEFAULT * 0.5)},
            action_affine=(0.25, DEFAULT),
            observation_order={"q": ORDER},
            action_order=ORDER,
        )
        pre.save_pretrained(tmp_path / "pre")
        post.save_pretrained(tmp_path / "post")
        loaded_pre = DataProcessorPipeline.from_pretrained(
            tmp_path / "pre",
            config_filename="controller_preprocessor.json",
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )
        loaded_post = DataProcessorPipeline.from_pretrained(
            tmp_path / "post",
            config_filename="controller_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        q = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
        torch.testing.assert_close(loaded_pre({"q": q})["q"], pre({"q": q})["q"])
        action = torch.tensor([1.0, -1.0, 0.5, 2.0])
        torch.testing.assert_close(loaded_post(action), post(action))


class TestFrameHistory:
    def test_starts_full_of_the_initial_frame(self):
        history = FrameHistory(3, np.array([1.0, 0.0]))
        assert len(history) == 3
        np.testing.assert_array_equal(history.stacked(), [1.0, 0.0] * 3)

    def test_push_drops_the_oldest_and_stacks_oldest_first(self):
        history = FrameHistory(3, np.zeros(1))
        for value in (1.0, 2.0, 3.0, 4.0):
            history.push(np.array([value]))

        np.testing.assert_array_equal(history.stacked(), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(history.newest, [4.0])

    def test_pushed_frames_are_copied(self):
        history = FrameHistory(2, np.zeros(2))
        frame = np.ones(2)
        history.push(frame)
        frame[:] = 5.0
        np.testing.assert_array_equal(history.newest, [1.0, 1.0])

    def test_reset_refills_with_the_initial_frame(self):
        history = FrameHistory(2, np.array([1.0, 0.0, 0.0, 0.0]))
        history.push(np.zeros(4))
        history.reset()
        assert all(np.array_equal(frame, [1.0, 0.0, 0.0, 0.0]) for frame in history)

    def test_accepts_tensors(self):
        history = FrameHistory(2, np.zeros(2, np.float32))
        history.push(torch.tensor([1.0, 2.0]))
        assert isinstance(history.newest, np.ndarray)
