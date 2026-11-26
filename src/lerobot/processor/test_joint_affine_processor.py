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

import math

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import DataProcessorPipeline, JointAffineProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION, OBS_STATE

# Degrees to radians for two joints, a 0-100 gripper to 0-1, and a flipped joint with an offset.
SCALES = [math.pi / 180, math.pi / 180, 0.01, -1.0]
OFFSETS = [0.0, 0.0, 0.0, 90.0]


def test_maps_the_leading_state_and_action_entries():
    step = JointAffineProcessorStep(scales=SCALES, offsets=OFFSETS)
    state = torch.tensor([[180.0, -90.0, 50.0, 30.0, 7.0]])
    out = step(create_transition(observation={OBS_STATE: state, "other": state}, action=state[0, :4]))

    expected = torch.tensor([[math.pi, -math.pi / 2, 0.5, 60.0, 7.0]])
    torch.testing.assert_close(out[TransitionKey.OBSERVATION][OBS_STATE], expected)
    torch.testing.assert_close(out[TransitionKey.OBSERVATION]["other"], state)
    torch.testing.assert_close(out[TransitionKey.ACTION], expected[0, :4])


def test_inverse_round_trips():
    forward = JointAffineProcessorStep(scales=SCALES, offsets=OFFSETS)
    inverse = JointAffineProcessorStep(scales=SCALES, offsets=OFFSETS, inverse=True)
    action = torch.tensor([[12.0, -45.0, 80.0, -10.0, 3.0]])

    mapped = forward(create_transition(action=action))
    torch.testing.assert_close(inverse(mapped)[TransitionKey.ACTION], action)


def test_empty_scales_is_a_no_op():
    transition = create_transition(observation={OBS_STATE: torch.ones(1, 3)}, action=torch.ones(3))

    assert JointAffineProcessorStep()(transition) is transition


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"scales": [1.0, 2.0], "offsets": [0.0]}, "match scales"), ({"scales": [1.0, 0.0]}, "non-zero")],
)
def test_rejects_invalid_coefficients(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointAffineProcessorStep(**kwargs)


def test_transform_stats_matches_the_mapped_values():
    step = JointAffineProcessorStep(scales=SCALES, offsets=OFFSETS)
    values = np.random.default_rng(0).uniform(-100.0, 100.0, size=(1000, 5))
    stats = {
        OBS_STATE: {
            "mean": values.mean(0),
            "std": values.std(0),
            "min": values.min(0),
            "max": values.max(0),
            "q01": np.quantile(values, 0.01, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
            "count": np.array([1000]),
        }
    }
    mapped_values = step._map(values).numpy()
    mapped = step.transform_stats(stats)[OBS_STATE]

    np.testing.assert_allclose(mapped["mean"], mapped_values.mean(0))
    np.testing.assert_allclose(mapped["std"], mapped_values.std(0))
    np.testing.assert_allclose(mapped["min"], mapped_values.min(0))
    np.testing.assert_allclose(mapped["max"], mapped_values.max(0))
    np.testing.assert_allclose(mapped["q01"], np.quantile(mapped_values, 0.01, axis=0))
    np.testing.assert_allclose(mapped["q99"], np.quantile(mapped_values, 0.99, axis=0))
    assert mapped["count"] is stats[OBS_STATE]["count"]
    assert ACTION not in step.transform_stats(stats)


@pytest.mark.parametrize(
    ("key", "stat", "partner"),
    [(OBS_STATE, "q10", "q90"), (ACTION, "q90", "q10"), (OBS_STATE, "min", "max"), (ACTION, "max", "min")],
)
def test_transform_stats_requires_partner_for_flipped_joints(key: str, stat: str, partner: str) -> None:
    step = JointAffineProcessorStep(scales=[2.0, -1.0])
    stats = {key: {stat: np.array([2.0, 5.0])}}

    with pytest.raises(ValueError, match=rf"{stat}.*{partner}"):
        step.transform_stats(stats)


def test_transform_stats_accepts_unpaired_stats_for_positive_scales() -> None:
    step = JointAffineProcessorStep(scales=[2.0], offsets=[1.0])
    stats = {OBS_STATE: {"q10": np.array([12.0, 7.0])}}

    np.testing.assert_array_equal(step.transform_stats(stats)[OBS_STATE]["q10"], [25.0, 7.0])


def test_save_and_load(tmp_path):
    step = JointAffineProcessorStep(scales=SCALES, offsets=OFFSETS, inverse=True)
    DataProcessorPipeline(steps=[step], name="post").save_pretrained(tmp_path)
    loaded = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="post.json").steps[0]

    assert loaded == step
