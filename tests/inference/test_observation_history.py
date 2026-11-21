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

"""Capture-time observation history and its declaration in ChunkPolicySpec."""

from __future__ import annotations

import math

import pytest
import torch

from lerobot.inference.observation_history import ObservationHistory
from lerobot.policies import ChunkPolicySpec

FPS = 10.0
STATE = "observation.state"


def record(history: ObservationHistory, *captures: tuple[float, float]) -> None:
    for capture_time, value in captures:
        history.record({"value": value}, capture_time)


def sampled(history: ObservationHistory, name: str = STATE) -> tuple[list[float], list[bool]]:
    frames = history.frames()
    indices, padding = history.select(frames)[name]
    return [frames[index].observation["value"] for index in indices], list(padding)


def test_offsets_select_the_captures_of_past_policy_ticks():
    history = ObservationHistory({STATE: (-2, -1, 0)}, FPS)
    record(history, *((1.0 + tick / FPS, float(tick)) for tick in range(6)))
    assert sampled(history) == ([3.0, 4.0, 5.0], [False, False, False])


def test_captures_republished_within_a_tick_replace_the_newest():
    history = ObservationHistory({STATE: (-1, 0)}, FPS)
    # A starved loop republishes on motor ticks between two policy ticks.
    record(history, (1.00, 0.0), (1.10, 1.0), (1.12, 1.2), (1.14, 1.4))
    assert [frame.observation["value"] for frame in history.frames()] == [0.0, 1.4]
    assert sampled(history) == ([0.0, 1.4], [False, False])


def test_offsets_before_the_first_capture_of_a_run_are_padding():
    history = ObservationHistory({STATE: (-2, -1, 0)}, FPS)
    record(history, (1.0, 5.0))
    assert sampled(history) == ([5.0, 5.0, 5.0], [True, True, False])
    record(history, (1.1, 6.0))
    assert sampled(history) == ([5.0, 5.0, 6.0], [True, False, False])


def test_clear_starts_a_new_run():
    history = ObservationHistory({STATE: (-1, 0)}, FPS)
    record(history, (1.0, 0.0), (1.1, 1.0), (1.2, 2.0))
    history.clear()
    record(history, (5.0, 10.0))
    assert sampled(history) == ([10.0, 10.0], [True, False])


def test_only_reachable_captures_are_kept_over_a_long_run():
    history = ObservationHistory({STATE: (-2, 0)}, FPS)
    record(history, *((1.0 + tick / FPS, float(tick)) for tick in range(200)))
    assert len(history.frames()) <= 4
    assert sampled(history) == ([197.0, 199.0], [False, False])


def test_assemble_prepares_each_capture_once_with_the_current_one_last():
    history = ObservationHistory({STATE: (-2, -1, 0)}, FPS)
    record(history, (1.0, 0.0), (1.1, 1.0))
    prepared: list[float] = []

    def prepare(observation):
        prepared.append(observation["value"])
        return {STATE: torch.tensor([[observation["value"], -observation["value"]]]), "task": ["pick"]}

    batch = history.assemble(history.frames(), prepare)
    assert prepared == [0.0, 1.0]
    assert batch[STATE].shape == (1, 3, 2)
    assert batch[STATE][0, :, 0].tolist() == [0.0, 0.0, 1.0]
    assert batch[f"{STATE}_is_pad"].tolist() == [[True, False, False]]
    assert batch["task"] == ["pick"]


def test_assemble_rejects_a_declared_feature_missing_after_preprocessing():
    history = ObservationHistory({"observation.images.front": (-1, 0)}, FPS)
    record(history, (1.0, 0.0), (1.1, 1.0))
    with pytest.raises(ValueError, match="observation.images.front"):
        history.assemble(history.frames(), lambda observation: {STATE: torch.zeros(1, 2)})


@pytest.mark.parametrize(
    ("offsets", "fps"),
    [({}, FPS), ({STATE: ()}, FPS), ({STATE: (-1, 1)}, FPS), ({STATE: (0,)}, 0.0), ({STATE: (0,)}, math.inf)],
)
def test_history_rejects_invalid_offsets_and_rates(offsets, fps):
    with pytest.raises(ValueError):
        ObservationHistory(offsets, fps)


def test_spec_declares_history_only_with_temporal_sampling():
    history = ((STATE, (-1, 0)), ("observation.images.front", (-1, 0)))
    spec = ChunkPolicySpec(
        prediction_steps=8, execution_steps=8, current_observation_only=False, observation_history=history
    )
    assert dict(spec.observation_history) == {STATE: (-1, 0), "observation.images.front": (-1, 0)}
    assert hash(spec) == hash(spec)
    with pytest.raises(ValueError, match="current_observation_only=False"):
        ChunkPolicySpec(prediction_steps=8, execution_steps=8, observation_history=history)


@pytest.mark.parametrize(
    "history",
    [
        ((STATE, ()),),
        ((STATE, (1, 0)),),
        ((STATE, (0, 0)),),
        ((STATE, (-1.0, 0)),),
        ((STATE, (-1, 0)), (STATE, (0,))),
        (("", (0,)),),
    ],
)
def test_spec_rejects_malformed_history(history):
    with pytest.raises(ValueError):
        ChunkPolicySpec(
            prediction_steps=8, execution_steps=8, current_observation_only=False, observation_history=history
        )
