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

"""Local RTC samples a policy's declared observation history from control-thread captures."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="local RTC tests import the dataset extra")

from lerobot.inference import rtc
from tests.inference.rtc_helpers import ControlledPolicy, make_engine, stop
from tests.utils import wait_for

pytestmark = pytest.mark.dataset

FPS = 30


class HistoryPolicy(ControlledPolicy):
    """Declares one previous and the current state, as an ``n_obs_steps=2`` policy would."""

    def __init__(self) -> None:
        super().__init__()
        self.padding: list[list[bool]] = []

    def chunk_inference_spec(self):
        return replace(
            ControlledPolicy.chunk_inference_spec(self),
            current_observation_only=False,
            observation_history=(("observation.state", (-1, 0)),),
        )

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None):
        self.padding.append(batch["observation.state_is_pad"][0].tolist())
        return super().predict_action_chunk(
            batch, inference_delay=inference_delay, prev_chunk_left_over=prev_chunk_left_over
        )


def make_history_engine():
    policy = HistoryPolicy()
    robot = SimpleNamespace(robot_type="mock", action_features={}, supports_hold=True, observation_time=None)
    return policy, robot, make_engine(policy=policy, robot=robot)


def publish(engine, robot, ticks):
    """Publish one capture per policy tick, the last one now; the state is the tick number."""
    now = time.monotonic()
    for tick in ticks:
        robot.observation_time = now + (tick - ticks[-1]) / FPS
        engine.notify_observation({"j1.pos": float(tick), "j2.pos": -float(tick)})


def test_declared_history_is_accepted_and_undeclared_temporal_state_is_not():
    assert rtc.validate_local_chunk_policy(HistoryPolicy()).observation_history
    policy = ControlledPolicy()
    spec = replace(policy.chunk_inference_spec(), current_observation_only=False)
    policy.chunk_inference_spec = lambda: spec
    with pytest.raises(ValueError, match="observation_history"):
        rtc.validate_local_chunk_policy(policy)


def test_prediction_receives_the_previous_and_current_policy_ticks():
    policy, robot, engine = make_history_engine()
    try:
        engine.start()
        publish(engine, robot, [0, 1, 2])
        engine.resume()
        assert wait_for(lambda: policy.action_observations)
        state = policy.action_observations[0]
        assert state.shape == (1, 2, 2)
        assert state[0, :, 0].tolist() == [1.0, 2.0]
        assert state[0, :, 1].tolist() == [-1.0, -2.0]
        assert policy.padding[0] == [False, False]
        policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert not engine.failed
    finally:
        stop(engine)


def test_the_first_capture_of_a_run_pads_the_past():
    policy, robot, engine = make_history_engine()
    try:
        engine.start()
        publish(engine, robot, [7])
        engine.resume()
        assert wait_for(lambda: policy.action_observations)
        assert policy.action_observations[0][0, :, 0].tolist() == [7.0, 7.0]
        assert policy.padding[0] == [True, False]
    finally:
        stop(engine)


@pytest.mark.parametrize("drop", ["pause", "reset"])
def test_dropping_published_captures_restarts_the_history(drop):
    _, robot, engine = make_history_engine()
    publish(engine, robot, [0, 1, 2])
    assert len(engine._history.frames()) == 2
    getattr(engine, drop)()
    assert engine._history.frames() == ()
