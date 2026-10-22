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

"""Deterministic RTC delays on scalar sims, without changing their lockstep contract."""

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from lerobot.robots.sim import SimRobot, SimRobotConfig
from lerobot.sims import backend
from tests.env_server.toy_env import ToyEnvConfig
from tests.inference.rtc_helpers import assert_delayed_chunk_overlap, sim_rtc
from tests.utils import wait_for


class LongToyEnv(backend.ToyEnv):
    def __init__(self):
        super().__init__()
        self._max_episode_steps = 64


@pytest.fixture(params=["local", "server"])
def robot(request, monkeypatch, tmp_path) -> Iterator[SimRobot]:
    monkeypatch.setattr(backend, "ToyEnv", LongToyEnv)
    endpoint = request.getfixturevalue("server")[1] if request.param == "server" else None
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=endpoint), seed=42, calibration_dir=tmp_path))
    try:
        robot.connect()
        yield robot
    finally:
        robot.disconnect()


@pytest.mark.parametrize("mode,delay_steps", [(m, d) for m in ("guided", "trained") for d in (0, 1, 3)])
def test_rtc_delays_are_control_steps_not_machine_speed(robot, mode, delay_steps):
    with sim_rtc(robot, mode=mode) as rig:
        initial = robot.get_observation()
        assert_delayed_chunk_overlap(rig, delay_steps)
        assert robot.get_observation()["state_0"] > initial["state_0"]


def test_guided_rtc_can_replace_a_chunk_near_buffer_exhaustion(robot):
    with sim_rtc(robot) as rig:
        assert_delayed_chunk_overlap(rig, delay_steps=6)


def test_rtc_startup_wait_does_not_advance_physics_or_trim_the_first_chunk(robot):
    with sim_rtc(robot) as rig:
        rig.wait_for_call(1)
        for _ in range(5):
            assert rig.tick() is None
            assert robot.episode_status.step == 0
        rig.complete(1)
        assert rig.engine.action_queue.qsize() == 8
        assert rig.engine._runtime.last_accept["trimmed_actions"] == 0
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[0][0])
        assert robot.episode_status.step == 1


def test_trained_rtc_rejects_delay_beyond_its_conditioned_prefix(robot):
    with sim_rtc(robot, mode="trained") as rig:
        rig.complete(1)
        assert rig.tick() is not None
        rig.wait_for_call(2)
        assert rig.policy.calls[1][1] == 4
        for _ in range(6):
            assert rig.tick() is not None
        rig.complete(2)
        assert not rig.engine._runtime.last_accept["accepted"]
        assert rig.engine.action_queue.qsize() == 1, "A rejected chunk must not replace the usable old tail"
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[0][7])
        assert robot.episode_status.step == 8
        assert not rig.engine.failed


def test_reset_discards_queued_and_inflight_actions_before_resuming(robot):
    with sim_rtc(robot) as rig:
        rig.complete(1)
        assert rig.tick() is not None
        rig.wait_for_call(2)
        old_generation = rig.engine._runtime.generation
        rig.engine.pause()
        rig.wrapper.reset_episode(seed=123)
        rig.engine.reset()
        rig.interpolator.reset()
        rig.engine.resume()
        rig.capture()
        rig.policy.release.release()  # Finish the prediction that started before reset.
        rig.wait_for_call(3)
        assert rig.engine._runtime.generation > old_generation
        assert rig.engine.action_queue.empty()
        assert robot.episode_status.step == 0
        assert rig.policy.calls[2][0] is None, "A new episode must not receive an old action prefix"
        expected = np.array([robot.get_observation()[name] for name in robot._state_feature.names])
        np.testing.assert_allclose(rig.policy.action_observations[2][0].numpy(), expected)
        for _ in range(2):
            assert rig.tick() is None
            assert robot.episode_status.step == 0
        rig.policy.release.release()
        assert wait_for(lambda: not rig.engine.action_queue.empty()), rig.engine.failure_traceback
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[2][0])
        assert robot.episode_status.step == 1


@pytest.mark.parametrize("recover", [False, True])
def test_rtc_starvation_freezes_sim_and_requires_fresh_prediction(robot, recover):
    with sim_rtc(robot, grace_s=1.0 if recover else 0.0) as rig:
        rig.complete(1)
        assert rig.tick() is not None
        rig.wait_for_call(2)
        for _ in range(7):
            assert rig.tick() is not None
        before = robot.get_observation()
        assert rig.tick() is None
        assert robot.episode_status.step == 8
        for key, value in before.items():
            np.testing.assert_array_equal(robot.get_observation()[key], value)
        if not recover:
            assert rig.engine.failed
            assert "buffer exhausted" in rig.engine.failure_traceback
            return
        assert not rig.engine.failed
        rig.policy.release.release()  # The stale in-flight result cannot resume motion.
        assert wait_for(lambda: not rig.engine._hold.requested)
        assert rig.engine.action_queue.empty()
        assert rig.tick() is None  # Capture after the acknowledged hold.
        rig.wait_for_call(3)
        rig.policy.release.release()
        assert wait_for(lambda: not rig.engine.action_queue.empty()), rig.engine.failure_traceback
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[2][0])
        assert robot.episode_status.step == 9
        assert not rig.engine.failed


def test_rtc_starvation_grace_expiry_never_resumes_motion(robot):
    with sim_rtc(robot) as rig:
        rig.complete(1)
        assert rig.tick() is not None
        rig.wait_for_call(2)
        for _ in range(7):
            assert rig.tick() is not None
        assert rig.tick() is None
        rig.clock.now = rig.engine._runtime.starvation_deadline + 0.01
        assert rig.tick() is None
        assert rig.engine.failed
        assert "grace expired" in rig.engine.failure_traceback
        rig.policy.release.release()
        assert wait_for(lambda: not rig.engine._rtc_thread.is_alive())
        assert rig.engine.action_queue.empty()
        assert rig.tick() is None
        assert robot.episode_status.step == 8
