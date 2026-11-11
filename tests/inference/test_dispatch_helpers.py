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

"""Dispatch helpers: permission gates, holds and acknowledgments without rollout."""

from contextlib import contextmanager

import pytest
import torch

from lerobot.inference import InferenceEngine, dispatch_tick
from lerobot.utils.action_interpolator import ActionInterpolator


class GateEngine(InferenceEngine):
    """Scripted permission and action production; counts every contract call."""

    control_thread_owns_policy = True

    def __init__(self, permissions=()):
        super().__init__("task")
        self.permissions = list(permissions)  # consumed first, then ``allowed`` applies
        self.allowed = True
        self.actions: list[torch.Tensor] = []
        self.frames = []
        self.acknowledged = 0
        self.dispatched = 0
        self.revoke_during_pull = False

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def dispatch_allowed(self):
        return self.permissions.pop(0) if self.permissions else self.allowed

    def acknowledge_hold(self):
        self.acknowledged += 1

    def record_dispatch(self):
        self.dispatched += 1

    def get_action(self, obs_frame):
        self.frames.append(obs_frame)
        if self.revoke_during_pull:
            self.allowed = False
        return self.actions.pop(0) if self.actions else None


class Robot:
    def __init__(self):
        self.sent = []
        self.holds = 0

    def send(self, target):
        self.sent.append(target.item())

    def hold(self):
        self.holds += 1


class Timer:
    def __init__(self):
        self.sections = []
        self.starved = 0

    @contextmanager
    def section(self, name):
        self.sections.append(name)
        yield

    def note_starved_tick(self):
        self.starved += 1


def test_permitted_ticks_pull_once_per_policy_action_and_send_each_interpolated_target():
    engine = GateEngine()
    engine.actions = [torch.tensor([3.0]), torch.tensor([6.0])]
    interpolator = ActionInterpolator(multiplier=3)
    robot, timer = Robot(), Timer()
    frames_built = 0

    def frame():
        nonlocal frames_built
        frames_built += 1
        return {"built": frames_built}

    targets = [
        dispatch_tick(engine, interpolator, frame, send=robot.send, hold=robot.hold, timer=timer)
        for _ in range(4)
    ]
    assert [target.item() for target in targets] == [3.0, 4.0, 5.0, 6.0] == robot.sent
    assert frames_built == 2, "the frame factory runs only when a new policy action is pulled"
    assert engine.frames == [{"built": 1}, {"built": 2}]
    assert engine.dispatched == 4 and engine.acknowledged == 0 and robot.holds == 0
    assert timer.sections.count("infer") == 2 and timer.sections.count("send") == 4 and timer.starved == 0


def test_denial_before_the_pull_holds_acknowledges_and_clears_interpolation():
    engine = GateEngine()
    engine.actions = [torch.tensor([1.0]), torch.tensor([2.0])]
    interpolator = ActionInterpolator(multiplier=2)
    robot = Robot()
    for _ in range(2):
        dispatch_tick(engine, interpolator, None, send=robot.send, hold=robot.hold)
    assert not interpolator.needs_new_action(), "an interpolated target is still buffered"

    engine.allowed = False
    assert dispatch_tick(engine, interpolator, None, send=robot.send, hold=robot.hold) is None
    assert len(engine.frames) == 2, "a denied tick pulls nothing"
    assert interpolator.needs_new_action(), "a denial clears buffered interpolation"
    assert robot.holds == 1 and engine.acknowledged == 1
    assert robot.sent == [1.0, 1.5] and engine.dispatched == 2


def test_permission_revoked_during_the_pull_never_sends_the_returned_action():
    engine = GateEngine()
    engine.revoke_during_pull = True
    engine.actions = [torch.tensor([1.0])]
    robot = Robot()
    assert dispatch_tick(engine, ActionInterpolator(), None, send=robot.send, hold=robot.hold) is None
    assert len(engine.frames) == 1 and robot.sent == [] and engine.dispatched == 0
    assert robot.holds == 1 and engine.acknowledged == 1


def test_revocation_between_interpolation_and_the_write_skips_the_send():
    engine = GateEngine(permissions=[True, True, False])
    engine.actions = [torch.tensor([1.0])]
    robot = Robot()
    assert dispatch_tick(engine, ActionInterpolator(), None, send=robot.send, hold=robot.hold) is None
    assert robot.sent == [] and engine.dispatched == 0
    assert robot.holds == 1 and engine.acknowledged == 1


def test_a_denial_without_a_configured_hold_is_still_acknowledged():
    engine = GateEngine()
    engine.allowed = False
    robot = Robot()
    assert dispatch_tick(engine, ActionInterpolator(), None, send=robot.send) is None
    assert engine.acknowledged == 1 and engine.frames == [] and robot.sent == []


def test_a_failed_hold_is_never_acknowledged():
    engine = GateEngine()
    engine.allowed = False

    def hold():
        raise RuntimeError("motor bus")

    with pytest.raises(RuntimeError, match="motor bus"):
        dispatch_tick(engine, ActionInterpolator(), None, send=Robot().send, hold=hold)
    assert engine.acknowledged == 0


def test_a_starved_tick_is_counted_and_sends_nothing():
    engine = GateEngine()  # no actions: the engine yields None
    robot, timer = Robot(), Timer()
    assert dispatch_tick(engine, ActionInterpolator(), None, send=robot.send, timer=timer) is None
    assert len(engine.frames) == 1 and timer.starved == 1 and timer.sections == ["infer"]
    assert robot.sent == [] and engine.acknowledged == 0 and engine.dispatched == 0
