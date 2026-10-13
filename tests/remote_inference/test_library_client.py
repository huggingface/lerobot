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

"""The documented library path over a real loopback server, without the rollout package.

Mirrors ``examples/remote_inference/client_loop.py``: discover, admit, build the
engine around the client, publish canonical captures and dispatch every tick
through ``dispatch_tick`` with a caller-owned hold.
"""

import time
from threading import Event

import numpy as np
import pytest
import torch

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")

from lerobot.inference import dispatch_tick
from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine
from lerobot.utils.action_interpolator import ActionInterpolator


class FakeRobot:
    """The whole hardware contract the library path relies on."""

    def __init__(self):
        self.sent = []
        self.holds = 0

    def get_observation(self):
        return {f"joint_{index}.pos": 1.0 for index in range(3)}

    def send(self, target):
        self.sent.append(target.clone())

    def hold(self):
        self.holds += 1


@pytest.mark.parametrize("remote_server", ["robot"], indirect=True)
def test_control_loop_outside_rollout_dispatches_and_holds(remote_server):
    _, config = remote_server
    client = RemoteClient.connect(config)
    capabilities = client.capabilities
    client.admit(
        features=capabilities.features,
        action_feature=capabilities.action_feature,
        semantics=config.semantics,
        action_interval=capabilities.action_interval,
        mode=config.mode,
    )
    state_names = next(
        feature.names for feature in capabilities.features if feature.name == "observation.state"
    )
    robot = FakeRobot()
    engine = RemoteInferenceEngine(client, config, task="pick up the cube")
    interpolator = ActionInterpolator(multiplier=3)

    def capture():
        captured = time.monotonic()
        observation = robot.get_observation()
        engine.submit_observation(
            {"observation.state": np.array([observation[name] for name in state_names], dtype=np.float32)},
            capture_time=captured,
        )

    engine.start()
    engine.resume()
    try:
        deadline = time.monotonic() + 3
        target = None
        while target is None and time.monotonic() < deadline:
            capture()
            target = dispatch_tick(engine, interpolator, send=robot.send, hold=robot.hold)
            if target is None:
                Event().wait(0.002)
        assert target is not None, "no action was dispatched within the deadline"
        torch.testing.assert_close(target, torch.tensor([1.0, 4.0, 7.0]), rtol=0, atol=0)
        assert robot.sent[-1].equal(target)
        # Startup ticks before the first chunk are denied and therefore held.
        holds_before = robot.holds
        assert engine.dispatched_task == "pick up the cube"
        assert engine.runtime.current is not None and engine.runtime.current.session_id == client.session_id

        # Leave interpolation mid-way, then fault: the next tick holds instead of sending.
        capture()
        assert dispatch_tick(engine, interpolator, send=robot.send, hold=robot.hold) is not None
        assert not interpolator.needs_new_action()
        assert robot.holds == holds_before, "a permitted interpolation tick does not hold"
        sent_before = len(robot.sent)
        engine._fault("test server failure")
        assert dispatch_tick(engine, interpolator, send=robot.send, hold=robot.hold) is None
        assert len(robot.sent) == sent_before and robot.holds == holds_before + 1
        assert interpolator.needs_new_action()
        assert engine.failed
    finally:
        engine.stop()
