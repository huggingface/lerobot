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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.common.robot_devices.robots.utils.make_robot` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import time

import pytest
import torch

from lerobot.scripts.server.helpers import TimedAction
from lerobot.scripts.server.robot_client import RobotClient

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class _DummyRobot:
    # TODO(fracapuano): move to robot_client.py, create a base class and make RobotClient and DummyRobotClient subclass
    """Minimal stub matching the interface used by `RobotClient`."""

    def __init__(self):
        self.actions_sent: list[torch.Tensor] = []
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send_action(self, action: torch.Tensor):
        # Simply store what was sent for inspection, no actuation.
        self.actions_sent.append(action)

    def capture_observation(self):  # pragma: no cover – not used in this file
        raise RuntimeError("Not needed for queue tests")


@pytest.fixture(autouse=True)
def patch_make_robot(monkeypatch):
    """Replace `make_robot` with a deterministic dummy implementation."""

    def _factory(*_a, **_kw):
        return _DummyRobot()

    # Patch the name 'make_robot' in the module where it is imported and used.
    monkeypatch.setattr("lerobot.scripts.server.robot_client.make_robot", _factory, raising=True)


@pytest.fixture()
def robot_client() -> RobotClient:
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    from lerobot.scripts.server.configs import RobotClientConfig

    # Use an arbitrary port; the gRPC channel is never used in these tests.
    test_config = RobotClientConfig(server_address="localhost:9999")
    client = RobotClient(test_config)

    yield client

    if client.robot.connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int) -> list[TimedAction]:
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    fps = 30  # emulates most common frame-rate
    actions: list[TimedAction] = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(timestamp, action_tensor, timestep))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client: RobotClient):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._update_action_queue(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(
    robot_client: RobotClient, chunk_size: int, queue_len: int, expected: bool
):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client._clear_action_queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(
    robot_client: RobotClient, g_threshold: float, expected: bool
):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client._clear_action_queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected
