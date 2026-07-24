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

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import threading
import time
from queue import Queue

import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

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
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_observation_queue_keeps_latest_and_preserves_must_go(robot_client):
    from lerobot.async_inference.helpers import TimedObservation

    first = TimedObservation(
        timestamp=time.time(),
        timestep=10,
        observation={"state": 1},
        must_go=True,
    )
    latest = TimedObservation(
        timestamp=time.time(),
        timestep=11,
        observation={"state": 2},
        must_go=False,
    )

    robot_client._queue_observation(first)
    robot_client._queue_observation(latest)

    queued = robot_client.observation_queue.get_nowait()
    robot_client.observation_queue.task_done()
    assert queued is latest
    assert queued.must_go is True
    assert robot_client.observation_request_pending.is_set()


def test_observation_upload_runs_in_background(monkeypatch, robot_client):
    from lerobot.async_inference.helpers import TimedObservation

    upload_started = threading.Event()
    upload_finished = threading.Event()

    def slow_send(_observation):
        upload_started.set()
        time.sleep(0.2)
        upload_finished.set()
        return True

    monkeypatch.setattr(robot_client, "send_observation", slow_send)
    robot_client._start_observation_sender()

    observation = TimedObservation(
        timestamp=time.time(),
        timestep=0,
        observation={"state": 1},
        must_go=True,
    )
    queue_start = time.perf_counter()
    robot_client._queue_observation(observation)
    queue_duration = time.perf_counter() - queue_start

    assert queue_duration < 0.05
    assert upload_started.wait(timeout=1)
    assert upload_finished.wait(timeout=1)


def test_pending_observation_blocks_normal_send_but_not_must_go(robot_client):
    robot_client.action_chunk_size = 20
    robot_client.action_queue = Queue()
    for action in _make_actions(start_ts=time.time(), start_t=0, count=8):
        robot_client.action_queue.put(action)

    robot_client.must_go.clear()
    robot_client.observation_request_pending.set()
    assert robot_client._ready_to_send_observation() is False

    robot_client.action_queue = Queue()
    robot_client.must_go.set()
    assert robot_client._ready_to_send_observation() is True


def test_threshold_observation_is_forced_through_server_filter(monkeypatch, robot_client):
    queued_observations = []
    monkeypatch.setattr(robot_client, "_queue_observation", queued_observations.append)

    robot_client.must_go.set()
    robot_client.control_loop_observation(task="test task")

    assert len(queued_observations) == 1
    assert queued_observations[0].must_go is True
    assert robot_client.must_go.is_set() is False


def test_rtc_client_config_reuses_rollout_hierarchy():
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.policies.rtc import RTCConfig
    from lerobot.rollout.inference.factory import RTCInferenceConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="smolvla",
        pretrained_name_or_path="test",
        actions_per_chunk=50,
        inference=RTCInferenceConfig(
            rtc=RTCConfig(execution_horizon=10, max_guidance_weight=7.5),
            queue_threshold=30,
        ),
    )

    assert config.inference.type == "rtc"
    assert config.inference.rtc.execution_horizon == 10
    assert config.inference.rtc.max_guidance_weight == 7.5
    assert config.inference.queue_threshold == 30


def test_rtc_prefix_snapshot_contains_policy_and_processed_actions(robot_client):
    from lerobot.async_inference.helpers import TimedAction

    robot_client._rtc_enabled = True
    for timestep in range(5, 8):
        robot_client.action_queue.put(
            TimedAction(
                timestamp=time.time(),
                timestep=timestep,
                action=torch.full((6,), float(timestep)),
                original_action=torch.full((6,), float(timestep) / 10),
            )
        )

    policy_prefix, processed_prefix = robot_client._get_rtc_prefixes()

    assert policy_prefix.shape == (3, 6)
    assert processed_prefix.shape == (3, 6)
    torch.testing.assert_close(policy_prefix[:, 0], torch.tensor([0.5, 0.6, 0.7]))
    torch.testing.assert_close(processed_prefix[:, 0], torch.tensor([5.0, 6.0, 7.0]))


def test_rtc_aggregation_preserves_policy_space_prefix(robot_client):
    from lerobot.async_inference.helpers import TimedAction

    robot_client.latest_action = 4
    robot_client.action_queue.put(
        TimedAction(
            timestamp=time.time(),
            timestep=5,
            action=torch.full((6,), 10.0),
            original_action=torch.full((6,), 1.0),
        )
    )
    incoming = [
        TimedAction(
            timestamp=time.time(),
            timestep=5,
            action=torch.full((6,), 20.0),
            original_action=torch.full((6,), 3.0),
        )
    ]

    robot_client._aggregate_action_queues(
        incoming,
        aggregate_fn=lambda old, new: 0.25 * old + 0.75 * new,
    )

    action = robot_client.action_queue.get_nowait()
    torch.testing.assert_close(action.get_action(), torch.full((6,), 17.5))
    torch.testing.assert_close(action.get_original_action(), torch.full((6,), 2.5))


def test_rtc_queue_threshold_and_latency_steps(robot_client):
    robot_client._rtc_enabled = True
    robot_client._rtc_queue_threshold = 30
    robot_client.action_chunk_size = 50
    robot_client.must_go.clear()

    robot_client.action_queue = Queue()
    for action in _make_actions(start_ts=time.time(), start_t=0, count=31):
        robot_client.action_queue.put(action)
    assert robot_client._ready_to_send_observation() is False

    robot_client.action_queue.get_nowait()
    assert robot_client._ready_to_send_observation() is True

    with robot_client._rtc_latency_lock:
        robot_client._rtc_latency_tracker.add(0.201)
    assert robot_client._get_rtc_inference_delay() == 7


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )
