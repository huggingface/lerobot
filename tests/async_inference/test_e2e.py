# Copyright 2025 The HuggingFace Inc. team.
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
"""End-to-end test of the asynchronous inference stack (client  ↔  server).

This test spins up a lightweight gRPC `PolicyServer` instance with a stubbed
policy network and launches a `RobotClient` that uses a `MockRobot`.  The goal
is to exercise the full communication loop:

1. Client sends policy specification → Server
2. Client streams observations → Server
3. Server streams action chunks → Client
4. Client executes received actions

The test succeeds if at least one action is executed and the server records at
least one predicted timestep - demonstrating that the gRPC round-trip works
end-to-end using real (but lightweight) protocol messages.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

import grpc
import torch

from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.transport import async_inference_pb2_grpc  # type: ignore
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.helpers import TimedObservation
from lerobot.scripts.server.policy_server import PolicyServer
from lerobot.scripts.server.robot_client import RobotClient
from tests.async_inference.test_policy_server import policy_server  # noqa: F401

# -----------------------------------------------------------------------------
# End-to-end test
# -----------------------------------------------------------------------------


def test_async_inference_e2e(policy_server, monkeypatch):  # noqa: F811
    """Tests the full asynchronous inference pipeline."""
    # ------------------------------------------------------------------
    # 1. Spawn a PolicyServer returning dummy action chunks
    # ------------------------------------------------------------------
    from lerobot.common.transport import async_inference_pb2  # type: ignore
    from lerobot.scripts.server.helpers import map_robot_keys_to_lerobot_features
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()
    mock_robot = make_robot_from_config(test_config)

    lerobot_features = map_robot_keys_to_lerobot_features(mock_robot)
    policy_server.lerobot_features = lerobot_features

    # Force server to produce deterministic action chunks in test mode
    policy_server.policy_type = "act"

    def _fake_get_action_chunk(_self, _obs, _type="test"):
        action_dim = 6
        batch_size = 1
        actions_per_chunk = policy_server.actions_per_chunk

        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    # Bypass potentially heavy model loading inside SendPolicyInstructions
    def _fake_send_policy_instructions(self, request, context):  # noqa: N802
        return async_inference_pb2.Empty()

    monkeypatch.setattr(PolicyServer, "SendPolicyInstructions", _fake_send_policy_instructions, raising=True)

    # Build gRPC server running a PolicyServer
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy_server"))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)

    # Use the host/port specified in the fixture's config
    server_address = f"{policy_server.config.host}:{policy_server.config.port}"
    server.add_insecure_port(server_address)
    server.start()

    # ------------------------------------------------------------------
    # 2. Create a RobotClient around the MockRobot
    # ------------------------------------------------------------------
    client_config = RobotClientConfig(
        server_address=server_address,
        robot=mock_robot,
        chunk_size_threshold=0.0,
        policy_type="test",
        pretrained_name_or_path="test",
        lerobot_features=lerobot_features,
    )

    client = RobotClient(client_config)
    assert client.start(), "Client failed initial handshake with the server"

    # Track action chunks received without modifying RobotClient
    action_chunks_received = {"count": 0}
    original_aggregate = client._aggregate_action_queues

    def counting_aggregate(*args, **kwargs):
        action_chunks_received["count"] += 1
        return original_aggregate(*args, **kwargs)

    monkeypatch.setattr(client, "_aggregate_action_queues", counting_aggregate)

    # Observation producer – very simple state vector
    def _make_observation():
        obs_dict = mock_robot.get_observation()

        return TimedObservation(
            timestamp=time.time(),
            observation=obs_dict,
            timestep=max(client.latest_action, 0),
        )

    # Start client threads
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, args=(_make_observation,), daemon=True)
    action_thread.start()
    control_thread.start()

    # ------------------------------------------------------------------
    # 3. System exchanges a few messages
    # ------------------------------------------------------------------
    # Wait for 5 seconds
    server.wait_for_termination(timeout=5)

    assert action_chunks_received["count"] > 0, "Client did not receive any action chunks"
    assert len(policy_server._predicted_timesteps) > 0, "Server did not record any predicted timesteps"

    # ------------------------------------------------------------------
    # 4. Stop the system
    # ------------------------------------------------------------------
    client.stop()
    action_thread.join()
    control_thread.join()
    policy_server.stop()
    server.stop(grace=None)
