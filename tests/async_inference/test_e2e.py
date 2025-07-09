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
from concurrent import futures

import pytest
import torch

# Skip entire module if grpc is not available
pytest.importorskip("grpc")

# -----------------------------------------------------------------------------
# End-to-end test
# -----------------------------------------------------------------------------


def test_async_inference_e2e(monkeypatch):
    """Tests the full asynchronous inference pipeline."""
    # Import grpc-dependent modules inside the test function
    import grpc

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.scripts.server.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.scripts.server.helpers import map_robot_keys_to_lerobot_features
    from lerobot.scripts.server.policy_server import PolicyServer
    from lerobot.scripts.server.robot_client import RobotClient
    from lerobot.transport import (
        async_inference_pb2,  # type: ignore
        async_inference_pb2_grpc,  # type: ignore
    )
    from tests.mocks.mock_robot import MockRobotConfig

    # Create a stub policy similar to test_policy_server.py
    class MockPolicy:
        """A minimal mock for an actual policy, returning zeros."""

        class _Config:
            robot_type = "dummy_robot"

            @property
            def image_features(self):
                """Empty image features since this test doesn't use images."""
                return {}

        def __init__(self):
            self.config = self._Config()

        def to(self, *args, **kwargs):
            return self

        def model(self, batch):
            # Return a chunk of 20 dummy actions.
            batch_size = len(batch["robot_type"])
            return torch.zeros(batch_size, 20, 6)

    # ------------------------------------------------------------------
    # 1. Create PolicyServer instance with mock policy
    # ------------------------------------------------------------------
    policy_server_config = PolicyServerConfig(host="localhost", port=9999)
    policy_server = PolicyServer(policy_server_config)
    # Replace the real policy with our fast, deterministic stub.
    policy_server.policy = MockPolicy()
    policy_server.actions_per_chunk = 20
    policy_server.device = "cpu"

    # Set up robot config and features
    robot_config = MockRobotConfig()
    mock_robot = make_robot_from_config(robot_config)

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
        robot=robot_config,
        chunk_size_threshold=0.0,
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        verify_robot_cameras=False,
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

    # Start client threads
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, args=({"task": ""}), daemon=True)
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
