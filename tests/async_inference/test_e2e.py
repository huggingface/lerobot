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
least one predicted timestep – demonstrating that the gRPC round-trip works
end-to-end using real (but lightweight) protocol messages.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

import grpc
import torch

from lerobot.common.robots.utils import make_robot_from_config
from lerobot.scripts.server import async_inference_pb2_grpc  # type: ignore
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.helpers import TimedObservation
from lerobot.scripts.server.policy_server import PolicyServer
from lerobot.scripts.server.robot_client import RobotClient

# -----------------------------------------------------------------------------
# End-to-end test
# -----------------------------------------------------------------------------


def test_async_inference_e2e(policy_server, monkeypatch):
    """Smoke-test the full asynchronous inference pipeline."""
    # ------------------------------------------------------------------
    # 1. Spawn a PolicyServer returning dummy action chunks
    # ------------------------------------------------------------------
    from lerobot.scripts.server import async_inference_pb2  # type: ignore

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        action_dim = 6
        batch_size = 1
        actions_per_chunk = policy_server.actions_per_chunk

        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    # Bypass potentially heavy model loading inside SendPolicyInstructions
    def _fake_send_policy_instructions(self, request, context):  # noqa: N802
        return async_inference_pb2.Empty()

    monkeypatch.setattr(PolicyServer, "SendPolicyInstructions", _fake_send_policy_instructions, raising=True)

    # Build gRPC server with our PolicyServer instance
    # Create ThreadPoolExecutor separately so we can shut it down properly
    executor = futures.ThreadPoolExecutor(max_workers=4)
    grpc_server = grpc.server(executor)
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)

    # Use the host/port specified in the fixture's config
    server_address = f"{policy_server.config.host}:{policy_server.config.port}"
    grpc_server.add_insecure_port(server_address)
    grpc_server.start()

    # ------------------------------------------------------------------
    # 2. Create a RobotClient backed by a MockRobot
    # ------------------------------------------------------------------
    from tests.mocks.mock_robot import MockRobotConfig

    mock_robot = make_robot_from_config(MockRobotConfig())

    client_config = RobotClientConfig(
        server_address=server_address,
        robot=mock_robot,
        chunk_size_threshold=0.0,
    )

    client = RobotClient(client_config)
    assert client.start(), "Client failed initial handshake with the server"

    # Observation producer – very simple state vector
    def _make_observation():
        obs_dict = mock_robot.get_observation()

        obs_dict = {"observation.state": torch.tensor(list(obs_dict.values()))}

        return TimedObservation(
            timestamp=time.time(),
            observation=obs_dict,
            timestep=max(client.latest_action, 0),
        )

    # Start client threads (daemon so they exit when main thread ends)
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, args=(_make_observation,), daemon=True)
    action_thread.start()
    control_thread.start()

    # ------------------------------------------------------------------
    # 3. Let the system exchange a few messages
    # ------------------------------------------------------------------
    # Wait for at least one chunk, but no more than 5 seconds
    deadline = time.perf_counter() + 5.0
    while client.running and time.perf_counter() < deadline:
        time.sleep(0.05)

    # ------------------------------------------------------------------
    # 4. Shutdown and assert expectations
    # ------------------------------------------------------------------
    client.stop()
    grpc_server.stop(grace=1.0)

    # Explicitly shutdown the ThreadPoolExecutor to prevent hanging
    executor.shutdown(wait=False)

    assert client.chunks_received > 0, "Client did not receive any action chunks"
    assert len(policy_server._predicted_timesteps) > 0, "Server did not record any predicted timesteps"
