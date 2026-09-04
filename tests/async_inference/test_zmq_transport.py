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
"""Tests for ZeroMQ transport in async_inference."""

import threading
import time

import pytest
import torch

pytest.importorskip("zmq")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
from lerobot.async_inference.policy_server import PolicyServer, serve_zmq
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


class MockPolicy:
    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self):
            return {}

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        return self

    def model(self, batch):
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


def test_zmq_async_inference_e2e(monkeypatch):
    """Test full async inference loop using ZeroMQ transport."""
    server_port = 9998
    server_address = f"127.0.0.1:{server_port}"

    policy_server_config = PolicyServerConfig(host="127.0.0.1", port=server_port, transport="zmq")
    policy_server = PolicyServer(policy_server_config)

    policy_server.policy = MockPolicy()
    policy_server.actions_per_chunk = 20
    policy_server.device = "cpu"
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor

    robot_config = MockRobotConfig()
    mock_robot = make_robot_from_config(robot_config)
    lerobot_features = map_robot_keys_to_lerobot_features(mock_robot)
    policy_server.lerobot_features = lerobot_features
    policy_server.policy_type = "act"

    def _fake_get_action_chunk(_self, _obs, _type="test"):
        return torch.zeros(1, policy_server.actions_per_chunk, 6)

    def _fake_handle_policy_instructions(self, _specs, client_id):
        self.client_id = client_id

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)
    monkeypatch.setattr(PolicyServer, "_handle_policy_instructions", _fake_handle_policy_instructions, raising=True)

    # Spin up ZMQ server thread
    server_thread = threading.Thread(
        target=serve_zmq, args=(policy_server_config, policy_server), daemon=True
    )
    server_thread.start()
    time.sleep(0.2)

    client = None
    try:
        # Create ZMQ client
        client_config = RobotClientConfig(
            server_address=server_address,
            transport="zmq",
            robot=robot_config,
            chunk_size_threshold=0.0,
            policy_type="act",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
        )

        client = RobotClient(client_config)
        assert client.start(), "Client failed ZeroMQ handshake with PolicyServer"

        action_chunks_received = {"count": 0}
        original_aggregate = client._aggregate_action_queues

        def counting_aggregate(*args, **kwargs):
            action_chunks_received["count"] += 1
            return original_aggregate(*args, **kwargs)

        monkeypatch.setattr(client, "_aggregate_action_queues", counting_aggregate)

        # Start action receiving thread
        action_thread = threading.Thread(target=client.receive_actions, daemon=True)
        control_thread = threading.Thread(target=client.control_loop, args=({"task": ""}), daemon=True)
        action_thread.start()
        control_thread.start()

        time.sleep(1.0)

        assert action_chunks_received["count"] > 0, "Client did not receive any action chunks over ZMQ"
        assert len(policy_server._predicted_timesteps) > 0, "Server recorded no predicted timesteps over ZMQ"

    finally:
        if client is not None:
            client.stop()
        policy_server.stop()
        server_thread.join(timeout=2.0)
