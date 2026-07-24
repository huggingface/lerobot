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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        self.last_predict_kwargs = kwargs
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()
        self.last_predict_kwargs = {}
        self.rtc_processor_initialized = False

    @classmethod
    def from_pretrained(cls, _pretrained_name_or_path):
        return cls()

    def init_rtc_processor(self):
        self.rtc_processor_initialized = True

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def eval(self):
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_time_action_chunk_carries_original_policy_actions(policy_server):
    start_ts = time.time()
    processed_actions = [torch.full((6,), 10.0), torch.full((6,), 20.0)]
    original_actions = [torch.full((6,), 1.0), torch.full((6,), 2.0)]

    timed_actions = policy_server._time_action_chunk(
        start_ts,
        processed_actions,
        3,
        original_actions,
    )

    torch.testing.assert_close(timed_actions[0].get_action(), processed_actions[0])
    torch.testing.assert_close(timed_actions[0].get_original_action(), original_actions[0])
    torch.testing.assert_close(timed_actions[1].get_original_action(), original_actions[1])


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_send_observations_decodes_jpeg_payload(policy_server):
    from lerobot.async_inference.helpers import encode_image_to_jpeg
    from lerobot.transport import services_pb2
    from lerobot.transport.utils import send_bytes_in_chunks

    class FakeContext:
        @staticmethod
        def peer():
            return "test-client"

    image = np.full((48, 64, 3), [200, 20, 10], dtype=np.uint8)
    observation = _make_obs(torch.zeros(6), timestep=4, must_go=True)
    observation.observation["front"] = encode_image_to_jpeg(image, quality=90)
    payload = pickle.dumps(observation)
    request_iterator = send_bytes_in_chunks(payload, services_pb2.Observation)

    policy_server.SendObservations(request_iterator, FakeContext())

    queued_observation = policy_server.observation_queue.get_nowait()
    decoded_image = queued_observation.get_observation()["front"]
    assert isinstance(decoded_image, np.ndarray)
    assert decoded_image.shape == image.shape
    assert decoded_image.dtype == np.uint8
    assert decoded_image[24, 32, 0] > 180


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6
        torch.testing.assert_close(ta.get_original_action(), torch.zeros(action_dim))


def test_get_action_chunk_forwards_rtc_kwargs(policy_server):
    from lerobot.policies.rtc import RTCConfig

    policy_server.rtc_config = RTCConfig(execution_horizon=4)
    prefix = torch.ones(4, 6)
    observation = {OBS_STATE: torch.zeros(1, 6)}

    chunk = policy_server._get_action_chunk(
        observation,
        inference_delay=3,
        prev_chunk_left_over=prefix,
    )

    assert chunk.shape == (1, 20, 6)
    assert policy_server.policy.last_predict_kwargs["inference_delay"] == 3
    assert policy_server.policy.last_predict_kwargs["execution_horizon"] == 4
    torch.testing.assert_close(
        policy_server.policy.last_predict_kwargs["prev_chunk_left_over"],
        prefix,
    )


def test_send_policy_instructions_initializes_remote_rtc(monkeypatch, policy_server):
    from lerobot.async_inference import policy_server as policy_server_module
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.policies.rtc import RTCConfig
    from lerobot.transport import services_pb2

    class FakeContext:
        @staticmethod
        def peer():
            return "test-client"

    class EmptyPipeline:
        steps = []

    monkeypatch.setattr(
        policy_server_module,
        "get_policy_class",
        lambda _policy_type: MockPolicy,
    )
    monkeypatch.setattr(
        policy_server_module,
        "make_pre_post_processors",
        lambda *_args, **_kwargs: (EmptyPipeline(), EmptyPipeline()),
    )

    rtc_config = RTCConfig(execution_horizon=4, max_guidance_weight=7.5)
    policy_specs = RemotePolicyConfig(
        policy_type="smolvla",
        pretrained_name_or_path="test-model",
        lerobot_features=policy_server.lerobot_features,
        actions_per_chunk=20,
        device="cpu",
        rtc_config=rtc_config,
    )

    policy_server.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(policy_specs)),
        FakeContext(),
    )

    assert policy_server.rtc_enabled is True
    assert policy_server.policy.config.rtc_config is rtc_config
    assert policy_server.policy.rtc_processor_initialized is True
    assert policy_server.policy.config.rtc_config.execution_horizon == 4


def test_predict_action_chunk_passes_remote_rtc_context(monkeypatch, policy_server):
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.policies.rtc import RTCConfig

    policy_server.policy_type = "smolvla"
    policy_server.rtc_config = RTCConfig(
        execution_horizon=4,
        max_guidance_weight=10.0,
    )
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor + 100

    captured = {}

    def _fake_get_action_chunk(
        _self,
        _obs,
        *,
        inference_delay=0,
        prev_chunk_left_over=None,
    ):
        captured["inference_delay"] = inference_delay
        captured["prev_chunk_left_over"] = prev_chunk_left_over
        return torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    prefix = torch.full((2, 6), 0.5)
    obs = _make_obs(torch.zeros(6), timestep=5)
    obs.rtc_action_prefix = prefix
    obs.rtc_processed_action_prefix = prefix + 10
    obs.rtc_inference_delay = 3

    timed_actions = policy_server._predict_action_chunk(obs)

    assert captured["inference_delay"] == 3
    assert captured["prev_chunk_left_over"].shape == (4, 6)
    torch.testing.assert_close(captured["prev_chunk_left_over"][:2], prefix)
    torch.testing.assert_close(
        captured["prev_chunk_left_over"][2:],
        torch.zeros(2, 6),
    )
    torch.testing.assert_close(
        timed_actions[0].get_original_action(),
        torch.arange(6, dtype=torch.float32),
    )
    torch.testing.assert_close(
        timed_actions[0].get_action(),
        torch.arange(6, dtype=torch.float32) + 100,
    )
