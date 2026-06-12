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

"""Shared fixtures for the remote-inference test suite.

The mock policy is deterministic: chunk[t, j] = state[j] + 0.01 * t (so
tests can predict exact values), accepts the RTC kwargs, and records
every call for assertions.  Pipelines mimic the
``PolicyProcessorPipeline`` surface the server uses (``__call__``,
``reset``, ``steps``); the mock postprocessor doubles actions so tests
can tell model-space from robot-space chunks.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from threading import Event

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policy_server.manifest import ModelSpec, PolicyServerManifest, ZenohSpec
from lerobot.policy_server.validation import PolicyClassification, ServingClass

ACTION_DIM = 6
CHUNK_SIZE = 20
STATE_DIM = 6
IMG_H, IMG_W = 48, 64
ACTION_NAMES = [f"joint_{i}.pos" for i in range(ACTION_DIM)]
TASK = "test task"
MODEL_ID = "mock/model"


# ---------------------------------------------------------------------------
# Mock policy & config
# ---------------------------------------------------------------------------


@dataclass
class MockPolicyConfig:
    type: str = "mockchunk"
    pretrained_path: str = MODEL_ID
    chunk_size: int = CHUNK_SIZE
    action_feature_names: list[str] = field(default_factory=lambda: list(ACTION_NAMES))
    input_features: dict = field(
        default_factory=lambda: {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_H, IMG_W)),
        }
    )
    rtc_config: object | None = None


class MockChunkPolicy:
    """Deterministic chunk policy with the RTC kwargs surface."""

    name = "mockchunk"

    def __init__(self, config: MockPolicyConfig | None = None):
        self.config = config or MockPolicyConfig()
        self.calls: list[dict] = []
        self.reset_count = 0
        self.rtc_initialized = False

    # nn.Module surface the server touches
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def reset(self):
        self.reset_count += 1

    def init_rtc_processor(self):
        self.rtc_initialized = True

    def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None):
        state = batch["observation.state"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        self.calls.append(
            {
                "state": state.detach().clone(),
                "inference_delay": inference_delay,
                "prev_chunk_left_over": None
                if prev_chunk_left_over is None
                else prev_chunk_left_over.detach().clone(),
                "task": batch.get("task"),
            }
        )
        steps = torch.arange(CHUNK_SIZE, dtype=torch.float32).unsqueeze(1) * 0.01
        return (state[:, :ACTION_DIM].unsqueeze(1) + steps.unsqueeze(0)).clone()


# ---------------------------------------------------------------------------
# Mock processor pipelines
# ---------------------------------------------------------------------------


class MockPipeline:
    """Mimics the PolicyProcessorPipeline surface used by the server."""

    def __init__(self, transform=None, steps=()):
        self._transform = transform
        self.steps = list(steps)
        self.reset_count = 0
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return self._transform(x) if self._transform is not None else x

    def reset(self):
        self.reset_count += 1


def make_mock_processors():
    """Identity preprocessor + doubling postprocessor (model vs robot space)."""
    return MockPipeline(), MockPipeline(transform=lambda actions: actions * 2.0)


# ---------------------------------------------------------------------------
# Server fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_policy():
    return MockChunkPolicy()


@pytest.fixture
def shared_rtc_classification():
    return PolicyClassification(
        ServingClass.SHARED, supports_rtc=True, needs_queue_population=False, reason="mock"
    )


def make_manifest(**overrides) -> PolicyServerManifest:
    kwargs = {
        "model": ModelSpec(repo_or_path=MODEL_ID, device="cpu"),
        "zenoh": ZenohSpec(mode="peer"),
        "default_task": TASK,
        "max_sessions": 4,
        "warmup_inferences": 0,
        "trained_fps": 30.0,
        "health_port": 0,
    }
    kwargs.update(overrides)
    return PolicyServerManifest(**kwargs)


@pytest.fixture
def manifest():
    return make_manifest()


def make_logic_server(
    manifest: PolicyServerManifest | None = None,
    policy: MockChunkPolicy | None = None,
    classification: PolicyClassification | None = None,
    processor_factory=None,
):
    """A PolicyServer with everything injected and no zenoh transport."""
    from lerobot.policy_server.server import PolicyServer

    policy = policy or MockChunkPolicy()
    factory_calls = []

    def default_factory():
        pair = make_mock_processors()
        factory_calls.append(pair)
        return pair

    server = PolicyServer(
        manifest or make_manifest(),
        policy=policy,
        policy_cfg=policy.config,
        processor_factory=processor_factory or default_factory,
        classification=classification
        or PolicyClassification(
            ServingClass.SHARED, supports_rtc=True, needs_queue_population=False, reason="mock"
        ),
    )
    server.load_policy()
    server.factory_calls = factory_calls
    return server


# ---------------------------------------------------------------------------
# Client-side fixtures (hw features, observations)
# ---------------------------------------------------------------------------


@pytest.fixture
def hw_features():
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(ACTION_NAMES),
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
    }


def make_robot_obs(seed: float = 1.0) -> dict:
    obs = {name: seed + 0.1 * i for i, name in enumerate(ACTION_NAMES)}
    rng = np.random.default_rng(int(seed * 10))
    obs["front"] = rng.integers(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    return obs


@pytest.fixture
def shutdown_event():
    return Event()


# ---------------------------------------------------------------------------
# Loopback helpers
# ---------------------------------------------------------------------------


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_loopback_manifest(port: int, **overrides) -> PolicyServerManifest:
    return make_manifest(
        zenoh=ZenohSpec(mode="peer", listen_endpoints=[f"tcp/127.0.0.1:{port}"]),
        **overrides,
    )


def make_remote_config(port: int, **overrides):
    """RemoteInferenceConfig dialing a loopback server (fast watchdogs)."""
    from lerobot.rollout.inference.factory import RemoteInferenceConfig

    kwargs = {
        "connect_endpoint": f"tcp/127.0.0.1:{port}",
        "zenoh_mode": "peer",
        "service_model_id": MODEL_ID,
        "service_task": TASK,
        "jpeg_quality": 0,  # raw images: byte-exact loopback
        "buffer_time_s": 0.2,
        "handshake_timeout_s": 2.0,
        "request_timeout_s": 1.0,
        "degraded_after_s": 0.3,
        "reconnect_initial_backoff_s": 0.1,
        "reconnect_max_backoff_s": 0.5,
        "max_offline_s": 8.0,
    }
    kwargs.update(overrides)
    return RemoteInferenceConfig(**kwargs)
