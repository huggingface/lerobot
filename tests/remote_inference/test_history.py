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

"""Declared history over a real Zenoh loopback: the client records, samples and sends it."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread

import numpy as np
import pytest
import torch

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")

from lerobot.configs.types import Deltas, FeatureType
from lerobot.inference import (
    ChunkPolicySpec,
    FeatureSpec,
    PolicyRunner,
    RemoteInferenceConfig,
    dispatch_tick,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.engine import RemoteInferenceEngine
from lerobot.remote_inference.server import PolicyServer, SessionWorker
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import ACTION, OBS_ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.policy_helpers import ConformingPolicy, processors, tiny_config
from tests.remote_inference.network_helpers import free_port, signal_server_ready

pytestmark = pytest.mark.dataset

INTERVAL = 1 / 30


@dataclass
class StateWindowConfig(ACTConfig):
    """Conditioned on the two states and the two commands before the current frame."""

    @property
    def inference_deltas(self) -> Deltas:
        return Deltas(observation={FeatureType.STATE: (-2, -1, 0), FeatureType.ACTION: (-2, -1)})


class WindowPolicy(ConformingPolicy):
    """Records each chunk call; conditions on two past states and two past commands."""

    name = "window_conformance"

    def __init__(self, config):
        super().__init__(StateWindowConfig(**vars(config)))
        self.batches = []

    def predict_action_chunk(self, batch):
        self.batches.append({key: value.clone() for key, value in batch.items() if torch.is_tensor(value)})
        current = batch[OBS_STATE][0, -1, 0].item()
        return {ACTION: torch.full((1, 8, 3), current)}


@dataclass
class KeyframeConfig(ACTConfig):
    """Reads every other observation of the last 8 ticks, the current one last."""

    @property
    def inference_deltas(self) -> Deltas:
        return Deltas(observation={FeatureType.ENV: range(-6, 1, 2)})


class CompletePolicy(ConformingPolicy):
    """Executes whole chunks, the first one shorter, and reads the frames they led to."""

    name = "complete_conformance"

    def __init__(self, config):
        super().__init__(KeyframeConfig(**vars(config)))
        self.batches = []

    def chunk_inference_spec(self):
        return ChunkPolicySpec(prediction_steps=8, execution_steps=8, complete_execution=True)

    def predict_action_chunk(self, batch):
        self.batches.append({key: value.clone() for key, value in batch.items() if torch.is_tensor(value)})
        return {ACTION: torch.zeros(1, 4 if len(self.batches) == 1 else 8, 3)}


@contextmanager
def serve(policy):
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    runner = PolicyRunner(
        policy,
        *processors(policy.config),
        action_interval=INTERVAL,
        features=tuple(
            FeatureSpec(key, (3,), "float32", semantics="radians-v1") for key in (OBS_STATE, OBS_ENV_STATE)
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", semantics="radians-v1"),
    )
    worker = SessionWorker(runner, deployment="history", artifact_identity="history", semantics="radians-v1")
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    ready = Event()
    signal_server_ready(transport, ready)
    thread = Thread(target=server.serve, daemon=True)
    try:
        thread.start()
        assert ready.wait(10), "server did not advertise readiness"
        config = RemoteInferenceConfig(
            endpoint=endpoint,
            deployment="history",
            semantics="radians-v1",
            handshake_timeout_s=2,
            action_timeout_s=2,
        )
        client = RemoteClient.connect(config)
        caps = client.capabilities
        client.admit(
            features=caps.features,
            action_feature=caps.action_feature,
            semantics=config.semantics,
            action_interval=caps.action_interval,
            mode=config.mode,
        )
        engine = RemoteInferenceEngine(client, config, task="stack")
        engine.start()
        engine.resume()
        try:
            yield worker, engine
        finally:
            engine.stop()
    finally:
        server.stop()
        thread.join(10)
        transport.close()
        worker.close()


class Robot:
    def __init__(self):
        self.sent: dict[int, torch.Tensor] = {}
        self.tick = 0

    def send(self, target):
        self.sent[self.tick] = target.clone()


def test_windows_and_commands_reach_the_server_as_the_dataset_lays_them_out():
    policy = WindowPolicy(tiny_config())
    with serve(policy) as (worker, engine):
        assert list(worker.descriptor["capabilities"]["history"]["actions"]) == [-2, -1]
        robot, interpolator = Robot(), ActionInterpolator(multiplier=1)
        start = time.monotonic() - 0.5
        for tick in range(40):
            robot.tick = tick
            state = np.full(3, tick, dtype=np.float32)
            engine.submit_observation(
                {OBS_STATE: state, OBS_ENV_STATE: np.zeros(3, dtype=np.float32)}, start + tick * INTERVAL
            )
            dispatch_tick(engine, interpolator, send=robot.send)
            time.sleep(0.01)
        assert len(policy.batches) >= 3, engine.failure_traceback

    for batch in policy.batches:
        current = int(batch[OBS_STATE][0, -1, 0])
        window = batch[OBS_STATE][0, :, 0].tolist()
        assert window == [max(0, current - 2), max(0, current - 1), current]
        assert batch[f"{OBS_STATE}_is_pad"][0].tolist() == [current < 2, current < 1, False]
        expected, pads = [], []
        for offset in (-2, -1):
            sent = [tick for tick in robot.sent if tick <= current + offset]
            pads.append(not sent)
            expected.append(robot.sent[max(sent)][0].item() if sent else 0.0)
        assert batch[f"{OBS_ACTION}_is_pad"][0].tolist() == pads
        assert batch[OBS_ACTION][0, :, 0].tolist() == expected


def test_complete_chunks_hold_at_each_boundary_and_read_the_frames_they_led_to():
    config = tiny_config()
    config.n_action_steps = 8
    policy = CompletePolicy(config)
    with serve(policy) as (worker, engine):
        robot, interpolator = Robot(), ActionInterpolator(multiplier=1)
        holds = 0
        start = time.monotonic()
        for tick in range(80):
            if len(policy.batches) >= 3:
                break
            robot.tick = tick
            capture = start + tick * INTERVAL  # a robot ticking at the control rate
            time.sleep(max(0.0, capture - time.monotonic()))
            engine.submit_observation(
                {OBS_STATE: np.zeros(3, dtype=np.float32), OBS_ENV_STATE: np.full(3, tick, dtype=np.float32)},
                capture,
            )

            def hold():
                nonlocal holds
                holds += 1

            dispatch_tick(engine, interpolator, send=robot.send, hold=hold)
        assert len(policy.batches) >= 3, engine.failure_traceback
        assert not engine.failed

    executed = sorted(robot.sent)
    first, second = executed[:4], executed[4:12]
    assert first == list(range(first[0], first[0] + 4)) and second == list(range(second[0], second[0] + 8))
    assert second[0] > first[-1] + 1, "the robot holds while the next chunk is predicted"
    assert holds > 0
    # Each request observes the result of the previous chunk's last action, and its window
    # holds the observations after every other executed action.
    window = policy.batches[1][OBS_ENV_STATE][0, :, 0].tolist()
    assert window[-2:] == [first[1] + 1, first[3] + 1]
    window = policy.batches[2][OBS_ENV_STATE][0, :, 0].tolist()
    assert window == [second[i] + 1 for i in (1, 3, 5, 7)]
