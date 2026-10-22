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

"""Shared lockstep execution, validation and failure handling."""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.env_server.session import SimSession, check_open
from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError


@pytest.fixture
def session():
    config = BackendConfig()
    probe = Backend(config)
    try:
        descriptor = probe.descriptor()
    finally:
        probe.close()
    world = SimSession(config, descriptor, 1, seeds=[0])
    try:
        yield world
    finally:
        world.close()


@pytest.mark.parametrize("count", [0, -1, 2, True, 1.0, "1"])
def test_invalid_batch_sizes_are_rejected(count):
    with pytest.raises(ProtocolError, match="Batch size"):
        check_open(count, 1)


def test_descriptor_has_no_unused_execution_options(session):
    assert (
        not {"clocks", "gripper_indices", "command_retention_indices"} & session.descriptor.to_dict().keys()
    )


@pytest.mark.parametrize("operation", ["apply", "hold"])
def test_realtime_operations_are_not_supported(session, operation):
    with pytest.raises(ProtocolError) as error:
        session.handle(operation, {})
    assert error.value.code == ErrorCode.UNSUPPORTED
    assert session.backend.snapshot().step.tolist() == [0]


@pytest.mark.parametrize(
    "actions",
    [np.zeros((1, 2), np.float64), np.zeros((1, 2), np.int32), [[0.0, 0.0]], np.zeros((1, 3), np.float32)],
)
def test_invalid_actions_do_not_advance_or_fault_the_world(session, actions):
    with pytest.raises(ProtocolError) as error:
        session.handle("step", {"actions": actions})
    assert error.value.code == ErrorCode.INCOMPATIBLE
    assert session.fault is None
    assert session.backend.snapshot().step.tolist() == [0]


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_invalid_results_fault_the_session(session, monkeypatch, operation):
    invalid = replace(session.backend.snapshot(), reward=np.array([np.nan], np.float32))
    monkeypatch.setattr(session.backend, operation, lambda *args: invalid)
    with pytest.raises(ValueError, match="reward"):
        session.handle(operation, {"actions": np.zeros((1, 2), np.float32)})
    assert session.fault is not None
    with pytest.raises(ProtocolError) as error:
        session.handle("snapshot", {})
    assert error.value.code == ErrorCode.EXECUTION


def test_step_returns_its_terminal_observation_until_explicit_reset(session):
    reply = session.handle("step", {"actions": np.ones((1, 2), np.float32) * 2})
    assert "execution" not in reply
    terminal = reply["result"]
    assert terminal["terminated"].tolist() == [True]
    frozen = session.handle("step", {"actions": np.ones((1, 2), np.float32)})["result"]
    np.testing.assert_array_equal(frozen["obs"]["observation.state"], terminal["obs"]["observation.state"])
    np.testing.assert_array_equal(frozen["step"], terminal["step"])
    assert frozen["reward"].tolist() == [0]
    reset = session.handle("reset", {"seeds": [0]})
    assert reset["generation"] == 1
    assert reset["result"]["step"].tolist() == [0]
