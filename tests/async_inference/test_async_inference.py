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

import math
import time
import pickle

import torch

from lerobot.scripts.server.helpers import (
    TimedAction,
    TimedObservation,
    observations_similar,
)

# ---------------------------------------------------------------------
# TimedData helpers
# ---------------------------------------------------------------------


def test_timed_action_getters():
    """TimedAction stores & returns timestamp, action tensor and timestep."""
    ts = time.time()
    action = torch.arange(10)
    ta = TimedAction(timestamp=ts, action=action, timestep=0)

    assert math.isclose(ta.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    torch.testing.assert_close(ta.get_action(), action)
    assert ta.get_timestep() == 0


def test_timed_observation_getters():
    """TimedObservation stores & returns timestamp, dict and timestep."""
    ts = time.time()
    obs_dict = {"observation.state": torch.ones(6)}
    to = TimedObservation(timestamp=ts, observation=obs_dict, timestep=0)

    assert math.isclose(to.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert to.get_observation() is obs_dict
    assert to.get_timestep() == 0


def test_timed_data_deserialization_data_getters():
    """TimedAction / TimedObservation survive a round-trip through ``pickle``.

    The async-inference stack uses ``pickle.dumps`` to move these objects across
    the gRPC boundary (see RobotClient.send_observation and PolicyServer.StreamActions).
    This test ensures that the payload keeps its content intact after
    the (de)serialization round-trip.
    """
    ts = time.time()

    # ------------------------------------------------------------------
    # TimedAction
    # ------------------------------------------------------------------
    original_action = torch.randn(6)
    ta_in = TimedAction(timestamp=ts, action=original_action, timestep=13)

    # Serialize → bytes → deserialize
    ta_bytes = pickle.dumps(ta_in)
    ta_out: TimedAction = pickle.loads(ta_bytes)  # nosec B301

    # Identity & content checks
    assert math.isclose(ta_out.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert ta_out.get_timestep() == 13
    torch.testing.assert_close(ta_out.get_action(), original_action)

    # ------------------------------------------------------------------
    # TimedObservation
    # ------------------------------------------------------------------
    obs_dict = {"observation.state": torch.arange(4).float()}
    to_in = TimedObservation(timestamp=ts, observation=obs_dict, timestep=7, transfer_state=2, must_go=True)

    to_bytes = pickle.dumps(to_in)
    to_out: TimedObservation = pickle.loads(to_bytes)  # nosec B301

    assert math.isclose(to_out.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert to_out.get_timestep() == 7
    assert to_out.transfer_state == 2
    assert to_out.must_go is True
    assert to_out.get_observation().keys() == obs_dict.keys()
    torch.testing.assert_close(to_out.get_observation()["observation.state"], obs_dict["observation.state"])


# ---------------------------------------------------------------------
# observations_similar()
# ---------------------------------------------------------------------


def _make_obs(state: torch.Tensor) -> TimedObservation:
    return TimedObservation(
        timestamp=time.time(),
        observation={"observation.state": state},
        timestep=0,
    )


def test_observations_similar_true():
    """Distance below atol → observations considered similar."""
    obs1 = _make_obs(torch.zeros(6))
    obs2 = _make_obs(0.5 * torch.ones(6))
    assert observations_similar(obs1, obs2, atol=2.0)

    obs3 = _make_obs(2.0 * torch.ones(6))
    assert not observations_similar(obs1, obs3, atol=2.0)

