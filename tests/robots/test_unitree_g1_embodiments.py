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

"""Embodiment contracts; no model downloads, simulation, or physical connections."""

import dataclasses

import draccus
import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import _DEFAULT_KD, _DEFAULT_KP, UnitreeG1Config
from lerobot.robots.unitree_g1.g1_embodiments import get_g1_embodiment
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    G1_23_JointArmIndex,
    G1_23_JointIndex,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)


def test_g1_29_definition_and_defaults_are_unchanged():
    spec = get_g1_embodiment("g1_29")
    config = UnitreeG1Config()
    assert spec.joint_index is G1_29_JointIndex
    assert spec.arm_index is G1_29_JointArmIndex
    assert config.embodiment == "g1_29"
    assert config.kp == _DEFAULT_KP
    assert config.kd == _DEFAULT_KD
    assert config.kp == [150, 150, 150, 300, 40, 40] * 2 + [250] * 3 + [50, 50, 80, 80, 40, 40, 40] * 2
    assert config.kd == [2, 2, 2, 4, 2, 2] * 2 + [5] * 3 + [3, 3, 3, 3, 1.5, 1.5, 1.5] * 2
    assert config.gravity_compensation is False
    assert config.controller is None
    assert config.is_simulation is True


def test_g1_23_sparse_indices_are_not_renumbered():
    assert NUM_MOTORS == 29
    assert len(G1_23_JointIndex) == 23
    assert len(G1_23_JointArmIndex) == 10
    assert [j.value for j in G1_23_JointArmIndex] == [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
    assert {j.value for j in G1_23_JointIndex} == set(range(29)) - {13, 14, 20, 21, 27, 28}
    for joint in G1_23_JointIndex:
        assert G1_29_JointIndex[joint.name].value == joint.value
    for joint in G1_23_JointArmIndex:
        assert G1_23_JointIndex[joint.name].value == joint.value


def test_g1_23_derived_defaults_and_inactive_slots():
    config = UnitreeG1Config(embodiment="g1_23")
    expected_kp = [300, 300, 300, 300, 80, 300] * 2 + [300, 0, 0]
    expected_kp += [80, 80, 80, 80, 40, 0, 0] * 2
    expected_kd = [3] * 13 + [0, 0] + [3, 3, 3, 3, 1.5, 0, 0] * 2
    assert config.kp == expected_kp
    assert config.kd == expected_kd
    assert config.default_positions == [0.0] * NUM_MOTORS


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_config_round_trip_and_independent_defaults(embodiment):
    config = UnitreeG1Config(embodiment=embodiment)
    restored = draccus.decode(UnitreeG1Config, draccus.encode(config))
    assert dataclasses.asdict(config) == dataclasses.asdict(restored)
    config.kp[0] = 999
    config.kd[0] = 999
    config.default_positions[0] = 999
    other = UnitreeG1Config(embodiment=embodiment)
    assert restored.kp == other.kp
    assert restored.kd == other.kd
    assert restored.default_positions == other.default_positions


def test_custom_gains_are_preserved():
    values = [0.0] * NUM_MOTORS
    for joint in G1_23_JointIndex:
        values[joint.value] = 12.0
    config = UnitreeG1Config(embodiment="g1_23", kp=values)
    assert config.kp == values
    assert config.kp is not values
    assert config.kd == UnitreeG1Config(embodiment="g1_23").kd


@pytest.mark.parametrize("field", ["kp", "kd", "default_positions"])
@pytest.mark.parametrize("values", [[0.0] * 23, [float("nan")] * 29, [float("inf")] * 29])
def test_bad_transport_vectors_are_rejected(field, values):
    with pytest.raises(ValueError, match="29 finite values"):
        UnitreeG1Config(embodiment="g1_23", **{field: values})


@pytest.mark.parametrize("field", ["kp", "kd", "default_positions"])
def test_nonzero_inactive_slot_is_rejected(field):
    values = [0.0] * NUM_MOTORS
    values[20] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        UnitreeG1Config(embodiment="g1_23", **{field: values})


@pytest.mark.parametrize("field", ["kp", "kd"])
def test_negative_gains_are_rejected(field):
    with pytest.raises(ValueError, match="nonnegative"):
        UnitreeG1Config(**{field: [-1.0] * NUM_MOTORS})


def test_unknown_and_unsupported_configuration():
    with pytest.raises(ValueError, match="Unknown G1 embodiment"):
        UnitreeG1Config(embodiment="g1_27")
    with pytest.raises(ValueError, match="Controllers"):
        UnitreeG1Config(embodiment="g1_23", controller="SonicWholeBodyController")
    with pytest.raises(ValueError, match="Gravity compensation"):
        UnitreeG1Config(embodiment="g1_23", gravity_compensation=True)


def test_model_selection_is_metadata_not_backend_support():
    spec = get_g1_embodiment("g1_23")
    assert spec.model_urdf.endswith("g1_body23.urdf")
    assert "unitree_lerobot" in spec.model_repository
    assert spec.simulation_env is None
    assert not spec.supports_hardware
    assert not spec.supports_gravity_compensation
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "g1_29"
