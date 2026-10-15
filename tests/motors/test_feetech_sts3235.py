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

from unittest.mock import patch

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

pytest.importorskip("scservo_sdk")


@pytest.fixture
def mixed_bus():
    models = ("sts3235", "sts3250", "sts3250", "sts3235", "sts3215", "sts3215")
    return FeetechMotorsBus(
        port="unused",
        motors={
            f"joint_{i}": Motor(i, model, MotorNormMode.RANGE_M100_100) for i, model in enumerate(models, 1)
        },
    )


def test_mixed_sts_models_are_identified(mixed_bus):
    reported = {1: 2057, 2: 2825, 3: 2825, 4: 2057, 5: 777, 6: 777}
    with patch.object(mixed_bus, "ping", side_effect=reported.__getitem__) as ping:
        mixed_bus._assert_motors_exist()
    assert ping.call_count == 6
    assert mixed_bus._model_nb_to_model(2057) == "sts3235"


@pytest.mark.parametrize("reported_model", [777, 2825, 9999, None])
def test_sts3235_rejects_incorrect_or_missing_motor(mixed_bus, reported_model):
    reported = {1: reported_model, 2: 2825, 3: 2825, 4: 2057, 5: 777, 6: 777}
    with patch.object(mixed_bus, "ping", side_effect=reported.__getitem__), pytest.raises(RuntimeError):
        mixed_bus._assert_motors_exist()


@pytest.mark.parametrize("value", [-2047, -1, 0, 2047])
def test_sts3235_homing_offset_encoding(mixed_bus, value):
    encoded = mixed_bus._encode_sign("Homing_Offset", {1: value})
    assert encoded[1] == (abs(value) | (1 << 11) if value < 0 else value)
    assert mixed_bus._decode_sign("Homing_Offset", encoded) == {1: value}


def test_sts3235_rejects_scs_protocol():
    with pytest.raises(RuntimeError, match="incompatible"):
        FeetechMotorsBus(
            port="unused",
            motors={"base": Motor(1, "sts3235", MotorNormMode.RANGE_M100_100)},
            protocol_version=1,
        )


def test_sts3235_reads_position_at_sts_register(mixed_bus):
    mixed_bus.port_handler.is_open = True
    with patch.object(mixed_bus, "_read", return_value=(2048, 0, 0)) as read:
        assert mixed_bus.read("Present_Position", "joint_1", normalize=False) == 2048
    assert read.call_args.args[:3] == (56, 2, 1)


def test_sts3235_homing_uses_12_bit_resolution(mixed_bus):
    assert mixed_bus._get_half_turn_homings({"joint_1": 2047, "joint_4": 1024}) == {
        "joint_1": 0,
        "joint_4": -1023,
    }
