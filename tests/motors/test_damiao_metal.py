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

import math
from types import SimpleNamespace

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import damiao as damiao_module
from lerobot.motors.damiao.tables import MODEL_NAMES, MOTOR_LIMIT_PARAMS, MotorType


@pytest.fixture
def bus(monkeypatch):
    """A Metal-wired bus that never opens CAN, so it runs without python-can."""
    monkeypatch.setattr(damiao_module, "require_package", lambda *a, **kw: None)
    motors = {
        "elbow_flex": Motor(3, "metal_jlo", MotorNormMode.DEGREES, motor_type_str="metal_jlo", recv_id=0x13),
        "shoulder_lift": Motor(2, "metal_j2", MotorNormMode.DEGREES, motor_type_str="metal_j2", recv_id=0x12),
    }
    return damiao_module.DamiaoMotorsBus("can0", motors, use_can_fd=False)


def test_metal_motor_types_carry_the_metal_mit_scaling():
    limits = {MODEL_NAMES[motor_type]: MOTOR_LIMIT_PARAMS[motor_type] for motor_type in MotorType}
    assert limits["metal_jlo"] == (6.28, 10, 30)
    assert limits["metal_j2"] == (6.28, 10, 120)
    assert limits["metal_jhi"] == (6.28, 30, 20)


def test_metal_replies_decode_with_the_metal_ranges(bus):
    # Motor 3, position code 0xC000 (+3.14 rad of 6.28), zero velocity and torque.
    bus._process_response(
        "elbow_flex", SimpleNamespace(data=bytes([0x13, 0xC0, 0x00, 0x7F, 0xF7, 0xFF, 30, 31]))
    )
    state = bus._last_known_states["elbow_flex"]
    assert state["position"] == pytest.approx(math.degrees(3.14), abs=0.01)
    assert (state["temp_mos"], state["temp_rotor"]) == (30.0, 31.0)


def test_mit_frames_use_the_metal_ranges(bus):
    # 90 degrees of 6.28 rad PMAX, and 60 N*m of the j2's 120 N*m TMAX.
    data = bus._encode_mit_packet(MotorType.METAL_J2, 0.0, 0.0, 90.0, 0.0, 60.0)
    position = (data[0] << 8) | data[1]
    torque = ((data[6] & 0xF) << 8) | data[7]
    assert position == int((math.pi / 2 + 6.28) / 12.56 * 0xFFFF)
    assert torque == int(180 / 240 * 0xFFF)
