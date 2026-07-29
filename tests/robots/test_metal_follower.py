#!/usr/bin/env python

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
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.damiao.tables import MIT_KD_RANGE, MIT_KP_RANGE, MOTOR_LIMIT_PARAMS
from lerobot.robots.metal_follower.config_metal_follower import MetalFollowerConfig
from lerobot.robots.metal_follower.metal_follower import MetalFollower

# The follower resolves its gains from the config defaults, so read them from there.
DEFAULT_GAINS = MetalFollowerConfig(port="can0").gains
DEFAULT_CAN_IDS = MetalFollowerConfig(port="can0").motor_can_ids


@pytest.fixture
def follower():
    # startup slow-sync off so these tests exercise soft limits / gains directly (a dedicated
    # test covers the sync behaviour). Velocity feedforward is also off for the legacy mock
    # harness; packet-level feedforward tests below use the real Damiao encoder.
    r = MetalFollower(
        MetalFollowerConfig(port="can0", startup_sync_speed_deg=None, velocity_feedforward=False)
    )
    r.bus = MagicMock()
    r.bus.sync_read.return_value = dict.fromkeys(r._joint_motor_names, 0.0)
    return r


def _make_wired_follower(**config_overrides):
    config = MetalFollowerConfig(port="can0", startup_sync_speed_deg=None, **config_overrides)
    follower = MetalFollower(config)
    wire = MagicMock()
    follower.bus.canbus = wire
    follower.bus._recv_all_responses = MagicMock(return_value={})

    def connect_bus():
        follower.bus._is_connected = True

    follower.bus.connect = MagicMock(side_effect=connect_bus)
    follower.bus.enable_torque = MagicMock()
    follower.connect(calibrate=False)
    wire.reset_mock()
    return follower, wire


def _last_frame(wire):
    return wire.send.call_args.args[0]


def _decode_dq_deg_s(follower, frame, motor="shoulder_pan"):
    dq_uint = (frame.data[2] << 4) | (frame.data[3] >> 4)
    vmax = MOTOR_LIMIT_PARAMS[follower.bus._motor_types[motor]][1]
    return math.degrees(dq_uint / ((1 << 12) - 1) * (2.0 * vmax) - vmax)


def _decode_gain(frame, value_range, field):
    if field == "kp":
        value_uint = ((frame.data[3] & 0x0F) << 8) | frame.data[4]
    else:
        value_uint = (frame.data[5] << 4) | (frame.data[6] >> 4)
    return value_uint / ((1 << 12) - 1) * (value_range[1] - value_range[0]) + value_range[0]


def test_has_all_seven_motors(follower):
    assert follower._joint_motor_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ]


def test_action_features_include_all_motors(follower):
    assert "shoulder_pan.pos" in follower.action_features
    assert "gripper.pos" in follower.action_features


def test_velocity_feedforward_config_defaults():
    config = MetalFollowerConfig()
    assert config.velocity_feedforward is True
    assert config.velocity_ff_alpha == 0.35
    assert config.velocity_ff_max_deg_s == 200.0


def test_send_action_writes_goal_and_clamps(follower):
    action = {f"{m}.pos": 999.0 for m in follower._joint_motor_names}
    out = follower.send_action(action)
    assert follower.bus.sync_write.called
    assert out["shoulder_pan.pos"] == 160.0  # clamped to shoulder_pan upper soft limit


def test_factory_builds_metal_follower():
    from lerobot.robots.metal_follower.config_metal_follower import MetalFollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    r = make_robot_from_config(MetalFollowerConfig(port="can0"))
    assert r.name == "metal_follower"
    assert type(r).__name__ == "MetalFollower"


def test_startup_slow_sync_clamps_then_syncs():
    r = MetalFollower(
        MetalFollowerConfig(
            port="can0",
            startup_sync_speed_deg=3.0,
            startup_sync_tolerance_deg=3.0,
            velocity_feedforward=False,
        )
    )
    r.bus = MagicMock()
    r.bus.sync_read.return_value = {"shoulder_pan": 0.0}
    out = r.send_action({"shoulder_pan.pos": 50.0})
    assert out["shoulder_pan.pos"] == 3.0  # clamped to +3 deg/step from present 0
    assert r._synced is False
    r.bus.sync_read.return_value = {"shoulder_pan": 48.0}
    r.send_action({"shoulder_pan.pos": 50.0})  # err 2 <= tolerance 3 -> synced
    assert r._synced is True


def test_connect_sets_follow_gains(follower):
    follower.cameras = {}
    follower.bus.is_connected = False  # so @check_if_already_connected doesn't trip
    follower.connect(calibrate=False)
    kp_call = {m: kp for m, (kp, kd) in DEFAULT_GAINS.items()}
    kd_call = {m: kd for m, (kp, kd) in DEFAULT_GAINS.items()}
    follower.bus.sync_write.assert_any_call("Kp", kp_call)
    follower.bus.sync_write.assert_any_call("Kd", kd_call)
    assert follower._resolved_gains == DEFAULT_GAINS


def test_moving_goals_encode_filtered_velocity_in_mit_frame():
    follower, wire = _make_wired_follower()

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", side_effect=[1.0, 1.02]):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 10.0})

    frame = _last_frame(wire)
    # raw velocity is 500 deg/s, clipped to 200, then first EMA update is 0.35 * 200 = 70.
    assert _decode_dq_deg_s(follower, frame) == pytest.approx(70.0, abs=0.3)
    dq_uint = (frame.data[2] << 4) | (frame.data[3] >> 4)
    assert dq_uint != ((1 << 12) - 1) // 2


def test_velocity_feedforward_off_is_byte_identical_to_legacy_goal_write():
    follower, wire = _make_wired_follower(velocity_feedforward=False)

    follower.send_action({"shoulder_pan.pos": 10.0})
    follower_frame = bytes(_last_frame(wire).data)
    wire.reset_mock()
    follower.bus.sync_write("Goal_Position", {"shoulder_pan": 10.0})
    legacy_frame = bytes(_last_frame(wire).data)

    assert follower_frame == legacy_frame
    dq_uint = (legacy_frame[2] << 4) | (legacy_frame[3] >> 4)
    assert dq_uint == ((1 << 12) - 1) // 2


def test_velocity_feedforward_is_clipped_to_configured_maximum():
    follower, wire = _make_wired_follower(velocity_ff_alpha=1.0, velocity_ff_max_deg_s=40.0)

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", side_effect=[2.0, 2.01]):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 100.0})

    assert _decode_dq_deg_s(follower, _last_frame(wire)) == pytest.approx(40.0, abs=0.3)


def test_mit_frames_use_configured_follow_gains():
    gains = dict.fromkeys(DEFAULT_GAINS, (123.0, 1.25))
    follower, wire = _make_wired_follower(gains=gains)

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", return_value=3.0):
        follower.send_action({"shoulder_pan.pos": 5.0})

    frame = _last_frame(wire)
    assert _decode_gain(frame, MIT_KP_RANGE, "kp") == pytest.approx(123.0, abs=0.13)
    assert _decode_gain(frame, MIT_KD_RANGE, "kd") == pytest.approx(1.25, abs=0.002)


def _decode_tau_nm(follower, frame, motor):
    tau_uint = ((frame.data[6] & 0x0F) << 8) | frame.data[7]
    tmax = MOTOR_LIMIT_PARAMS[follower.bus._motor_types[motor]][2]
    return tau_uint / ((1 << 12) - 1) * (2.0 * tmax) - tmax


def _frame_for(wire, send_id):
    for call in wire.send.call_args_list:
        if call.args[0].arbitration_id == send_id:
            return call.args[0]
    raise AssertionError(f"no frame sent to CAN id 0x{send_id:02X}")


def test_mit_frames_carry_no_feedforward_torque():
    follower, wire = _make_wired_follower()

    follower.send_action({"shoulder_lift.pos": -40.0})

    frame = _last_frame(wire)
    tau_uint = ((frame.data[6] & 0x0F) << 8) | frame.data[7]
    assert tau_uint == ((1 << 12) - 1) // 2  # exact 0.0 encoding




def test_velocity_estimator_resets_after_long_action_gap():
    follower, wire = _make_wired_follower(velocity_ff_alpha=1.0)

    with patch(
        "lerobot.robots.metal_follower.metal_follower.time.perf_counter",
        side_effect=[4.0, 4.1, 4.7],
    ):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 10.0})
        moving_velocity = _decode_dq_deg_s(follower, _last_frame(wire))
        follower.send_action({"shoulder_pan.pos": 20.0})

    assert moving_velocity == pytest.approx(100.0, abs=0.3)
    assert _decode_dq_deg_s(follower, _last_frame(wire)) == pytest.approx(0.0, abs=0.3)
