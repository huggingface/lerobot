#!/usr/bin/env python

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

import hashlib
import math
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.metal_leader import urdf as urdf_module
from lerobot.teleoperators.metal_leader.config_metal_leader import MetalLeaderConfig
from lerobot.teleoperators.metal_leader.gravity import MetalGravityModel
from lerobot.teleoperators.metal_leader.gripper_friction import (
    STATIC_FRICTION_NM,
    STOP_TORQUE_NM,
    VISCOUS_COEFFICIENT,
    gripper_friction_torque,
)
from lerobot.teleoperators.metal_leader.metal_leader import (
    ARM_JOINT_NAMES,
    GRIPPER_NAME,
    MOTOR_MODELS,
    MetalLeader,
)
from lerobot.utils.import_utils import _pinocchio_available


def _leader(**overrides) -> MetalLeader:
    """A leader with a mocked bus and a stub gravity model, ready for `_gravity_tick`."""
    leader = MetalLeader(MetalLeaderConfig(port="can0", **overrides))
    leader.bus = MagicMock()
    leader.bus.sync_read.return_value = dict.fromkeys(leader._joint_motor_names, 0.0)
    leader.bus.sync_read_all_states.return_value = {
        motor: {"position": 0.0, "velocity": 0.0, "torque": 0.0} for motor in leader._joint_motor_names
    }
    leader._gravity = MagicMock()
    leader._gravity.feedforward_torque.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    leader._gravity.blended_feedforward_torque.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return leader


def _written_commands(leader: MetalLeader) -> dict[str, tuple[float, float, float, float, float]]:
    return leader.bus.sync_write_metal.call_args.args[0]


# ── Motor tables ──────────────────────────────────────────────────────────


def test_has_all_seven_motors():
    assert _leader()._joint_motor_names == [*ARM_JOINT_NAMES, GRIPPER_NAME]


def test_action_features_match_the_follower():
    from lerobot.robots.metal_follower.config_metal_follower import MetalFollowerConfig
    from lerobot.robots.metal_follower.metal_follower import MetalFollower

    follower = MetalFollower(MetalFollowerConfig(port="can1"))
    assert _leader().action_features == follower.action_features


def test_motor_models_match_the_follower():
    """The leader duplicates MOTOR_MODELS rather than importing it from `robots/` (that would
    invert the package layering). This test is what keeps the copy honest."""
    from lerobot.robots.metal_follower.metal_follower import MOTOR_MODELS as FOLLOWER_MOTOR_MODELS

    assert MOTOR_MODELS == FOLLOWER_MOTOR_MODELS


def test_motor_can_ids_match_the_follower():
    from lerobot.robots.metal_follower.config_metal_follower import MetalFollowerConfig

    assert MetalLeaderConfig().motor_can_ids == MetalFollowerConfig().motor_can_ids


# ── Transport guard ───────────────────────────────────────────────────────


def test_slcan_is_accepted():
    """slcan is the only CAN transport available on macOS/Windows, where SocketCAN does not exist.

    Measured on a Metal arm over a CANable: a full gravity tick (28 frames) loses 0 replies and
    runs ~4.2 ms p50, so the transport carries the load -- what it cannot carry is gravity_hz=200,
    which `test_slcan_warns_above_100_hz` covers.
    """
    leader = MetalLeader(MetalLeaderConfig(port="/dev/ttyACM0", can_interface="slcan", gravity_hz=100))
    assert leader.bus.can_interface == "slcan"


def test_unknown_can_interface_is_rejected():
    with pytest.raises(ValueError, match="socketcan"):
        MetalLeader(MetalLeaderConfig(port="can0", can_interface="pcan"))


def test_slcan_warns_above_100_hz(caplog):
    with caplog.at_level("WARNING"):
        MetalLeader(MetalLeaderConfig(port="/dev/ttyACM0", can_interface="slcan", gravity_hz=200))
    assert "gravity_hz=200" in caplog.text


def test_slcan_does_not_warn_at_100_hz(caplog):
    with caplog.at_level("WARNING"):
        MetalLeader(MetalLeaderConfig(port="/dev/ttyACM0", can_interface="slcan", gravity_hz=100))
    assert "gravity_hz" not in caplog.text


def test_partial_arm_is_rejected():
    ids = dict(MetalLeaderConfig().motor_can_ids)
    del ids["wrist_roll"]
    with pytest.raises(ValueError, match="wrist_roll"):
        MetalLeader(MetalLeaderConfig(port="can0", motor_can_ids=ids))


# ── Gravity tick wire format ──────────────────────────────────────────────


def test_gravity_tick_commands_zero_kp_on_every_motor():
    """kp must be 0 everywhere: any stiffness would pull the arm toward a position and fight the
    operator instead of just holding its weight."""
    leader = _leader()
    leader._gravity_tick()

    commands = _written_commands(leader)
    assert len(commands) == 7
    assert all(kp == 0.0 for kp, _, _, _, _ in commands.values())


def test_gravity_tick_streams_the_model_torque_per_joint():
    leader = _leader()
    leader._gravity_tick()

    commands = _written_commands(leader)
    torques = [commands[motor][4] for motor in ARM_JOINT_NAMES]
    assert torques == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_gravity_tick_commands_the_present_position_with_zero_velocity():
    leader = _leader()
    leader.bus.sync_read_all_states.return_value = {
        motor: {"position": 12.5, "velocity": 0.0, "torque": 0.0} for motor in leader._joint_motor_names
    }
    leader._gravity_tick()

    for _kp, _kd, position, velocity, _torque in _written_commands(leader).values():
        assert position == 12.5
        assert velocity == 0.0


def test_gripper_is_backdrivable_with_its_own_friction_torque():
    leader = _leader()
    states = leader.bus.sync_read_all_states.return_value
    states[GRIPPER_NAME] = {"position": 30.0, "velocity": 0.0, "torque": 0.0}
    leader._gravity_tick()

    kp, _kd, _position, _velocity, torque = _written_commands(leader)[GRIPPER_NAME]
    assert kp == 0.0
    # At rest the gripper gets the stiction-breaking stop torque, not a model torque.
    assert torque == pytest.approx(STOP_TORQUE_NM)


def test_gripper_friction_can_be_disabled():
    leader = _leader(gripper_friction_scale=0.0)
    leader._gravity_tick()

    assert _written_commands(leader)[GRIPPER_NAME][4] == 0.0


def test_leader_kd_accepts_a_scalar_or_a_per_joint_dict():
    scalar = _leader(leader_kd=0.7)
    scalar._gravity_tick()
    assert all(kd == 0.7 for _, kd, _, _, _ in _written_commands(scalar).values())

    per_joint = _leader(leader_kd={"shoulder_lift": 1.5})
    per_joint._gravity_tick()
    commands = _written_commands(per_joint)
    assert commands["shoulder_lift"][1] == 1.5
    # Motors absent from the dict fall back to 0 rather than inheriting another joint's value.
    assert commands["shoulder_pan"][1] == 0.0


# ── Velocity feedforward ──────────────────────────────────────────────────


def test_velocity_feedforward_off_uses_gravity_only():
    leader = _leader(use_velocity_feedforward=False)
    leader.bus.sync_read_all_states.return_value = {
        motor: {"position": 0.0, "velocity": 90.0, "torque": 0.0} for motor in leader._joint_motor_names
    }
    leader._gravity_tick()

    leader._gravity.blended_feedforward_torque.assert_not_called()
    _q, dq = leader._gravity.feedforward_torque.call_args.args
    assert dq == [0.0] * 6


def test_velocity_deadzone_zeroes_small_measured_velocities():
    """Motor velocity noise at rest would otherwise make the friction term chatter."""
    leader = _leader(velocity_deadzone_rad_s=0.05)
    below = math.degrees(0.04)
    above = math.degrees(0.5)
    states = leader.bus.sync_read_all_states.return_value
    states["shoulder_pan"] = {"position": 0.0, "velocity": below, "torque": 0.0}
    states["shoulder_lift"] = {"position": 0.0, "velocity": above, "torque": 0.0}
    leader._gravity_tick()

    _q, dq, _scales = leader._gravity.blended_feedforward_torque.call_args.args
    assert dq[0] == 0.0
    assert dq[1] == pytest.approx(0.5)


def test_friction_scale_is_passed_per_joint_in_urdf_order():
    leader = _leader(friction_scale={"shoulder_lift": 3.3, "wrist_yaw": 0.3})
    leader._gravity_tick()

    _q, _dq, scales = leader._gravity.blended_feedforward_torque.call_args.args
    assert scales == [0.0, 3.3, 0.0, 0.0, 0.3, 0.0]


def test_default_friction_scales_are_the_tuned_values():
    assert MetalLeaderConfig().friction_scale == {
        "shoulder_pan": 1.4,
        "shoulder_lift": 3.3,
        "elbow_flex": 1.1,
        "wrist_flex": 0.7,
        "wrist_yaw": 0.3,
        "wrist_roll": 0.7,
    }


def test_a_failing_tick_does_not_propagate():
    """The gravity thread must survive a dropped CAN reply: losing it would silently drop the arm
    out of compensation while it stays powered."""
    leader = _leader()
    leader.bus.sync_read_all_states.side_effect = RuntimeError("packet drop")
    leader._gravity_tick()  # must not raise


# ── Gripper friction model ────────────────────────────────────────────────


def test_gripper_friction_is_symmetric_about_zero_velocity():
    assert gripper_friction_torque(1.0) == pytest.approx(-gripper_friction_torque(-1.0))


def test_gripper_friction_applies_coulomb_in_the_direction_of_travel():
    assert gripper_friction_torque(0.5) == pytest.approx(0.5 * VISCOUS_COEFFICIENT + STATIC_FRICTION_NM)
    assert gripper_friction_torque(-0.5) == pytest.approx(-0.5 * VISCOUS_COEFFICIENT - STATIC_FRICTION_NM)


def test_gripper_friction_clamps_the_viscous_term():
    assert gripper_friction_torque(50.0) == gripper_friction_torque(3.0)


# ── Lifecycle ─────────────────────────────────────────────────────────────


def test_connect_does_not_energize_the_arm_if_the_gravity_model_fails():
    """The arm must never be left powered with no gravity thread behind it."""
    leader = _leader()
    leader.bus.is_connected = False

    with (
        patch.object(urdf_module, "metal_urdf_path", side_effect=urdf_module.MetalUrdfError("offline")),
        patch(
            "lerobot.teleoperators.metal_leader.metal_leader.metal_urdf_path",
            side_effect=urdf_module.MetalUrdfError("offline"),
        ),
        pytest.raises(urdf_module.MetalUrdfError),
    ):
        leader.connect()

    leader.bus.enable_torque.assert_not_called()
    leader.bus.disconnect.assert_called_once_with(disable_torque=False)
    assert leader._gravity_thread is None


def test_disconnect_holds_the_pose_instead_of_dropping_the_arm():
    leader = _leader(hold_kp_on_disconnect=50.0, hold_kd_on_disconnect=1.0)
    leader.bus.is_connected = True
    leader.bus.sync_read.return_value = dict.fromkeys(leader._joint_motor_names, 20.0)

    leader.disconnect()

    commands = _written_commands(leader)
    assert all(kp == 50.0 and kd == 1.0 for kp, kd, _, _, _ in commands.values())
    assert all(position == 20.0 for _, _, position, _, _ in commands.values())
    # Torque stays on, otherwise the hold is meaningless and the arm free-falls.
    leader.bus.disconnect.assert_called_once_with(disable_torque=False)


def test_disconnect_can_leave_the_arm_limp():
    leader = _leader(hold_kp_on_disconnect=0.0)
    leader.bus.is_connected = True

    leader.disconnect()

    leader.bus.sync_write_metal.assert_not_called()


# ── URDF fetch and cache ──────────────────────────────────────────────────

_FAKE_URDF = b"<robot name='metal'/>"
_FAKE_SHA = hashlib.sha256(_FAKE_URDF).hexdigest()


def test_urdf_is_downloaded_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(urdf_module, "METAL_URDF_SHA256", _FAKE_SHA)
    download = MagicMock(return_value=_FAKE_URDF)
    monkeypatch.setattr(urdf_module, "_download", download)

    path = urdf_module.metal_urdf_path(tmp_path)
    assert path.read_bytes() == _FAKE_URDF

    # Second call is served from cache, with no network access.
    assert urdf_module.metal_urdf_path(tmp_path) == path
    download.assert_called_once()


def test_a_corrupt_cache_entry_is_refetched(tmp_path, monkeypatch):
    monkeypatch.setattr(urdf_module, "METAL_URDF_SHA256", _FAKE_SHA)
    cached = tmp_path / urdf_module.METAL_URDF_FILENAME
    cached.write_bytes(b"truncated")
    monkeypatch.setattr(urdf_module, "_download", MagicMock(return_value=_FAKE_URDF))

    assert urdf_module.metal_urdf_path(tmp_path).read_bytes() == _FAKE_URDF


def test_a_bad_download_is_rejected_rather_than_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(urdf_module, "METAL_URDF_SHA256", _FAKE_SHA)
    monkeypatch.setattr(urdf_module, "_download", MagicMock(return_value=b"something else"))

    with pytest.raises(urdf_module.MetalUrdfError, match="checksum mismatch"):
        urdf_module.metal_urdf_path(tmp_path)

    assert not (tmp_path / urdf_module.METAL_URDF_FILENAME).exists()


def test_urdf_url_points_at_the_six_joint_variant():
    """The `metal_description` package models the gripper jaws as prismatic joints (nq=8) and
    would raise inside Pinocchio on every tick; only the SDK example variant has nq=6."""
    assert "metal_sdk/example/urdf" in urdf_module.METAL_URDF_URL
    assert "metal_description" not in urdf_module.METAL_URDF_URL


def test_urdf_url_is_pinned_to_a_commit():
    """A branch ref would let an upstream edit silently retune the arm's dynamics."""
    assert len(urdf_module.METAL_URDF_COMMIT) == 40
    assert urdf_module.METAL_URDF_COMMIT in urdf_module.METAL_URDF_URL


# ── Gravity model (requires pinocchio, i.e. `lerobot[metal]`) ─────────────

_ONE_JOINT_URDF = """<?xml version="1.0"?>
<robot name="stub">
  <link name="base"/>
  <link name="tip">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="tip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


@pytest.mark.skipif(not _pinocchio_available, reason="pinocchio is only installed by lerobot[metal]")
def test_gravity_model_rejects_a_urdf_with_the_wrong_joint_count(tmp_path):
    """Guards against the `metal_description` variant, whose prismatic gripper jaws give nq=8 and
    would otherwise raise deep inside Pinocchio on every control tick."""
    stub = tmp_path / "stub.urdf"
    stub.write_text(_ONE_JOINT_URDF)

    with pytest.raises(ValueError, match="expected exactly 6"):
        MetalGravityModel(str(stub))


@pytest.mark.skipif(not _pinocchio_available, reason="pinocchio is only installed by lerobot[metal]")
def test_real_urdf_builds_a_six_joint_model():
    """End-to-end over the real description: fetches (or reuses the cache) and builds the model.
    Downloads ~8.5 KB once, then runs offline."""
    model = MetalGravityModel(str(urdf_module.metal_urdf_path()))

    assert model.model.nq == 6
    assert list(model.model.names)[1:] == ["JOINT1", "JOINT2", "JOINT3", "JOINT4", "JOINT5", "JOINT6"]


@pytest.mark.skipif(not _pinocchio_available, reason="pinocchio is only installed by lerobot[metal]")
def test_real_model_produces_finite_torques_across_the_workspace():
    model = MetalGravityModel(str(urdf_module.metal_urdf_path()))

    for shoulder_lift_deg in (0.0, -45.0, -90.0, -135.0, -180.0):
        torques = model.feedforward_torque(
            [0.0, math.radians(shoulder_lift_deg), 0.0, 0.0, 0.0, 0.0], [0.0] * 6
        )
        assert len(torques) == 6
        assert all(math.isfinite(t) for t in torques)
        # A ~4 kg arm cannot need tens of N·m; anything larger means a unit or model error.
        assert all(abs(t) < 20.0 for t in torques)


@pytest.mark.skipif(not _pinocchio_available, reason="pinocchio is only installed by lerobot[metal]")
def test_zero_friction_scale_gives_gravity_only():
    """A joint scaled to 0 must be unaffected by how fast it is moving."""
    model = MetalGravityModel(str(urdf_module.metal_urdf_path()))
    q = [0.0, math.radians(-90.0), 0.0, 0.0, 0.0, 0.0]

    at_rest = model.feedforward_torque(q, [0.0] * 6)
    moving = model.blended_feedforward_torque(q, [1.0] * 6, [0.0] * 6)

    assert moving == pytest.approx(at_rest)


@pytest.mark.skipif(not _pinocchio_available, reason="pinocchio is only installed by lerobot[metal]")
def test_friction_scale_interpolates_between_gravity_and_full_compensation():
    model = MetalGravityModel(str(urdf_module.metal_urdf_path()))
    q = [0.0, math.radians(-90.0), 0.0, 0.0, 0.0, 0.0]
    dq = [0.5] * 6

    gravity_only = model.feedforward_torque(q, [0.0] * 6)
    full = model.feedforward_torque(q, dq)
    half = model.blended_feedforward_torque(q, dq, [0.5] * 6)

    expected = [g + 0.5 * (f - g) for g, f in zip(gravity_only, full, strict=True)]
    assert half == pytest.approx(expected)


# ── Factory ───────────────────────────────────────────────────────────────


def test_factory_builds_metal_leader():
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    leader = make_teleoperator_from_config(MetalLeaderConfig(port="can0"))
    assert leader.name == "metal_leader"
    assert type(leader).__name__ == "MetalLeader"


def test_factory_builds_bi_metal_leader():
    from lerobot.teleoperators.bi_metal_leader import BiMetalLeaderConfig
    from lerobot.teleoperators.metal_leader import MetalLeaderConfigBase
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    leader = make_teleoperator_from_config(
        BiMetalLeaderConfig(
            left_arm_config=MetalLeaderConfigBase(port="can0"),
            right_arm_config=MetalLeaderConfigBase(port="can1"),
        )
    )
    assert leader.name == "bi_metal_leader"
    assert leader.left_arm.config.port == "can0"
    assert leader.right_arm.config.port == "can1"


def test_bimanual_action_keys_are_side_prefixed():
    from lerobot.teleoperators.bi_metal_leader import BiMetalLeader, BiMetalLeaderConfig
    from lerobot.teleoperators.metal_leader import MetalLeaderConfigBase

    leader = BiMetalLeader(
        BiMetalLeaderConfig(
            left_arm_config=MetalLeaderConfigBase(port="can0"),
            right_arm_config=MetalLeaderConfigBase(port="can1"),
        )
    )
    assert "left_shoulder_pan.pos" in leader.action_features
    assert "right_gripper.pos" in leader.action_features
    assert len(leader.action_features) == 14


def test_port_is_required():
    with pytest.raises(ValueError, match="requires `port`"):
        MetalLeader(MetalLeaderConfig())


def test_defaults_to_slcan_at_100_hz():
    """gravity_hz defaults to 100, not 200: a full tick measures ~4.8 ms p50 over slcan, which
    fits a 10 ms period but not 200 Hz's 5 ms. Keeps the default config self-consistent."""
    config = MetalLeaderConfig()
    assert config.can_interface == "slcan"
    assert config.port is None
    assert config.gravity_hz == 100


def test_default_config_does_not_warn(caplog):
    """The shipped defaults must not trip the slcan/gravity_hz warning."""
    with caplog.at_level("WARNING"):
        MetalLeader(MetalLeaderConfig(port="/dev/ttyACM0"))
    assert "gravity_hz" not in caplog.text
