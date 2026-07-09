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

"""Unit tests for the Hans Robot S30 adapter.

All tests run fully offline using a mock CPSClient – no physical robot
or network connection is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig
from lerobot.robots.hans_s30.cps_client import CPSClient, RobotFSM
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
FAKE_JOINT_POS = [10.0, -20.0, 30.0, 0.0, 45.5, -10.0]


def _make_cps_mock() -> MagicMock:
    """Build a CPSClient mock that simulates a healthy, connected robot."""
    cps = MagicMock(spec=CPSClient)
    cps.HRIF_IsConnected.return_value = True

    # All commands return 0 (success) by default.
    for method in [
        "HRIF_Connect",
        "HRIF_Electrify",
        "HRIF_Connect2Controller",
        "HRIF_GrpEnable",
        "HRIF_GrpDisable",
        "HRIF_DisConnect",
        "HRIF_SetOverride",
        "HRIF_GrpOpenFreeDriver",
        "HRIF_GrpCloseFreeDriver",
        "HRIF_MoveJ",
    ]:
        getattr(cps, method).return_value = 0

    # wait_for_fsm returns STANDBY immediately.
    cps.wait_for_fsm.return_value = int(RobotFSM.STANDBY)

    # ReadActJointPos fills result with fake joint values.
    def _read_joints(_box, _rbt, result):
        result.clear()
        result.extend([str(v) for v in FAKE_JOINT_POS])
        return 0

    cps.HRIF_ReadActJointPos.side_effect = _read_joints
    return cps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robot():
    """Return a HansS30 instance with a fully mocked CPSClient (no hardware)."""
    cfg = HansS30RobotConfig(ip="192.168.0.1", id="test_s30")

    with patch("lerobot.robots.hans_s30.hans_s30.CPSClient", return_value=_make_cps_mock()):
        r = HansS30(cfg)
        yield r
        if r.is_connected:
            r.disconnect()


# ---------------------------------------------------------------------------
# Tests: CPSClient protocol
# ---------------------------------------------------------------------------


class TestCPSClientProtocol:
    """Verify that CPSClient builds the correct TCP command strings."""

    def setup_method(self):
        self.cps = CPSClient()
        self.captured: list[str] = []

        class FakeChannel:
            def send_recv(self_, cmd, result):  # noqa: N805
                self.captured.append(cmd)
                return 0

        self.cps._channels[0] = FakeChannel()

    def test_grp_enable_command(self):
        self.cps.HRIF_GrpEnable(0, 0)
        assert self.captured[-1] == "GrpPowerOn,0,;"

    def test_grp_disable_command(self):
        self.cps.HRIF_GrpDisable(0, 0)
        assert self.captured[-1] == "GrpPowerOff,0,;"

    def test_set_override_command(self):
        self.cps.HRIF_SetOverride(0, 0, 0.5)
        assert self.captured[-1] == "SetOverride,0,0.5,;"

    def test_read_joint_pos_command(self):
        result: list = []
        self.cps.HRIF_ReadActJointPos(0, 0, result)
        assert self.captured[-1] == "ReadActACS,0,;"

    def test_move_j_builds_waypoint(self):
        self.cps.HRIF_MoveJ(0, 0, [10.0, -20.0, 30.0, 0.0, 45.0, -10.0])
        cmd = self.captured[-1]
        assert cmd.startswith("WayPoint,")
        assert "10.0,-20.0,30.0,0.0,45.0,-10.0" in cmd
        assert "TCP" in cmd
        assert "Base" in cmd
        # is_joint flag must be 1 for pure joint-space motion
        assert ",1," in cmd
        assert cmd.endswith(",;")

    def test_open_free_driver_command(self):
        self.cps.HRIF_GrpOpenFreeDriver(0, 0)
        assert self.captured[-1] == "GrpOpenFreeDriver,0,;"

    def test_raise_on_error_zero(self):
        CPSClient.raise_on_error(0)  # should not raise

    def test_raise_on_error_nonzero(self):
        with pytest.raises(RuntimeError, match="39504"):
            CPSClient.raise_on_error(39504, "connect")


# ---------------------------------------------------------------------------
# Tests: HansS30 Robot interface
# ---------------------------------------------------------------------------


class TestHansS30Config:
    def test_default_config_values(self):
        cfg = HansS30RobotConfig(ip="10.0.0.1", id="arm")
        assert cfg.port == 10003
        assert cfg.velocity == 50.0
        assert cfg.speed_override == 0.5
        assert cfg.tcp_name == "TCP"
        assert cfg.type == "hans_s30"

    def test_custom_config(self):
        cfg = HansS30RobotConfig(ip="192.168.1.1", port=10004, velocity=30.0, id="custom")
        assert cfg.port == 10004
        assert cfg.velocity == 30.0


class TestHansS30Features:
    def test_observation_features_keys(self, robot):
        keys = set(robot.observation_features.keys())
        assert keys == {f"{j}.pos" for j in JOINT_NAMES}

    def test_action_features_keys(self, robot):
        keys = set(robot.action_features.keys())
        assert keys == {f"{j}.pos" for j in JOINT_NAMES}

    def test_observation_features_types(self, robot):
        for key, typ in robot.observation_features.items():
            assert typ is float, f"{key} should be float"

    def test_is_calibrated_always_true(self, robot):
        assert robot.is_calibrated is True

    def test_calibrate_is_noop(self, robot):
        robot.calibrate()  # must not raise

    def test_name(self, robot):
        assert robot.name == "hans_s30"


class TestHansS30Connection:
    def test_initial_state_disconnected(self, robot):
        assert not robot.is_connected

    def test_connect_sets_connected(self, robot):
        robot.connect()
        assert robot.is_connected

    def test_disconnect_clears_connected(self, robot):
        robot.connect()
        robot.disconnect()
        assert not robot.is_connected

    def test_connect_calls_correct_sequence(self, robot):
        robot.connect()
        cps = robot._cps
        cps.HRIF_Connect.assert_called_once()
        cps.HRIF_Electrify.assert_called_once()
        cps.HRIF_Connect2Controller.assert_called_once()
        cps.HRIF_GrpEnable.assert_called_once()
        cps.HRIF_SetOverride.assert_called_once()

    def test_disconnect_calls_disable_and_close(self, robot):
        robot.connect()
        robot.disconnect()
        cps = robot._cps
        cps.HRIF_GrpDisable.assert_called_once()
        cps.HRIF_DisConnect.assert_called_once()

    def test_double_connect_raises(self, robot):
        robot.connect()
        with pytest.raises(DeviceAlreadyConnectedError):
            robot.connect()

    def test_get_observation_without_connect_raises(self, robot):
        with pytest.raises(DeviceNotConnectedError):
            robot.get_observation()

    def test_send_action_without_connect_raises(self, robot):
        action = {f"{j}.pos": 0.0 for j in JOINT_NAMES}
        with pytest.raises(DeviceNotConnectedError):
            robot.send_action(action)


class TestHansS30Observation:
    def test_get_observation_returns_six_joints(self, robot):
        robot.connect()
        obs = robot.get_observation()
        assert set(obs.keys()) == {f"{j}.pos" for j in JOINT_NAMES}

    def test_get_observation_values_are_floats(self, robot):
        robot.connect()
        obs = robot.get_observation()
        for key, val in obs.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_get_observation_values_match_mock(self, robot):
        robot.connect()
        obs = robot.get_observation()
        for i, j in enumerate(JOINT_NAMES):
            assert obs[f"{j}.pos"] == pytest.approx(FAKE_JOINT_POS[i])


class TestHansS30Action:
    def test_send_action_returns_dict_with_correct_keys(self, robot):
        robot.connect()
        action = {f"{j}.pos": float(i * 10) for i, j in enumerate(JOINT_NAMES, 1)}
        returned = robot.send_action(action)
        assert set(returned.keys()) == {f"{j}.pos" for j in JOINT_NAMES}

    def test_send_action_calls_move_j(self, robot):
        robot.connect()
        action = {f"{j}.pos": float(i * 5) for i, j in enumerate(JOINT_NAMES, 1)}
        robot.send_action(action)
        robot._cps.HRIF_MoveJ.assert_called_once()

    def test_send_action_passes_correct_joint_values(self, robot):
        robot.connect()
        target = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        action = {f"{j}.pos": target[i] for i, j in enumerate(JOINT_NAMES)}
        robot.send_action(action)

        call_args = robot._cps.HRIF_MoveJ.call_args
        sent_joints = call_args.kwargs.get("acs_pos") or call_args.args[2]
        assert sent_joints == pytest.approx(target)

    def test_send_action_with_max_relative_target_clamps(self):
        """When max_relative_target is set, large jumps must be clamped."""
        cfg = HansS30RobotConfig(ip="0.0.0.0", id="clamp_test", max_relative_target=5.0)

        with patch("lerobot.robots.hans_s30.hans_s30.CPSClient", return_value=_make_cps_mock()):
            r = HansS30(cfg)
            r.connect()

            # Current position from mock: FAKE_JOINT_POS = [10, -20, 30, 0, 45.5, -10]
            # Request a 50-degree jump on joint_1 → should be clamped to +5
            action = {f"{j}.pos": FAKE_JOINT_POS[i] + 50.0 for i, j in enumerate(JOINT_NAMES)}
            returned = robot.send_action(action) if False else r.send_action(action)

            for j in JOINT_NAMES:
                i = JOINT_NAMES.index(j)
                assert returned[f"{j}.pos"] == pytest.approx(FAKE_JOINT_POS[i] + 5.0)

            r.disconnect()


class TestHansS30FreeDriver:
    def test_enable_free_driver(self, robot):
        robot.connect()
        robot.enable_free_driver()
        robot._cps.HRIF_GrpOpenFreeDriver.assert_called_once()

    def test_disable_free_driver(self, robot):
        robot.connect()
        robot.disable_free_driver()
        robot._cps.HRIF_GrpCloseFreeDriver.assert_called_once()

    def test_free_driver_without_connect_raises(self, robot):
        with pytest.raises(DeviceNotConnectedError):
            robot.enable_free_driver()


# ---------------------------------------------------------------------------
# Tests: RobotFSM enum
# ---------------------------------------------------------------------------


class TestRobotFSM:
    def test_standby_value(self):
        assert int(RobotFSM.STANDBY) == 33

    def test_free_driver_value(self):
        assert int(RobotFSM.FREE_DRIVER) == 31

    def test_disabled_value(self):
        assert int(RobotFSM.DISABLED) == 24

    def test_moving_value(self):
        assert int(RobotFSM.MOVING) == 25
