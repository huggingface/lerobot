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

"""Tests for the UnitreeG1 robot class.

The Unitree SDK, the cameras and the arm IK are all mocked, so these run without hardware
or the SDK installed. Pure helper/config tests live in ``test_unitree_g1_utils.py``.
"""

import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)

TOKEN_DIM = 64

# ---------------------------------------------------------------------------
# Controller stubs
# ---------------------------------------------------------------------------


class TokenController:
    """Stand-in for SonicWholeBodyController: declares its own action and proprio spaces."""

    control_dt = 0.02

    def __init__(self, dim: int = TOKEN_DIM):
        self.action_ft = {f"motion_token.{i}.pos": float for i in range(dim)}
        self.observation_ft = {f"motion_token_state.{i}.pos": float for i in range(dim)}
        self.kp = np.full(NUM_MOTORS, 100.0, np.float32)
        self.kd = np.full(NUM_MOTORS, 2.0, np.float32)
        self.default_angles = np.zeros(NUM_MOTORS, np.float32)
        self.reset_calls = 0

    def run_step(self, action: dict, lowstate) -> dict:
        return {}

    def reset(self) -> None:
        self.reset_calls += 1

    def observation_state(self) -> dict[str, float]:
        return dict.fromkeys(self.observation_ft, 0.0)


class LocomotionOnlyController:
    """Stand-in for GR00T/Holosoma: declares no optional features, so the robot's defaults apply."""

    control_dt = 0.02

    def run_step(self, action: dict, lowstate) -> dict:
        return {}

    def reset(self) -> None:
        pass


def make_camera_config(width: int = 640, height: int = 480, use_depth: bool = False):
    """Duck-typed camera config.

    ``_cameras_ft`` reads the resolution/rgb/depth attributes, and ``RobotConfig`` validates
    that width, height and fps are all set.
    """
    return SimpleNamespace(width=width, height=height, fps=30, use_rgb=True, use_depth=use_depth)


# ---------------------------------------------------------------------------
# SDK mocks
# ---------------------------------------------------------------------------


def _make_lowstate_msg_mock():
    """Create a mock that mimics the SDK LowState_ message."""
    msg = MagicMock()
    msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
        idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
    )
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = 0
    return msg


def _make_sdk_mocks():
    """Create mocks for the Unitree SDK modules used by UnitreeG1."""
    lowcmd_default = MagicMock()
    lowcmd_default.mode_pr = 0
    lowcmd_default.motor_cmd = [MagicMock() for _ in range(35)]

    crc_mock = MagicMock()
    crc_mock.Crc.return_value = 0

    lowstate_msg = _make_lowstate_msg_mock()

    subscriber_mock = MagicMock()
    subscriber_mock.Read.return_value = lowstate_msg

    return {
        "lowcmd_default": lowcmd_default,
        "crc_mock": crc_mock,
        "subscriber_mock": subscriber_mock,
        "publisher_mock": MagicMock(),
        "lowstate_msg": lowstate_msg,
    }


@pytest.fixture
def make_robot():
    """Factory for a UnitreeG1 with every hardware dependency mocked out.

    Controllers are injected directly rather than resolved by name, so no controller
    checkpoint is ever downloaded.
    """
    mocks = _make_sdk_mocks()
    module = "lerobot.robots.unitree_g1.unitree_g1"

    with ExitStack() as stack:
        # require_package would refuse to build the robot without the Unitree SDK installed.
        stack.enter_context(patch(f"{module}.require_package", MagicMock()))
        stack.enter_context(
            patch(
                f"{module}.make_cameras_from_configs",
                lambda cfgs: {name: MagicMock(is_connected=True) for name in cfgs},
            )
        )
        stack.enter_context(patch(f"{module}.G1_29_ArmIK", MagicMock()))
        stack.enter_context(patch(f"{module}._SDKChannelFactoryInitialize", MagicMock()))
        stack.enter_context(
            patch(f"{module}._SDKChannelPublisher", MagicMock(return_value=mocks["publisher_mock"]))
        )
        stack.enter_context(
            patch(f"{module}._SDKChannelSubscriber", MagicMock(return_value=mocks["subscriber_mock"]))
        )
        stack.enter_context(
            patch(f"{module}.unitree_hg_msg_dds__LowCmd_", MagicMock(return_value=mocks["lowcmd_default"]))
        )
        stack.enter_context(patch(f"{module}.hg_LowCmd", MagicMock))
        stack.enter_context(patch(f"{module}.hg_LowState", MagicMock))
        stack.enter_context(patch(f"{module}.CRC", MagicMock(return_value=mocks["crc_mock"])))

        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        built = []

        def _factory(controller=None, cameras=None, **config_kwargs):
            cfg = UnitreeG1Config(
                is_simulation=True,
                gravity_compensation=False,
                cameras=cameras or {},
                **config_kwargs,
            )
            robot = UnitreeG1(cfg)
            if controller is not None:
                robot.controller = controller
                # observation_features/action_features are cached_property; drop any cached value
                # so the injected controller is taken into account.
                robot.__dict__.pop("observation_features", None)
                robot.__dict__.pop("action_features", None)
            built.append(robot)
            return robot

        yield _factory, mocks

        for robot in built:
            if robot.is_connected:
                robot.disconnect()


def arm_for_publish(robot, mocks, kp_value: float = 50.0, kd_value: float = 1.0):
    """Attach the state that ``connect()`` would normally set up, for publish-only tests."""
    robot.msg = mocks["lowcmd_default"]
    robot.crc = mocks["crc_mock"]
    robot.lowcmd_publisher = mocks["publisher_mock"]
    robot.kp = np.full(NUM_MOTORS, kp_value, np.float32)
    robot.kd = np.full(NUM_MOTORS, kd_value, np.float32)
    for cmd in robot.msg.motor_cmd:
        cmd.q = -1.0  # sentinel: untouched joints keep this value
    return robot


def published_targets(robot):
    return {motor.name: robot.msg.motor_cmd[motor.value].q for motor in G1_29_JointIndex}


def hardware_mode(robot):
    """Switch to the real-robot branch after construction.

    Building with ``is_simulation=False`` would make ``__init__`` import the ZMQ bridge, and
    pyzmq is not installed in the test environment, so the flag is flipped once the robot exists.
    """
    robot.config.is_simulation = False
    return robot


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_disconnected(self, make_robot):
        factory, _ = make_robot
        robot = factory()
        assert not robot.is_connected
        assert robot.controller is None

    def test_observation_empty_before_connect(self, make_robot):
        factory, _ = make_robot
        assert factory().get_observation() == {}

    def test_disconnect_is_safe_before_connect(self, make_robot):
        factory, _ = make_robot
        factory().disconnect()

    def test_name_and_config_class(self, make_robot):
        factory, _ = make_robot
        robot = factory()
        assert robot.name == "unitree_g1"
        assert robot.config_class is UnitreeG1Config

    def test_stubs_satisfy_protocol(self):
        """Controllers are duck-typed against a runtime-checkable Protocol."""
        from lerobot.robots.unitree_g1.unitree_g1 import RobotController

        assert isinstance(TokenController(), RobotController)
        assert isinstance(LocomotionOnlyController(), RobotController)


class TestMakeRobotController:
    def test_none_disables_controller(self):
        from lerobot.robots.unitree_g1.unitree_g1 import make_robot_controller

        assert make_robot_controller(None) is None

    def test_unknown_name_raises(self):
        from lerobot.robots.unitree_g1.unitree_g1 import make_robot_controller

        with pytest.raises(ValueError, match="Unknown controller") as excinfo:
            make_robot_controller("NotAController")
        # The error should tell the user what they can pick instead.
        assert "SonicWholeBodyController" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------


class TestObservationFeatures:
    def test_no_controller_joint_positions(self, make_robot):
        factory, _ = make_robot
        features = factory().observation_features
        assert len(features) == NUM_MOTORS
        assert set(features) == {f"{joint.name}.q" for joint in G1_29_JointIndex}

    def test_whole_body_replaces_state(self, make_robot):
        """Regression: merging the 29 joint keys with the 64-D token gave a 93-D state and
        broke inference against a checkpoint trained on the token echo alone."""
        factory, _ = make_robot
        controller = TokenController()
        features = factory(controller=controller).observation_features

        assert len(features) == TOKEN_DIM
        assert set(features) == set(controller.observation_ft)
        assert not [key for key in features if key.endswith(".q")]

    def test_locomotion_keeps_joint_state(self, make_robot):
        """Controllers that declare no proprio features must not lose the joint keys."""
        factory, _ = make_robot
        features = factory(controller=LocomotionOnlyController()).observation_features
        assert set(features) == {f"{joint.name}.q" for joint in G1_29_JointIndex}

    def test_raw_proprio_takes_the_state_back(self, make_robot):
        """A policy trained on joint angles while a token controller drove the body wants the
        robot's own proprio, hands included -- the opposite of the echo above, and not
        deducible from the controller name."""
        factory, _ = make_robot
        robot = factory(controller=TokenController(), raw_proprio=True, grippers=True)

        assert list(robot.observation_features) == [
            *(f"{joint.name}.q" for joint in G1_29_JointIndex),
            "left_gripper.pos",
            "right_gripper.pos",
        ]

    def test_gripper_state_latches_the_last_command(self, make_robot):
        """The CAN hands report no position, and the recordings never had one: their gripper
        state is the previous command verbatim."""
        factory, _ = make_robot
        robot = factory(controller=TokenController(), raw_proprio=True, grippers=True)

        assert robot._gripper_obs == {"left_gripper.pos": 0.0, "right_gripper.pos": 0.0}

        robot._record_gripper_obs({"left_gripper.pos": 1.0, "right_gripper.pos": 0.25})
        assert robot._gripper_obs["left_gripper.pos"] == 1.0
        assert robot._gripper_obs["right_gripper.pos"] == 0.25

        # A token-only action leaves the hands holding their last command.
        robot._record_gripper_obs({"motion_token.0.pos": 0.5})
        assert robot._gripper_obs["left_gripper.pos"] == 1.0

        robot._record_gripper_obs({"left_gripper.pos": 1.7})
        assert robot._gripper_obs["left_gripper.pos"] == 1.0

    def test_cameras_are_added_alongside_state(self, make_robot):
        factory, _ = make_robot
        robot = factory(controller=TokenController(), cameras={"ego_view": make_camera_config()})
        features = robot.observation_features

        assert len(features) == TOKEN_DIM + 1
        assert features["ego_view"] == (480, 640, 3)

    def test_depth_camera_adds_a_second_entry(self, make_robot):
        factory, _ = make_robot
        robot = factory(cameras={"ego_view": make_camera_config(use_depth=True)})
        features = robot.observation_features

        assert features["ego_view"] == (480, 640, 3)
        assert features["ego_view_depth"] == (480, 640, 1)


class TestActionFeatures:
    def test_no_controller_is_full_joint_teleop(self, make_robot):
        factory, _ = make_robot
        features = factory().action_features
        assert set(features) == {f"{joint.name}.q" for joint in G1_29_JointIndex}

    def test_whole_body_owns_action_space(self, make_robot):
        factory, _ = make_robot
        controller = TokenController()
        features = factory(controller=controller).action_features

        assert len(features) == TOKEN_DIM
        assert set(features) == set(controller.action_ft)

    def test_locomotion_gets_arms_joystick(self, make_robot):
        factory, _ = make_robot
        features = factory(controller=LocomotionOnlyController()).action_features

        expected = {f"{joint.name}.q" for joint in G1_29_JointArmIndex} | set(REMOTE_AXES)
        assert set(features) == expected
        assert len(features) == len(G1_29_JointArmIndex) + len(REMOTE_AXES)

    def test_token_spaces_match(self, make_robot):
        """A token policy consumes its own previous action as state; both must be 64-D."""
        factory, _ = make_robot
        robot = factory(controller=TokenController())
        assert len(robot.action_features) == len(robot.observation_features) == TOKEN_DIM


# ---------------------------------------------------------------------------
# Command publishing
# ---------------------------------------------------------------------------


class TestPublishLowcmd:
    def test_writes_only_present_joints(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)

        robot.publish_lowcmd({f"{G1_29_JointIndex.kLeftKnee.name}.q": 0.42})

        assert robot.msg.motor_cmd[G1_29_JointIndex.kLeftKnee.value].q == pytest.approx(0.42)
        assert robot.msg.motor_cmd[G1_29_JointIndex.kRightKnee.value].q == pytest.approx(-1.0)

    def test_applies_configured_gains(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks, kp_value=77.0, kd_value=3.0)

        robot.publish_lowcmd({f"{G1_29_JointIndex.kWaistYaw.name}.q": 0.0})

        cmd = robot.msg.motor_cmd[G1_29_JointIndex.kWaistYaw.value]
        assert cmd.kp == pytest.approx(77.0)
        assert cmd.kd == pytest.approx(3.0)

    def test_gain_overrides_win(self, make_robot):
        """The controller thread passes its own gains (SONIC reads them from the ONNX)."""
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks, kp_value=77.0)

        override_kp = np.full(NUM_MOTORS, 10.0, np.float32)
        override_kd = np.full(NUM_MOTORS, 0.5, np.float32)
        robot.publish_lowcmd({f"{G1_29_JointIndex.kWaistYaw.name}.q": 0.0}, kp=override_kp, kd=override_kd)

        cmd = robot.msg.motor_cmd[G1_29_JointIndex.kWaistYaw.value]
        assert cmd.kp == pytest.approx(10.0)
        assert cmd.kd == pytest.approx(0.5)

    def test_concurrent_publishes_dont_tear(self, make_robot):
        """Publishes from two threads must never emit a half-updated command.

        The controller thread, ``send_action()``, ``reset()`` and the shutdown path share one
        lowcmd message, and the CRC covers whatever it holds at that moment, so an interleaved
        update goes out as a valid-CRC mix of two commands. One publisher is parked after
        writing its first joint and the other is let through, which forces that interleaving
        instead of relying on timing.
        """
        first_joint_written = threading.Event()
        other_publish_done = threading.Event()

        class ParkedAction(dict):
            """Blocks once the first joint has been written, inviting an interleaved publish."""

            reads = 0

            def __getitem__(self, key):
                self.reads += 1
                if self.reads == 2:
                    first_joint_written.set()
                    other_publish_done.wait(timeout=0.5)
                return super().__getitem__(key)

        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)
        knees = (G1_29_JointIndex.kLeftKnee, G1_29_JointIndex.kRightKnee)

        published = []
        robot.lowcmd_publisher.Write = lambda msg: published.append(
            tuple(msg.motor_cmd[knee.value].q for knee in knees)
        )

        def publish_parked():
            robot.publish_lowcmd(ParkedAction({f"{knee.name}.q": 1.0 for knee in knees}))

        def publish_other():
            first_joint_written.wait(timeout=1.0)
            robot.publish_lowcmd({f"{knee.name}.q": 2.0 for knee in knees})
            other_publish_done.set()

        threads = [threading.Thread(target=publish_parked), threading.Thread(target=publish_other)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(published) == 2
        for left, right in published:
            assert left == right, f"published a mix of both commands: {left} and {right}"

    def test_zero_gains_make_joints_passive(self, make_robot):
        """Shutdown path: zero kp/kd/tau must reach the message."""
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)

        zeros = np.zeros(NUM_MOTORS, np.float32)
        action = {f"{joint.name}.q": 0.0 for joint in G1_29_JointIndex}
        robot.publish_lowcmd(action, kp=zeros, kd=zeros, tau=zeros)

        for joint in G1_29_JointIndex:
            cmd = robot.msg.motor_cmd[joint.value]
            assert cmd.kp == pytest.approx(0.0)
            assert cmd.kd == pytest.approx(0.0)
            assert cmd.tau == pytest.approx(0.0)

    def test_tau_defaults_to_zero(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)

        robot.publish_lowcmd({f"{G1_29_JointIndex.kLeftElbow.name}.q": 0.1})

        assert robot.msg.motor_cmd[G1_29_JointIndex.kLeftElbow.value].tau == pytest.approx(0.0)

    def test_stamps_crc_and_writes_once(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)
        mocks["publisher_mock"].Write.reset_mock()
        mocks["crc_mock"].Crc.reset_mock()

        robot.publish_lowcmd({f"{G1_29_JointIndex.kLeftKnee.name}.q": 0.0})

        mocks["crc_mock"].Crc.assert_called_once_with(robot.msg)
        mocks["publisher_mock"].Write.assert_called_once_with(robot.msg)

    def test_ignores_unknown_keys(self, make_robot):
        """Token and joystick keys share the action dict with joint targets."""
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)

        robot.publish_lowcmd({"motion_token.0.pos": 1.0, "remote.lx": 0.5})

        for joint in G1_29_JointIndex:
            assert robot.msg.motor_cmd[joint.value].q == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


class TestSendAction:
    def test_token_only_action_stays_off_the_wire(self, make_robot):
        """With a controller loaded, a token-only action has nothing for send_action to publish.

        Writing anyway just repeats the controller thread's last command with a fresh CRC, so
        the robot sees commands at the policy rate on top of the controller rate.
        """
        factory, mocks = make_robot
        robot = arm_for_publish(factory(controller=TokenController()), mocks)
        mocks["publisher_mock"].Write.reset_mock()

        robot.send_action({f"motion_token.{i}.pos": 0.5 for i in range(TOKEN_DIM)})

        mocks["publisher_mock"].Write.assert_not_called()

    def test_tokens_still_reach_the_controller(self, make_robot):
        """Skipping the publish must not skip handing the action to the controller thread."""
        factory, mocks = make_robot
        robot = arm_for_publish(factory(controller=TokenController()), mocks)

        robot.send_action({"motion_token.0.pos": 1.5, "remote.lx": 0.25})

        assert robot.controller_input["motion_token.0.pos"] == 1.5
        assert robot.controller_input["remote.lx"] == 0.25

    def test_arm_targets_are_published(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(controller=TokenController()), mocks)
        shoulder = G1_29_JointArmIndex.kLeftShoulderPitch
        mocks["publisher_mock"].Write.reset_mock()

        robot.send_action({f"{shoulder.name}.q": 0.3, "motion_token.0.pos": 0.5})

        mocks["publisher_mock"].Write.assert_called_once_with(robot.msg)
        assert robot.msg.motor_cmd[shoulder.value].q == pytest.approx(0.3)

    def test_joint_action_is_published_without_ctrl(self, make_robot):
        """Without a controller the caller owns every joint, so nothing is filtered out."""
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)
        mocks["publisher_mock"].Write.reset_mock()

        robot.send_action({f"{motor.name}.q": 0.2 for motor in G1_29_JointIndex})

        mocks["publisher_mock"].Write.assert_called_once_with(robot.msg)
        assert all(q == pytest.approx(0.2) for q in published_targets(robot).values())


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def sim_ready(robot):
    """Point the robot at a stub sim env so reset() takes its single-publish branch."""
    robot.sim_env = MagicMock()
    return robot


class TestReset:
    def test_homes_to_controller_pose(self, make_robot):
        """The controller's home pose is what its policy expects, so it wins over the config."""
        factory, mocks = make_robot
        controller = TokenController()
        controller.default_angles = np.full(NUM_MOTORS, 0.25, np.float32)
        robot = sim_ready(arm_for_publish(factory(controller=controller), mocks))

        robot.reset()

        assert all(q == pytest.approx(0.25) for q in published_targets(robot).values())

    def test_homes_to_config_without_ctrl(self, make_robot):
        factory, mocks = make_robot
        robot = sim_ready(arm_for_publish(factory(default_positions=[0.7] * NUM_MOTORS), mocks))

        robot.reset()

        assert all(q == pytest.approx(0.7) for q in published_targets(robot).values())

    def test_explicit_pose_still_wins(self, make_robot):
        factory, mocks = make_robot
        controller = TokenController()
        controller.default_angles = np.full(NUM_MOTORS, 0.25, np.float32)
        robot = sim_ready(arm_for_publish(factory(controller=controller), mocks))

        robot.reset(default_positions=np.full(NUM_MOTORS, -0.1, np.float32))

        assert all(q == pytest.approx(-0.1) for q in published_targets(robot).values())

    def test_clears_controller_state(self, make_robot):
        factory, mocks = make_robot
        controller = TokenController()
        robot = sim_ready(arm_for_publish(factory(controller=controller), mocks))

        robot.reset()

        assert controller.reset_calls == 1

    def test_holds_control_authority(self, make_robot):
        """A controller tick starting mid-reset must wait instead of fighting the sweep.

        The tick is launched from inside reset's own publish, so it is guaranteed to land while
        the homing sweep is in progress rather than depending on thread scheduling.
        """
        factory, mocks = make_robot
        controller = TokenController()
        robot = sim_ready(arm_for_publish(factory(controller=controller), mocks))

        ticks: list[threading.Thread] = []
        blocked: list[bool] = []

        def controller_tick():
            with robot._control_lock:  # what the controller loop holds around a tick
                pass

        def on_write(msg):
            tick = threading.Thread(target=controller_tick)
            tick.start()
            tick.join(timeout=0.2)
            blocked.append(tick.is_alive())
            ticks.append(tick)

        robot.lowcmd_publisher.Write = on_write
        robot.reset()
        for tick in ticks:
            tick.join(timeout=1.0)

        assert blocked == [True], "a controller tick got through while reset was homing"
        assert all(not tick.is_alive() for tick in ticks), "tick never got control back"


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestDisconnect:
    def test_zero_torque_is_the_last_word(self, make_robot):
        """Going passive is only meaningful as the final command on the wire.

        The controller thread publishes at its own rate, so it has to be stopped and joined
        before the zero-gain command; otherwise its next tick re-stiffens the joints and the
        robot never actually goes limp.
        """
        factory, mocks = make_robot
        robot = arm_for_publish(hardware_mode(factory()), mocks)

        published = []
        robot._send_zero_torque = lambda: published.append("zero_torque")
        tick_started = threading.Event()

        def controller_loop():
            """Mirrors the real loop: check the flag, spend time in inference, then publish.

            The publish is unconditional, so a tick already under way still sends stiff
            targets even though the shutdown flag flipped while it was running.
            """
            while not robot._shutdown_event.is_set():
                tick_started.set()
                time.sleep(0.05)
                published.append("controller")

        robot._controller_thread = threading.Thread(target=controller_loop)
        robot._controller_thread.start()
        assert tick_started.wait(timeout=1.0), "controller loop never started a tick"

        robot.disconnect()

        assert "zero_torque" in published
        after_going_passive = published[published.index("zero_torque") :]
        assert "controller" not in after_going_passive, "controller published after zero torque"

    def test_simulation_skips_zero_torque(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(), mocks)  # the fixture builds simulated robots
        calls = []
        robot._send_zero_torque = lambda: calls.append("zero_torque")

        robot.disconnect()

        assert calls == []


# ---------------------------------------------------------------------------
# Controller input plumbing
# ---------------------------------------------------------------------------


class TestControllerInput:
    def test_starts_from_a_zeroed_remote(self, make_robot):
        factory, _ = make_robot
        robot = factory(controller=TokenController())
        assert all(value == 0.0 for value in robot.controller_input.values())

    def test_action_values_are_merged_in(self, make_robot):
        factory, _ = make_robot
        robot = factory(controller=TokenController())

        robot._update_controller_action({"remote.lx": 0.5, "motion_token.0.pos": 1.5})

        assert robot.controller_input["remote.lx"] == 0.5
        assert robot.controller_input["motion_token.0.pos"] == 1.5

    def test_none_values_are_skipped(self, make_robot):
        factory, _ = make_robot
        robot = factory(controller=TokenController())

        robot._update_controller_action({"remote.lx": 0.5})
        robot._update_controller_action({"remote.lx": None})

        assert robot.controller_input["remote.lx"] == 0.5

    def test_partial_update_keeps_keys(self, make_robot):
        """The controller thread reads a snapshot; stale axes must not be dropped."""
        factory, _ = make_robot
        robot = factory(controller=TokenController())

        robot._update_controller_action({"remote.lx": 0.5, "remote.ly": -0.5})
        robot._update_controller_action({"remote.lx": 0.25})

        assert robot.controller_input["remote.lx"] == 0.25
        assert robot.controller_input["remote.ly"] == -0.5
