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

import json
import time

import pytest

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.accessible_teleop import (
    AccessibleTeleop,
    AccessibleTeleopConfig,
    ChannelCalibration,
    InputSource,
    JointBinding,
)
from lerobot.teleoperators.accessible_teleop.control import (
    InputFrame,
    JointController,
    normalize_channel,
    shape_axis,
)
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _websockets_available

JOINTS = ["shoulder_pan", "gripper"]
LIMITS = {"shoulder_pan": (-90.0, 90.0), "gripper": (0.0, 100.0)}
START = {"shoulder_pan": 0.0, "gripper": 50.0}


def make_controller(bindings: dict[str, JointBinding], **kwargs) -> JointController:
    return JointController(
        joints=JOINTS,
        bindings=bindings,
        joint_limits=LIMITS,
        start_pose=START,
        **kwargs,
    )


# ── signal conditioning ──────────────────────────────────────────────────


def test_shape_axis_suppresses_the_dead_zone():
    assert shape_axis(0.1, deadzone=0.2, gain=1.0) == 0.0
    assert shape_axis(-0.2, deadzone=0.2, gain=1.0) == 0.0
    assert shape_axis(0.21, deadzone=0.2, gain=1.0) > 0.0


def test_shape_axis_is_sign_preserving_and_bounded():
    for value in (-1.0, -0.5, 0.5, 1.0):
        shaped = shape_axis(value, deadzone=0.05, gain=3.0)
        assert (shaped > 0) == (value > 0)
        assert abs(shaped) <= 1.0


def test_shape_axis_gain_reaches_full_output_before_full_input():
    modest = shape_axis(0.6, deadzone=0.1, gain=1.0)
    amplified = shape_axis(0.6, deadzone=0.1, gain=2.5)
    assert amplified > modest
    assert amplified == pytest.approx(1.0)


def test_normalize_channel_scales_each_direction_independently():
    calibration = ChannelCalibration(neutral=0.2, negative_range=0.1, positive_range=0.4, ready=True)
    assert normalize_channel(0.2, calibration) == pytest.approx(0.0)
    assert normalize_channel(0.6, calibration) == pytest.approx(1.0)
    assert normalize_channel(0.1, calibration) == pytest.approx(-1.0)
    # Beyond the captured range the value saturates rather than growing.
    assert normalize_channel(5.0, calibration) == pytest.approx(1.0)


# ── binding resolution ───────────────────────────────────────────────────


def test_keyboard_binding_reads_both_directions():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", negative_key="KeyG")
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})

    assert controller.axis_for_joint("shoulder_pan", InputFrame(keys={"KeyF": True})) == 1.0
    assert controller.axis_for_joint("shoulder_pan", InputFrame(keys={"KeyG": True})) == -1.0
    # Holding both keys is a wash rather than an error.
    assert controller.axis_for_joint("shoulder_pan", InputFrame(keys={"KeyF": True, "KeyG": True})) == 0.0


def test_invert_swaps_direction():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", negative_key="KeyG", invert=True)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})
    assert controller.axis_for_joint("shoulder_pan", InputFrame(keys={"KeyF": True})) == -1.0


def test_unbound_joint_never_moves():
    controller = make_controller({"shoulder_pan": JointBinding(), "gripper": JointBinding()})
    frame = InputFrame(channels={"pad_a.x": 1.0}, engaged=True, tracking=True)
    for _ in range(50):
        controller.step(frame, 0.05)
    assert controller.pose == START


def test_face_binding_is_inert_until_calibrated():
    binding = JointBinding(source=InputSource.FACE, channel="face.mouth_open", deadzone=0.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})
    frame = InputFrame(channels={"face.mouth_open": 1.0}, engaged=True, tracking=True)

    assert controller.axis_for_joint("shoulder_pan", frame) == 0.0

    controller.calibrations["face.mouth_open"] = ChannelCalibration(
        neutral=0.0, negative_range=0.5, positive_range=0.5, ready=True
    )
    assert controller.axis_for_joint("shoulder_pan", frame) > 0.0


def test_face_binding_stops_when_tracking_is_lost():
    binding = JointBinding(source=InputSource.FACE, channel="face.mouth_open", deadzone=0.0)
    controller = make_controller(
        {"shoulder_pan": binding, "gripper": JointBinding()},
        calibrations={
            "face.mouth_open": ChannelCalibration(
                neutral=0.0, negative_range=0.5, positive_range=0.5, ready=True
            )
        },
    )
    channels = {"face.mouth_open": 1.0}
    assert controller.axis_for_joint("shoulder_pan", InputFrame(channels=channels, tracking=True)) > 0.0
    assert controller.axis_for_joint("shoulder_pan", InputFrame(channels=channels, tracking=False)) == 0.0


def test_losing_tracking_leaves_other_sources_working():
    controller = make_controller(
        {
            "shoulder_pan": JointBinding(source=InputSource.FACE, channel="face.mouth_open"),
            "gripper": JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF"),
        }
    )
    frame = InputFrame(keys={"KeyF": True}, engaged=True, tracking=False)
    assert controller.axis_for_joint("gripper", frame) == 1.0


# ── integration ──────────────────────────────────────────────────────────


def test_clutch_gates_all_motion():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=30.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})

    controller.step(InputFrame(keys={"KeyF": True}, engaged=False), 0.1)
    assert controller.pose["shoulder_pan"] == 0.0

    controller.step(InputFrame(keys={"KeyF": True}, engaged=True), 0.05)
    assert controller.pose["shoulder_pan"] > 0.0


def test_velocity_integrates_at_the_configured_speed():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=30.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})
    frame = InputFrame(keys={"KeyF": True}, engaged=True)

    for _ in range(20):
        controller.step(frame, 0.05)  # 20 steps x 0.05 s x 30 deg/s = 30 deg

    assert controller.pose["shoulder_pan"] == pytest.approx(30.0)


def test_step_is_clamped_so_a_stalled_loop_cannot_lurch():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=60.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()}, max_step_s=0.06)

    # The caller reports a 10 s gap, e.g. after a breakpoint or a suspended laptop.
    controller.step(InputFrame(keys={"KeyF": True}, engaged=True), 10.0)

    assert controller.pose["shoulder_pan"] == pytest.approx(60.0 * 0.06)


def test_pose_never_leaves_the_joint_limits():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=120.0)
    negative = JointBinding(source=InputSource.KEYBOARD, negative_key="KeyG", smoothing=0.0, speed=120.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": negative})

    for _ in range(200):
        controller.step(InputFrame(keys={"KeyF": True, "KeyG": True}, engaged=True), 0.06)

    assert controller.pose["shoulder_pan"] == pytest.approx(90.0)
    assert controller.pose["gripper"] == pytest.approx(0.0)


def test_smoothing_delays_full_speed_but_still_reaches_it():
    smooth = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.9, speed=30.0)
    sharp = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=30.0)
    smooth_controller = make_controller({"shoulder_pan": smooth, "gripper": JointBinding()})
    sharp_controller = make_controller({"shoulder_pan": sharp, "gripper": JointBinding()})
    frame = InputFrame(keys={"KeyF": True}, engaged=True)

    smooth_controller.step(frame, 0.05)
    sharp_controller.step(frame, 0.05)
    assert smooth_controller.pose["shoulder_pan"] < sharp_controller.pose["shoulder_pan"]

    for _ in range(60):
        smooth_controller.step(frame, 0.05)
    assert smooth_controller.velocity["shoulder_pan"] == pytest.approx(30.0, rel=1e-2)


def test_releasing_an_input_actually_stops_the_joint():
    binding = JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.8, speed=30.0)
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})

    for _ in range(20):
        controller.step(InputFrame(keys={"KeyF": True}, engaged=True), 0.05)
    for _ in range(200):
        controller.step(InputFrame(engaged=True), 0.05)

    assert controller.velocity["shoulder_pan"] == 0.0


def test_reset_takes_a_measured_pose_as_given():
    # Anchoring is a statement about where the robot is, so clamping it here would turn
    # re-anchoring into a movement command.
    controller = make_controller({"shoulder_pan": JointBinding(), "gripper": JointBinding()})
    controller.reset({"shoulder_pan": -102.5, "gripper": 2.3})
    assert controller.pose["shoulder_pan"] == pytest.approx(-102.5)
    assert controller.pose["gripper"] == pytest.approx(2.3)


def test_reset_clamps_the_configured_start_pose():
    controller = JointController(
        joints=JOINTS,
        bindings={"shoulder_pan": JointBinding(), "gripper": JointBinding()},
        joint_limits=LIMITS,
        start_pose={"shoulder_pan": 500.0, "gripper": 50.0},
    )
    assert controller.pose["shoulder_pan"] == pytest.approx(90.0)


def test_a_joint_parked_outside_its_limits_can_still_come_back():
    binding = JointBinding(
        source=InputSource.KEYBOARD,
        positive_key="KeyF",
        negative_key="KeyG",
        smoothing=0.0,
        speed=30.0,
    )
    controller = make_controller({"shoulder_pan": binding, "gripper": JointBinding()})
    # The arm was parked against its mechanical stop, past the configured limit.
    controller.reset({"shoulder_pan": 102.5, "gripper": 50.0})

    controller.step(InputFrame(keys={"KeyF": True}, engaged=True), 0.05)
    assert controller.pose["shoulder_pan"] == pytest.approx(102.5), "must not travel further out"

    for _ in range(20):
        controller.step(InputFrame(keys={"KeyG": True}, engaged=True), 0.05)
    assert controller.pose["shoulder_pan"] < 102.5, "must be able to come back into range"


# ── teleoperator ─────────────────────────────────────────────────────────


def test_registered_in_the_teleoperator_factory():
    config = AccessibleTeleopConfig(id="factory_check")
    assert config.type == "accessible_teleop"
    assert isinstance(make_teleoperator_from_config(config), AccessibleTeleop)


def test_action_features_match_the_configured_joints():
    teleop = AccessibleTeleop(AccessibleTeleopConfig(id="features", joints=list(JOINTS)))
    assert set(teleop.action_features) == {"shoulder_pan.pos", "gripper.pos"}
    assert teleop.feedback_features == teleop.action_features


def test_get_action_requires_a_connection():
    teleop = AccessibleTeleop(AccessibleTeleopConfig(id="disconnected"))
    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()


def test_a_profile_with_only_keyboard_bindings_needs_no_calibration():
    config = AccessibleTeleopConfig(
        id="keys_only",
        joints=list(JOINTS),
        bindings={
            "shoulder_pan": JointBinding(source=InputSource.KEYBOARD, positive_key="KeyF"),
            "gripper": JointBinding(),
        },
    )
    assert AccessibleTeleop(config).is_calibrated


def test_a_bound_face_channel_must_be_calibrated():
    config = AccessibleTeleopConfig(
        id="face_bound",
        joints=list(JOINTS),
        bindings={
            "shoulder_pan": JointBinding(source=InputSource.FACE, channel="face.mouth_open"),
            "gripper": JointBinding(),
        },
    )
    teleop = AccessibleTeleop(config)
    assert not teleop.is_calibrated

    teleop.channel_calibrations["face.mouth_open"] = ChannelCalibration(
        neutral=0.0, negative_range=0.3, positive_range=0.3, ready=True
    )
    assert teleop.is_calibrated


def test_profile_round_trips_through_the_calibration_file(tmp_path):
    config = AccessibleTeleopConfig(
        id="round_trip",
        calibration_dir=tmp_path,
        joints=list(JOINTS),
        bindings={
            "shoulder_pan": JointBinding(source=InputSource.JOYSTICK, channel="pad_a.x", speed=17.0),
            "gripper": JointBinding(source=InputSource.FACE, channel="face.mouth_open", invert=True),
        },
    )
    saved = AccessibleTeleop(config)
    saved.channel_calibrations["face.mouth_open"] = ChannelCalibration(
        neutral=0.1, negative_range=0.2, positive_range=0.4, ready=True
    )
    saved._save_calibration()

    profile = json.loads((tmp_path / "round_trip.json").read_text())
    assert profile["bindings"]["shoulder_pan"]["source"] == "joystick"

    loaded = AccessibleTeleop(AccessibleTeleopConfig(id="round_trip", calibration_dir=tmp_path))
    assert loaded.bindings["shoulder_pan"].channel == "pad_a.x"
    assert loaded.bindings["shoulder_pan"].speed == pytest.approx(17.0)
    assert loaded.bindings["gripper"].invert
    assert loaded.channel_calibrations["face.mouth_open"].ready
    assert loaded.is_calibrated


def test_send_feedback_anchors_the_commanded_pose():
    teleop = AccessibleTeleop(AccessibleTeleopConfig(id="anchor", joints=list(JOINTS)))
    teleop.sync_to({"shoulder_pan": -33.0, "gripper": 12.0})
    assert teleop.controller.pose["shoulder_pan"] == pytest.approx(-33.0)
    assert teleop.controller.pose["gripper"] == pytest.approx(12.0)


# ── web bridge ───────────────────────────────────────────────────────────

pytestmark_ws = pytest.mark.skipif(not _websockets_available, reason="requires lerobot[accessible-teleop]")


@pytest.fixture
def connected_teleop(tmp_path):
    config = AccessibleTeleopConfig(
        id="bridge",
        calibration_dir=tmp_path,
        joints=list(JOINTS),
        joint_limits=dict(LIMITS),
        start_pose=dict(START),
        bindings={
            "shoulder_pan": JointBinding(
                source=InputSource.KEYBOARD, positive_key="KeyF", smoothing=0.0, speed=30.0
            ),
            "gripper": JointBinding(),
        },
        host="127.0.0.1",
        web_port=0,
        open_browser=False,
        connect_timeout_s=10.0,
    )
    teleop = AccessibleTeleop(config)
    yield teleop
    if teleop.is_connected:
        teleop.disconnect()


@pytestmark_ws
def test_control_page_is_served_and_drives_the_robot(connected_teleop):
    import threading
    from urllib.request import urlopen

    from websockets.sync.client import connect as ws_connect

    teleop = connected_teleop
    thread = threading.Thread(target=teleop.connect, kwargs={"calibrate": False}, daemon=True)
    thread.start()

    # Wait for the server socket before dialling it.
    deadline = time.monotonic() + 10.0
    while teleop._bridge is None or not teleop._bridge.is_running:
        assert time.monotonic() < deadline, "bridge never started"
        time.sleep(0.02)
    url = teleop._bridge.url

    page = urlopen(url, timeout=5).read().decode()  # noqa: S310 - loopback URL built above
    assert "__LEROBOT_BOOTSTRAP__" not in page, "bootstrap placeholder was not substituted"
    assert "shoulder_pan" in page

    with ws_connect(url.replace("http://", "ws://") + "ws") as socket:
        thread.join(timeout=10.0)
        assert teleop.is_connected

        # Nothing has been engaged, so the pose is held rather than withheld: callers pass
        # this action straight to send_action, which cannot accept an empty one.
        socket.send(json.dumps({"type": "input", "keys": {"KeyF": False}, "engaged": False}))
        _wait_for_frame(teleop)
        idle = teleop.get_action()
        assert set(idle) == set(teleop.action_features)
        time.sleep(0.05)
        assert teleop.get_action() == pytest.approx(idle)

        socket.send(json.dumps({"type": "input", "keys": {"KeyF": True}, "engaged": True}))
        _wait_for_frame(teleop)
        teleop.get_action()  # first engaged call only establishes the time base
        time.sleep(0.05)
        action = teleop.get_action()

        assert set(action) == {"shoulder_pan.pos", "gripper.pos"}
        assert action["shoulder_pan.pos"] > 0.0
        assert action["gripper.pos"] == pytest.approx(50.0)


@pytestmark_ws
def test_stale_input_stops_the_robot(connected_teleop):
    import threading

    from websockets.sync.client import connect as ws_connect

    teleop = connected_teleop
    teleop.config.input_timeout_s = 0.1
    thread = threading.Thread(target=teleop.connect, kwargs={"calibrate": False}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while teleop._bridge is None or not teleop._bridge.is_running:
        assert time.monotonic() < deadline, "bridge never started"
        time.sleep(0.02)

    with ws_connect(teleop._bridge.url.replace("http://", "ws://") + "ws") as socket:
        thread.join(timeout=10.0)
        socket.send(json.dumps({"type": "input", "keys": {"KeyF": True}, "engaged": True}))
        _wait_for_frame(teleop)
        teleop.get_action()
        time.sleep(0.05)
        moving = teleop.get_action()["shoulder_pan.pos"]
        assert moving > 0.0

        # The page goes quiet: the joint must hold rather than keep travelling.
        time.sleep(0.3)
        held = teleop.get_action()["shoulder_pan.pos"]
        time.sleep(0.2)
        assert teleop.get_action()["shoulder_pan.pos"] == pytest.approx(held)


@pytestmark_ws
def test_every_action_is_complete_enough_for_a_motor_bus(connected_teleop):
    """An action with no entries reaches the bus as a write addressed to no motors.

    `MotorsBus.sync_write` takes the control table from the first motor in the request, so an
    empty mapping raises StopIteration deep inside the transport. Nothing between this
    teleoperator and the bus filters that out: the default processors are the identity, and
    `send_action` forwards whatever it is given.
    """
    import threading

    from websockets.sync.client import connect as ws_connect

    teleop = connected_teleop
    thread = threading.Thread(target=teleop.connect, kwargs={"calibrate": False}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while teleop._bridge is None or not teleop._bridge.is_running:
        assert time.monotonic() < deadline, "bridge never started"
        time.sleep(0.02)

    with ws_connect(teleop._bridge.url.replace("http://", "ws://") + "ws") as socket:
        thread.join(timeout=10.0)
        # Before any input at all, while idle, and while driving.
        assert set(teleop.get_action()) == set(teleop.action_features)

        socket.send(json.dumps({"type": "input", "keys": {}, "engaged": False}))
        _wait_for_frame(teleop)
        assert set(teleop.get_action()) == set(teleop.action_features)

        socket.send(json.dumps({"type": "input", "keys": {"KeyF": True}, "engaged": True}))
        time.sleep(0.1)
        assert set(teleop.get_action()) == set(teleop.action_features)


@pytestmark_ws
def test_an_idle_second_page_cannot_override_the_page_holding_the_clutch(connected_teleop):
    import threading

    from websockets.sync.client import connect as ws_connect

    teleop = connected_teleop
    thread = threading.Thread(target=teleop.connect, kwargs={"calibrate": False}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while teleop._bridge is None or not teleop._bridge.is_running:
        assert time.monotonic() < deadline, "bridge never started"
        time.sleep(0.02)
    ws_url = teleop._bridge.url.replace("http://", "ws://") + "ws"

    with ws_connect(ws_url) as driver, ws_connect(ws_url) as bystander:
        thread.join(timeout=10.0)
        driver.send(json.dumps({"type": "input", "keys": {"KeyF": True}, "engaged": True}))
        _wait_for_frame(teleop)
        # The second page reports after the first, and would win on recency alone.
        bystander.send(json.dumps({"type": "input", "keys": {}, "engaged": False}))
        time.sleep(0.1)

        frame, _ = teleop._bridge.read_frame()
        assert frame.engaged, "an idle page took the clutch away from the page being driven"
        assert frame.keys.get("KeyF")

        # Once the driving page releases, the freshest frame wins again.
        driver.send(json.dumps({"type": "input", "keys": {}, "engaged": False}))
        time.sleep(0.1)
        frame, _ = teleop._bridge.read_frame()
        assert not frame.engaged


@pytestmark_ws
def test_a_page_that_goes_quiet_loses_the_clutch_to_a_live_page(connected_teleop):
    import threading

    from websockets.sync.client import connect as ws_connect

    teleop = connected_teleop
    teleop.config.input_timeout_s = 0.1
    thread = threading.Thread(target=teleop.connect, kwargs={"calibrate": False}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while teleop._bridge is None or not teleop._bridge.is_running:
        assert time.monotonic() < deadline, "bridge never started"
        time.sleep(0.02)
    ws_url = teleop._bridge.url.replace("http://", "ws://") + "ws"

    with ws_connect(ws_url) as frozen, ws_connect(ws_url) as live:
        thread.join(timeout=10.0)
        # A backgrounded tab stops its loop while still claiming the clutch.
        frozen.send(json.dumps({"type": "input", "keys": {"KeyF": True}, "engaged": True}))
        _wait_for_frame(teleop)
        time.sleep(0.2)

        live.send(json.dumps({"type": "input", "keys": {}, "engaged": False}))
        time.sleep(0.05)
        frame, age = teleop._bridge.read_frame()
        assert not frame.engaged, "a frozen page kept the clutch closed"
        assert age < teleop.config.input_timeout_s


def _wait_for_frame(teleop: AccessibleTeleop, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        _, age = teleop._bridge.read_frame()
        if age is not None:
            return
        assert time.monotonic() < deadline, "no input frame arrived from the test client"
        time.sleep(0.01)
