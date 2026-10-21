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

"""LightNav client against a fake lightnav-serve speaking its WebSocket protocol."""

import base64
import json
import math
import threading
import time
from dataclasses import dataclass

import cv2
import draccus
import numpy as np
import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from lerobot.teleoperators import TeleoperatorConfig  # noqa: E402
from lerobot.teleoperators.lightnav import LightNavTeleop, LightNavTeleopConfig  # noqa: E402
from lerobot.teleoperators.lightnav.teleop_lightnav import limit_velocity, waypoint_velocity  # noqa: E402
from lerobot.teleoperators.mobile_base import MobileBaseTeleop  # noqa: E402
from lerobot.teleoperators.utils import make_teleoperator_from_config  # noqa: E402

BASE = ("x.vel", "y.vel", "theta.vel")
ARM = ("arm_shoulder.pos", "arm_gripper.pos")
STOPPED = dict.fromkeys(BASE, 0.0)
INSTRUCTION = "Go to the chair on your right and stop."


class FakeLightNavServer:
    """Answer login/reset with ok and every prediction request with ``reply(request_data)``."""

    def __init__(self, reply):
        self.reply = reply
        self.requests: list[dict] = []
        self.drop = threading.Event()
        self._server = serve(self._handle, "127.0.0.1", 0, max_size=None)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"ws://127.0.0.1:{self._server.socket.getsockname()[1]}"

    def _handle(self, websocket):
        for message in websocket:
            request = json.loads(message)
            self.requests.append(request)
            if self.drop.is_set():
                websocket.close()
                return
            action, data = request["action"], request["data"]
            reply = {"rc": 0, "seq": data["seq"], **self.reply(data)} if action == "next" else {"rc": 0}
            websocket.send(json.dumps({"action": action, "data": reply}))

    def actions(self) -> list[str]:
        return [request["action"] for request in self.requests]

    def close(self):
        self._server.shutdown()
        self._thread.join()


def prediction(*waypoints, stop=False):
    return {"actions": {"step": 1, "actions": [list(w) for w in waypoints]}, "stop": stop, "latency_ms": 5.0}


def wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


def red_frame():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[..., 0] = 255
    return frame


@pytest.fixture
def server():
    server = FakeLightNavServer(lambda data: prediction((0.2, 0.1, 0.1)))
    yield server
    server.close()


def make_teleop(server, **overrides):
    # One-second rows keep the first waypoint active for the whole test.
    config = LightNavTeleopConfig(server_url=server.url, instruction=INSTRUCTION, waypoint_dt_s=1.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return LightNavTeleop(config)


@pytest.fixture
def teleop(server):
    teleop = make_teleop(server)
    teleop.connect()
    yield teleop
    if teleop.is_connected:
        teleop.disconnect()


def test_first_segment_is_divided_by_the_row_duration():
    velocity = waypoint_velocity(np.array([[0.2, 0.1, 0.1], [0.4, 0.2, 0.2]]), 0.05, 0.1)
    assert velocity == pytest.approx((2.0, 1.0, 1.0))


def test_later_segments_are_expressed_in_their_start_pose():
    # Turn left by 90 degrees, then move along the original +y axis: that is straight ahead.
    waypoints = np.array([[0.0, 0.0, math.pi / 2], [0.0, 0.3, math.pi / 2]])
    assert waypoint_velocity(waypoints, 0.15, 0.1) == pytest.approx((3.0, 0.0, 0.0))


def test_exhausted_chunk_returns_none_and_clock_skew_uses_the_first_row():
    waypoints = np.array([[0.1, 0.0, 0.0]])
    assert waypoint_velocity(waypoints, 0.1, 0.1) is None
    assert waypoint_velocity(waypoints, -0.01, 0.1) == pytest.approx((1.0, 0.0, 0.0))


def test_speed_limits_scale_the_command_uniformly():
    assert limit_velocity(0.6, 0.0, 30.0, 0.3, 90.0) == pytest.approx((0.3, 0.0, 15.0))
    assert limit_velocity(0.1, 0.1, 180.0, 0.3, 90.0) == pytest.approx((0.05, 0.05, 90.0))
    assert limit_velocity(0.1, 0.0, -10.0, 0.3, 90.0) == (0.1, 0.0, -10.0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"instruction": " "}, "instruction"),
        ({"server_url": "http://gpu:8050"}, "ws://"),
        ({"waypoint_dt_s": 0.0}, "waypoint_dt_s"),
        ({"jpeg_quality": 0}, "jpeg_quality"),
    ],
)
def test_config_rejects_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        LightNavTeleopConfig(**{"instruction": INSTRUCTION, **overrides})


def test_cli_config_builds_the_teleoperator():
    @dataclass
    class _Config:
        teleop: TeleoperatorConfig

    cfg = draccus.parse(
        _Config,
        args=["--teleop.type=lightnav", f"--teleop.instruction={INSTRUCTION}", "--teleop.camera_key=front"],
    )
    teleop = make_teleoperator_from_config(cfg.teleop)
    assert isinstance(teleop, LightNavTeleop)
    assert teleop.action_features == dict.fromkeys(BASE, float)
    assert teleop.feedback_features == {}


def test_streams_frames_and_follows_the_prediction(server, teleop):
    assert server.actions() == ["login", "reset"]
    assert teleop.get_action() == STOPPED

    teleop.on_observation({"front": red_frame(), "arm_shoulder.pos": 0.0})
    wait_until(lambda: teleop.last_prediction is not None)

    request = server.requests[2]
    assert request["action"] == "next"
    assert request["data"]["instruction"] == INSTRUCTION
    jpeg = np.frombuffer(base64.b64decode(request["data"]["image"]), dtype=np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert rgb.shape == (48, 64, 3)
    assert rgb[..., 0].mean() > 200 and rgb[..., 2].mean() < 50

    action = teleop.get_action()
    assert action == pytest.approx({"x.vel": 0.2, "y.vel": 0.1, "theta.vel": math.degrees(0.1)})
    assert all(type(value) is float for value in action.values())


def test_non_holonomic_bases_never_strafe(server):
    teleop = make_teleop(server, holonomic=False)
    teleop.connect()
    try:
        teleop.on_observation({"front": red_frame()})
        wait_until(lambda: teleop.last_prediction is not None)
        assert teleop.get_action() == pytest.approx(
            {"x.vel": 0.2, "y.vel": 0.0, "theta.vel": math.degrees(0.1)}
        )
    finally:
        teleop.disconnect()


def test_only_the_newest_frame_is_sent_and_one_request_is_in_flight(server):
    release = threading.Event()

    def slow_reply(data):
        release.wait(5.0)
        return prediction((0.2, 0.0, 0.0))

    server.reply = slow_reply
    teleop = make_teleop(server)
    teleop.connect()
    try:
        teleop.on_observation({"front": red_frame()})
        wait_until(lambda: len(server.requests) == 3)
        for _ in range(5):
            teleop.on_observation({"front": red_frame()})
        release.set()
        wait_until(lambda: len(server.requests) == 4)
        time.sleep(0.05)
        assert [request["data"]["seq"] for request in server.requests[2:]] == [1, 2]
    finally:
        teleop.disconnect()


def test_a_repeated_camera_array_is_sent_once(server, teleop):
    # A camera slower than the loop keeps returning its latest picture, as the same array.
    cached = red_frame()
    for _ in range(5):
        teleop.on_observation({"front": cached})
        time.sleep(0.02)
    assert server.actions() == ["login", "reset", "next"]

    teleop.on_observation({"front": red_frame()})
    wait_until(lambda: len(server.requests) == 4)
    time.sleep(0.05)
    assert server.actions() == ["login", "reset", "next", "next"]


def test_arrival_latches_a_stop_until_reset_episode(server, teleop):
    server.reply = lambda data: prediction((0.0, 0.0, 0.0), stop=True)
    teleop.on_observation({"front": red_frame()})
    wait_until(lambda: teleop.arrived)
    assert teleop.get_action() == STOPPED

    teleop.on_observation({"front": red_frame()})
    time.sleep(0.05)
    assert server.actions() == ["login", "reset", "next"]

    server.reply = lambda data: prediction((0.2, 0.0, 0.0))
    teleop.reset_episode("Walk to the leftmost TV in front")
    teleop.on_observation({"front": red_frame()})
    wait_until(lambda: teleop.last_prediction is not None)
    assert server.actions() == ["login", "reset", "next", "reset", "next"]
    assert server.requests[-1]["data"]["instruction"] == "Walk to the leftmost TV in front"
    assert not teleop.arrived
    assert teleop.get_action()["x.vel"] == pytest.approx(0.2)


def test_server_errors_keep_the_session(server, teleop):
    replies = iter([{"rc": 500, "msg": "decode failed"}, prediction((0.2, 0.0, 0.0))])
    server.reply = lambda data: next(replies)
    teleop.on_observation({"front": red_frame()})
    wait_until(lambda: len(server.requests) == 3)
    teleop.on_observation({"front": red_frame()})
    wait_until(lambda: teleop.last_prediction is not None)
    assert teleop.is_connected
    assert teleop.get_action()["x.vel"] == pytest.approx(0.2)


def test_missing_or_malformed_camera_frames_are_rejected(teleop):
    with pytest.raises(KeyError, match="camera_key"):
        teleop.on_observation({"wrist": red_frame()})
    with pytest.raises(ValueError, match="uint8"):
        teleop.on_observation({"front": red_frame().astype(np.float32)})


class FakeMobileRobot:
    name = "fake_mobile"
    action_features = dict.fromkeys(ARM + BASE, float)
    observation_features = action_features


def test_drives_a_mobile_base_and_stops_it_when_the_server_goes_away(server, teleop):
    mobile = MobileBaseTeleop(FakeMobileRobot(), base=teleop, hold_uncommanded=True)
    observation = {"arm_shoulder.pos": 10.0, "arm_gripper.pos": 50.0, **STOPPED, "front": red_frame()}
    assert mobile.action_features == dict.fromkeys(ARM + BASE, float)

    mobile.on_observation(observation)
    wait_until(lambda: teleop.last_prediction is not None)
    action = mobile.get_action()
    assert action["arm_shoulder.pos"] == 10.0 and action["arm_gripper.pos"] == 50.0
    assert action["x.vel"] == pytest.approx(0.2)

    server.drop.set()
    mobile.on_observation({**observation, "front": red_frame()})
    wait_until(lambda: not teleop.is_connected)
    assert {key: mobile.get_action()[key] for key in BASE} == STOPPED
