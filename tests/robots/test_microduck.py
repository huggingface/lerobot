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

"""The Microduck robot speaks robotd's NDJSON JSON-RPC protocol.

A fake robotd on a Unix socket answers requests, streams ``robot.state`` to subscribers and records the
notifications it receives, so these tests pin the wire messages LeRobot sends and how it reads the state.
"""

import asyncio
import json
import math
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests

from lerobot.robots.microduck import MicroDuck, MicroDuckConfig
from lerobot.robots.microduck.microduck import JOINT_NAMES
from lerobot.utils.errors import DeviceNotConnectedError

STATE = {
    "t": 1.0,
    "move": {"requested": [0.1, 0.0, 0.2], "applied": [0.1, 0.0, 0.2]},
    "head": [0.0, 0.0, 0.0, 0.0],
    "policy": "walk",
    "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
    "loop": {"hz": 50.0, "missed": 0},
    "joints": [0.01 * i for i in range(len(JOINT_NAMES))],
    "targets": [0.0] * len(JOINT_NAMES),
    "odom": {"position": [1.0, 2.0, 0.0], "yaw": 0.5},
    "imu": {"gyro": [0.1, 0.2, 0.3], "quat": [1.0, 0.0, 0.0, 0.0]},
}


class FakeRobotd:
    def __init__(self, path: str):
        self.path = path
        self.requests: list[dict] = []
        self.notifications: list[dict] = []
        self.state_connection: socket.socket | None = None
        self.close_before_first_state = False
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen()
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket):
        reader = conn.makefile("r")
        for line in reader:
            message = json.loads(line)
            if "id" not in message:
                self.notifications.append(message)
                continue
            self.requests.append(message)
            method = message["method"]
            result = {"api_version": 41} if method == "hello" else {"accepted": True}
            conn.sendall(
                (json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n").encode()
            )
            if method == "robot.subscribe":
                self.state_connection = conn
                if self.close_before_first_state:
                    self.close_state()
                    return
                while not self._stop.is_set():
                    state = {"jsonrpc": "2.0", "method": "robot.state", "params": STATE}
                    try:
                        conn.sendall((json.dumps(state) + "\n").encode())
                    except OSError:
                        return
                    time.sleep(0.01)
                return

    def close(self):
        self._stop.set()
        self._server.close()

    def methods(self) -> list[str]:
        return [r["method"] for r in self.requests]

    def close_state(self):
        self.state_connection.shutdown(socket.SHUT_RDWR)
        self.state_connection.close()


@pytest.fixture
def robotd():
    # Unix socket paths are limited to ~104 bytes on macOS, so stay out of pytest's long tmp_path.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        server = FakeRobotd(str(Path(tmp) / "robotd.sock"))
        yield server
        server.close()


@pytest.fixture
def robot(robotd):
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False))
    robot.connect()
    yield robot
    if robot.is_connected:
        robot.disconnect()


def test_connect_says_hello_subscribes_and_enables(robot, robotd):
    methods = robotd.methods()
    assert methods[0] == "hello"
    assert "robot.subscribe" in methods
    assert methods[-1] == "robot.enable"
    enable = robotd.requests[-1]
    assert enable["params"] == {"on": True}


def test_observation_reads_the_state_stream(robot):
    obs = robot.get_observation()
    assert set(obs) == set(robot.observation_features)
    assert obs["left_hip_yaw.pos"] == 0.0
    assert obs["right_ankle.pos"] == pytest.approx(0.14)
    assert (obs["gyro.x"], obs["gyro.y"], obs["gyro.z"]) == (0.1, 0.2, 0.3)
    assert obs["gravity.z"] == -1.0
    assert (obs["odom.x"], obs["odom.y"], obs["odom.yaw"]) == (1.0, 2.0, 0.5)
    assert obs["theta.vel.applied"] == pytest.approx(math.degrees(0.2))


def test_send_action_clips_and_notifies_move_and_head(robot, robotd):
    sent = robot.send_action({"x.vel": 1.0, "y.vel": -0.1, "theta.vel": 200.0, "head_yaw.pos": 2.0})
    assert sent["x.vel"] == pytest.approx(0.3)
    assert sent["theta.vel"] == pytest.approx(86.0)
    assert sent["head_yaw.pos"] == pytest.approx(0.8)
    assert set(sent) == set(robot.action_features)

    deadline = time.monotonic() + 1.0
    while len(robotd.notifications) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    move, head = robotd.notifications[-2:]
    assert move["method"] == "robot.move"
    assert move["params"] == {
        "vx": pytest.approx(0.3),
        "vy": pytest.approx(-0.1),
        "vyaw": pytest.approx(math.radians(86.0)),
    }
    assert head["method"] == "robot.head"
    assert head["params"] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.8, "head_roll": 0.0}


def test_disconnect_stops_without_relaxing(robot, robotd):
    robot.disconnect()
    assert not robot.is_connected
    assert "robot.stop" in robotd.methods()
    assert "robot.relax" not in robotd.methods()


def test_state_eof_rejects_observations_and_actions_but_allows_cleanup(robot, robotd):
    robotd.close_state()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            robot.get_observation()
        except DeviceNotConnectedError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("The closed state stream still supplies observations.")

    assert robot.is_connected
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.send_action({"x.vel": 0.1})
    assert robotd.notifications == []
    robot.disconnect()
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []
    assert "robot.stop" in robotd.methods()

    robot.connect()
    assert robot.get_observation()["right_ankle.pos"] == pytest.approx(0.14)
    robot.send_action({})


def test_state_eof_before_first_state_closes_both_connections(robotd, tmp_path):
    robotd.close_before_first_state = True
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False, calibration_dir=tmp_path))
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.connect()
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []


def test_observation_keeps_validated_state_if_eof_arrives_after_check(robot, robotd, monkeypatch):
    check_state = robot._check_state_stream

    def fail_after_check():
        snapshot = check_state()
        # Force the former check/snapshot race without scheduling a thread.
        with robot._state_lock:
            robot._state_error = ConnectionError("EOF after state validation")
            robot._latest_state = None
        robotd.close_state()
        return snapshot

    monkeypatch.setattr(robot, "_check_state_stream", fail_after_check)
    observation = robot.get_observation()
    assert observation["right_ankle.pos"] == pytest.approx(0.14)
    assert observation["gravity.z"] == -1
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.get_observation()


@pytest.mark.parametrize("method", ["hello", "robot.subscribe", "robot.enable"])
def test_connect_error_closes_both_connections(robotd, tmp_path, monkeypatch, method):
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False, calibration_dir=tmp_path))
    rpc = robot._state_rpc if method == "robot.subscribe" else robot._cmd
    request = rpc.request

    def fail_request(name, *args, **kwargs):
        if name == method:
            raise ConnectionError("Injected handshake failure")
        return request(name, *args, **kwargs)

    monkeypatch.setattr(rpc, "request", fail_request)
    start = time.monotonic()
    with pytest.raises(ConnectionError, match="Injected handshake failure"):
        robot.connect()
    assert time.monotonic() - start < robot.config.rpc_timeout_s + robot.config.camera_timeout_s + 1
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []


class SnapshotCamera:
    """Synthetic HTTP responses, including one request held across disconnect."""

    def __init__(self):
        self.failed = False
        self.delay = False
        self.request_started = threading.Event()
        self.release_request = threading.Event()
        encoded, png = cv2.imencode(".png", np.full((16, 16, 3), (40, 80, 120), dtype=np.uint8))
        assert encoded
        self.response = SimpleNamespace(content=png.tobytes(), raise_for_status=lambda: None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, timeout):
        if self.failed:
            raise requests.HTTPError("503 from the camera")
        if self.delay:
            self.request_started.set()
            self.release_request.wait()
        return self.response


@pytest.mark.parametrize("delayed_response", [False, True], ids=["cached-frame", "late-response"])
def test_camera_reconnect_discards_previous_session(robotd, tmp_path, monkeypatch, delayed_response):
    camera = SnapshotCamera()
    monkeypatch.setattr("lerobot.robots.microduck.microduck.requests.Session", lambda: camera)
    robot = MicroDuck(
        MicroDuckConfig(
            robotd_address=robotd.path,
            calibration_dir=tmp_path,
            use_camera=True,
            camera_transport="snapshot",
            camera_width=16,
            camera_height=16,
            camera_timeout_s=0.01,
        )
    )
    try:
        robot.connect()
        previous = robot.get_observation()["front"]
        previous_worker = next(thread for thread in robot._threads if thread.name == "microduck-camera")
        assert np.any(previous)
        if delayed_response:
            camera.delay = True
            assert camera.request_started.wait(timeout=1)
        camera.failed = True
        robot.disconnect()
        robot.connect()
        reconnected = robot.get_observation()["front"]
        assert reconnected is not previous
        assert not np.any(reconnected)
        assert not robot._frame_ready.is_set()
        camera.release_request.set()
        previous_worker.join(timeout=1)
        assert not previous_worker.is_alive()
        assert not np.any(robot.get_observation()["front"])
        assert not robot._frame_ready.is_set()
        camera.delay = False
        camera.failed = False
        assert robot._frame_ready.wait(timeout=1)
        recovered = robot.get_observation()["front"]
        assert recovered is not previous
        assert np.any(recovered)
    finally:
        camera.release_request.set()
        if robot.is_connected:
            robot.disconnect()


def test_camera_failure_keeps_current_session_frame(robotd, tmp_path, monkeypatch):
    camera = SnapshotCamera()
    monkeypatch.setattr("lerobot.robots.microduck.microduck.requests.Session", lambda: camera)
    robot = MicroDuck(
        MicroDuckConfig(
            robotd_address=robotd.path,
            calibration_dir=tmp_path,
            camera_transport="snapshot",
            camera_timeout_s=0.01,
        )
    )
    try:
        robot.connect()
        camera.failed = True
        time.sleep(0.05)
        previous = robot.get_observation()["front"]
        time.sleep(0.25)
        assert robot.get_observation()["front"] is previous
        assert np.any(previous)
        robot.send_action({})
        camera.failed = False
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and robot.get_observation()["front"] is previous:
            time.sleep(0.01)
        assert robot.get_observation()["front"] is not previous
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    "rotate,expected", [(0, (2, 3)), (90, (3, 2)), (180, (2, 3)), (270, (3, 2))], ids=str
)
def test_upright_turns_clockwise_mounts_back(rotate, expected):
    pytest.importorskip("aiortc")
    from lerobot.robots.microduck.webrtc import upright

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = 255  # top-left
    turned = upright(frame, rotate)
    assert turned.shape[:2] == expected
    # A camera mounted a quarter turn clockwise sees its top-left where the upright view's top-right is.
    corner = {0: (0, 0), 90: (0, expected[1] - 1), 180: (1, 2), 270: (2, 0)}[rotate]
    assert turned[corner].tolist() == [255, 255, 255]


def test_config_rejects_unknown_camera_transport():
    with pytest.raises(ValueError, match="camera_transport"):
        MicroDuckConfig(camera_transport="mjpeg")


class FakeMediad:
    """webrtcsink's signalling server with an aiortc producer on loopback.

    The picture is landscape with a red top half, as the sideways-mounted camera sees an upright scene
    whose right side is red. ``rotate=None`` leaves media.video unanswered; ``refusals`` sessions are
    refused first, as when the robot's console holds its one session.
    """

    def __init__(self, rotate: int | None = 90, refusals: int = 0):
        self.rotate = rotate
        self.refusals = refusals
        self.sessions = 0
        self.ended = threading.Event()
        self.paused = threading.Event()
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._start(ready), self._loop).result(timeout=5)
        assert ready.is_set()

    async def _start(self, ready: threading.Event):
        from websockets.asyncio.server import serve

        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        ready.set()

    async def _handle(self, websocket):
        import av
        from aiortc import (
            RTCConfiguration,
            RTCPeerConnection,
            RTCRtpSender,
            RTCSessionDescription,
            VideoStreamTrack,
        )

        paused = self.paused

        class Landscape(VideoStreamTrack):
            async def recv(self):
                while paused.is_set():
                    await asyncio.sleep(0.01)
                pts, time_base = await self.next_timestamp()
                picture = np.zeros((180, 320, 3), dtype=np.uint8)
                picture[:90] = (255, 0, 0)
                frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
                frame.pts, frame.time_base = pts, time_base
                return frame

        pc = None
        await websocket.send(json.dumps({"type": "welcome", "peerId": "consumer-1"}))
        try:
            async for raw in websocket:
                message = json.loads(raw)
                if message["type"] == "list":
                    await websocket.send(json.dumps({"type": "list", "producers": [{"id": "robot-1"}]}))
                elif message["type"] == "startSession":
                    if self.refusals:
                        self.refusals -= 1
                        await websocket.send(json.dumps({"type": "error", "details": "session slot taken"}))
                        continue
                    self.sessions += 1
                    await websocket.send(
                        json.dumps({"type": "sessionStarted", "peerId": "robot-1", "sessionId": "s1"})
                    )
                    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                    channel = pc.createDataChannel("control")
                    sender = pc.addTrack(Landscape())
                    transceiver = next(t for t in pc.getTransceivers() if t.sender is sender)
                    transceiver.setCodecPreferences(
                        [
                            c
                            for c in RTCRtpSender.getCapabilities("video").codecs
                            if c.mimeType == "video/H264"
                        ]
                    )

                    def answer(text, channel=channel):
                        call = json.loads(text)
                        if call.get("method") == "media.video" and self.rotate is not None:
                            geometry = {"width": 320, "height": 180, "rotate": self.rotate}
                            channel.send(json.dumps({"jsonrpc": "2.0", "id": call["id"], "result": geometry}))

                    channel.on("message")(answer)
                    await pc.setLocalDescription(await pc.createOffer())
                    offer = {"type": "offer", "sdp": pc.localDescription.sdp}
                    await websocket.send(json.dumps({"type": "peer", "sessionId": "s1", "sdp": offer}))
                elif message["type"] == "peer" and message.get("sdp"):
                    await pc.setRemoteDescription(RTCSessionDescription(**message["sdp"]))
                elif message["type"] == "endSession":
                    self.ended.set()
        finally:
            if pc is not None:
                await pc.close()

    def close(self):
        async def stop():
            self._server.close()
            await self._server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def mediad(request):
    pytest.importorskip("aiortc")
    server = FakeMediad(**getattr(request, "param", {}))
    yield server
    server.close()


def assert_upright_red_right(frame: np.ndarray):
    height, width, _ = frame.shape
    # Lossy codec: dominance, not equality.
    assert frame[height // 2, width - 5, 0] > 200 and frame[height // 2, width - 5, 2] < 60
    assert frame[height // 2, 4].max() < 60


def webrtc_robot(robotd, mediad, tmp_path, **overrides):
    config = MicroDuckConfig(
        robotd_address=robotd.path,
        calibration_dir=tmp_path,
        remote_ip="127.0.0.1",
        signalling_port=mediad.port,
        camera_width=90,
        camera_height=160,
        **overrides,
    )
    return MicroDuck(config)


def test_webrtc_camera_streams_upright_frames_and_ends_its_session(robotd, mediad, tmp_path):
    robot = webrtc_robot(robotd, mediad, tmp_path)
    robot.connect()
    try:
        assert robot._frame_ready.wait(timeout=10)
        first = robot.get_observation()["front"]
        assert first.shape == (160, 90, 3)
        assert_upright_red_right(first)
        deadline = time.monotonic() + 5
        while robot.get_observation()["front"] is first and time.monotonic() < deadline:
            time.sleep(0.01)
        # Every decoded frame is a new array, so consumers can tell a fresh frame from a repeat.
        assert robot.get_observation()["front"] is not first
    finally:
        robot.disconnect()
    assert mediad.ended.wait(timeout=5)


@pytest.mark.parametrize("mediad", [{"refusals": 1}], indirect=True)
def test_webrtc_camera_retries_a_refused_session(robotd, mediad, tmp_path, caplog):
    robot = webrtc_robot(robotd, mediad, tmp_path)
    robot.connect()
    try:
        assert robot._frame_ready.wait(timeout=10)
        assert mediad.sessions == 1
        assert "session slot taken" in caplog.text
        assert_upright_red_right(robot.get_observation()["front"])
    finally:
        robot.disconnect()


@pytest.mark.parametrize("mediad", [{"rotate": None}], indirect=True)
def test_webrtc_camera_assumes_the_known_mount_without_media_video(mediad):
    from lerobot.robots.microduck.webrtc import receive_video

    frames = []
    stop = threading.Event()

    def on_frame(frame):
        frames.append(frame)
        stop.set()

    asyncio.run(
        receive_video(
            f"ws://127.0.0.1:{mediad.port}", on_frame, stop, stall_timeout_s=10, rotation_timeout_s=0.2
        )
    )
    assert frames[0].image.shape == (320, 180, 3)
    assert_upright_red_right(frames[0].image)


def test_webrtc_metadata_matches_cached_observation(robotd, mediad, tmp_path):
    robot = webrtc_robot(robotd, mediad, tmp_path)
    robot.connect()
    try:
        assert robot._frame_ready.wait(timeout=5)
        first = robot.get_camera_frame()
        assert first is not None and first.media_time_s is not None
        assert first.session_id == "s1"
        assert 0 <= time.monotonic() - first.received_at < 2
        mediad.paused.set()
        # Drain frames already in flight, then the same sample must keep its identity and age.
        time.sleep(0.2)
        cached = robot.get_camera_frame()
        time.sleep(0.1)
        assert robot.get_camera_frame() is cached
        assert robot.get_observation()["front"] is cached.image
        mediad.paused.clear()
        deadline = time.monotonic() + 2
        while robot.get_camera_frame() is cached and time.monotonic() < deadline:
            time.sleep(0.01)
        newer = robot.get_camera_frame()
        assert newer.sequence > cached.sequence
        assert newer.media_time_s > cached.media_time_s
        assert newer.received_at > cached.received_at
    finally:
        robot.disconnect()


def test_webrtc_stall_closes_session(mediad):
    from lerobot.robots.microduck.webrtc import receive_video

    frames = []

    def on_frame(frame):
        frames.append(frame)
        mediad.paused.set()

    with pytest.raises(ConnectionError, match="No camera frame"):
        asyncio.run(
            receive_video(
                f"ws://127.0.0.1:{mediad.port}",
                on_frame,
                threading.Event(),
                stall_timeout_s=1,
            )
        )
    assert frames
    assert mediad.ended.wait(timeout=2)


def test_webrtc_stop_during_signalling(monkeypatch):
    pytest.importorskip("aiortc")
    from contextlib import asynccontextmanager

    from lerobot.robots.microduck import webrtc

    entered = threading.Event()
    cancelled = threading.Event()
    stop = threading.Event()

    @asynccontextmanager
    async def never_connect(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
            yield
        finally:
            cancelled.set()

    monkeypatch.setattr(webrtc, "connect", never_connect)

    async def run():
        task = asyncio.create_task(
            webrtc.receive_video("ws://unused", lambda frame: None, stop, stall_timeout_s=30)
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())
    assert cancelled.is_set()


def test_camera_diagnostic_reports_h264_stream(mediad, tmp_path, capsys):
    from examples.microduck.check_camera import measure

    args = SimpleNamespace(
        host="127.0.0.1",
        transport="webrtc",
        signalling_port=mediad.port,
        seconds=0.3,
        timeout=5,
        min_fps=4,
        save_frame=tmp_path / "frame.png",
    )
    assert asyncio.run(measure(args))
    assert "PASS:" in capsys.readouterr().out
    assert args.save_frame.exists()
    frame = cv2.imread(str(args.save_frame))
    assert_upright_red_right(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_camera_placeholder_is_not_a_new_frame(robotd, tmp_path, monkeypatch):
    robot = MicroDuck(
        MicroDuckConfig(
            robotd_address=robotd.path,
            calibration_dir=tmp_path,
            camera_transport="snapshot",
            camera_timeout_s=0.01,
        )
    )
    monkeypatch.setattr(robot, "_camera_loop", lambda stop: stop.wait())
    robot.connect()
    try:
        assert robot.get_camera_frame() is None
        assert robot.get_observation()["front"] is robot.get_observation()["front"]
    finally:
        robot.disconnect()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_camera_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="camera_timeout_s"):
        MicroDuckConfig(camera_timeout_s=timeout)
