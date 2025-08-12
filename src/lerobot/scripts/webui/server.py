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

from __future__ import annotations

import atexit
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from lerobot.cameras.kinect.configuration_kinect import KinectCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,  # noqa: F401
)
from lerobot.robots import Robot, RobotConfig, make_robot_from_config


@dataclass
class WebUIConfig:
    robot: RobotConfig
    host: str = "127.0.0.1"
    port: int = 9091
    stream_fps: int = 15


def _is_image_feature(value: Any) -> bool:
    try:
        return isinstance(value, tuple) and len(value) == 3 and int(value[-1]) in (1, 3, 4)
    except Exception:
        return False


def _to_builtin_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin_types(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, type):
        return obj.__name__
    return obj


def run_webui(robot: Robot, host: str, port: int, stream_fps: int) -> None:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir.resolve()))

    state_lock = threading.Lock()
    connected = True

    def _cleanup():
        nonlocal connected
        with state_lock:
            if connected:
                try:
                    robot.disconnect()
                except Exception:
                    pass
                connected = False

    atexit.register(_cleanup)

    @app.route("/")
    def index():
        return render_template("robot_ui.html")

    @app.route("/api/status", methods=["GET"])  # lightweight status
    def api_status():
        with state_lock:
            return jsonify({"connected": connected, "robot_name": getattr(robot, "name", "unknown")})

    @app.route("/api/features", methods=["GET"])  # features for client UI
    def api_features():
        obs_ft = getattr(robot, "observation_features", {})
        act_ft = getattr(robot, "action_features", {})
        camera_keys = [k for k, v in obs_ft.items() if _is_image_feature(v)]
        return jsonify(
            {
                "observation_features": _to_builtin_types(obs_ft),
                "action_features": _to_builtin_types(act_ft),
                "camera_keys": camera_keys,
            }
        )

    @app.route("/api/action", methods=["POST"])  # send an action dict to the robot
    def api_action():
        nonlocal connected
        if not connected:
            return jsonify({"error": "robot not connected"}), 400
        data = request.get_json(silent=True) or {}
        action = data.get("action", {})
        if not isinstance(action, dict):
            return jsonify({"error": "'action' must be a JSON object"}), 400
        try:
            sent = robot.send_action(action)
            return jsonify({"sent": _to_builtin_types(sent)})
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    @app.route("/stream/<cam_key>")  # MJPEG stream for a camera key
    def stream_camera(cam_key: str):
        nonlocal connected
        if not connected:
            return Response(status=404)
        camera = getattr(robot, "cameras", {}).get(cam_key)
        if camera is None:
            def obs_frames():
                dt = max(1.0 / float(stream_fps), 1e-3)
                while True:
                    try:
                        obs = robot.get_observation()
                        frame = obs.get(cam_key)
                        if frame is None:
                            time.sleep(dt)
                            continue
                        if not isinstance(frame, np.ndarray):
                            time.sleep(dt)
                            continue
                        if frame.ndim == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        ok, jpg = cv2.imencode(".jpg", frame)
                        if not ok:
                            time.sleep(dt)
                            continue
                        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                        time.sleep(dt)
                    except GeneratorExit:
                        break
                    except Exception:
                        time.sleep(dt)
            return Response(obs_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

        def cam_frames():
            dt = max(1.0 / float(stream_fps), 1e-3)
            while True:
                try:
                    frame = camera.async_read()
                    if frame is None:
                        time.sleep(dt)
                        continue
                    if frame.ndim == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    ok, jpg = cv2.imencode(".jpg", frame)
                    if not ok:
                        time.sleep(dt)
                        continue
                    yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                    time.sleep(dt)
                except GeneratorExit:
                    break
                except Exception:
                    time.sleep(dt)
        return Response(cam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    app.run(host=host, port=port, threaded=True)


@draccus.wrap()
def serve(cfg: WebUIConfig):
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        run_webui(robot=robot, host=cfg.host, port=cfg.port, stream_fps=cfg.stream_fps)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def main() -> None:
    serve()


if __name__ == "__main__":
    main()






