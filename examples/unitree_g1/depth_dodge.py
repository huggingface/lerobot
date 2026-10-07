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

"""Depth-only ball dodging on the Unitree G1: frozen SONIC + decoder LoRAs + a Theia percept.

The policy (https://huggingface.co/nepyope/g1_depth_dodge) is a frozen SONIC whole-body controller
whose decoder carries seven rank-16 LoRAs, conditioned on a 128-D percept of the last 16 head-camera
depth frames. It runs as three ONNX graphs:

- ``theia_image.onnx``  uint8 depth image ``[1,224,224]`` -> frozen Theia-Tiny tokens ``[1,197,192]``
- ``perception.onnx``   16-slot token ring ``[1,16,197,192]`` -> ``percept[1,128]``
- ``actor.onnx``        ``tokenizer[640]``, ``policy[930]``, ``conditioning[768]`` -> ``actions[29]``

Theia runs at camera rate (25 Hz) on the camera thread; the perception + actor pair runs at 50 Hz on
the robot's controller thread. The reference is a held stand, so the controller takes no action input
and needs no policy server: it stands, and steps out of the way of balls it sees.

Everything runs on the G1's Jetson Orin, on the CPU: the head D435i is read directly with
pyrealsense2, and joint commands go through ``run_g1_server.py`` on the same machine, so nothing in
the control loop crosses the network. With ``--camera none`` the ring holds all-black frames: the
policy stands and never dodges.

Setup on the robot (see docs/source/unitree_g1.mdx for networking and unitree_sdk2_python):

```bash
ssh unitree@192.168.123.164
cd lerobot && pip install -e ".[unitree_g1,intelrealsense]"

# Terminal 1: DDS <-> ZMQ bridge. No --camera: this example opens the head RealSense itself.
python src/lerobot/robots/unitree_g1/run_g1_server.py

# Terminal 2, with the robot hanging on its gantry
python examples/unitree_g1/depth_dodge.py --real --camera realsense
```

The weights download from the Hub on first run; on a robot without internet, run the script once
elsewhere and copy the Hugging Face cache over. The camera stream is cropped to the policy's 45 deg
vertical FOV at 0 deg pitch, so the D435i must be mounted level. Throws should come from the front,
1.8-2.8 m away. Keep the robot on the gantry until it stands steadily, and stop with Ctrl+C.

To try it without a robot, in the MuJoCo simulation (no camera, so it only stands):

```bash
python examples/unitree_g1/depth_dodge.py --camera none --onscreen
```

Theia-Tiny is released for non-commercial research use only; see the model card before deploying.
"""

import argparse
import json
import logging
import math
import threading
import time
from collections import deque

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from torch.nn.functional import interpolate

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    G1_29_JointIndex,
    get_gravity_orientation,
    make_ort_session_options,
)

logger = logging.getLogger(__name__)

DODGE_REPO_ID = "nepyope/g1_depth_dodge"
DODGE_REVISION = "10e207cb734a6c3edf792d4995ca59ae4641bf29"
SONIC_REPO_ID = "lerobot/sonic_decoder"
SONIC_REVISION = "15e87db84e7f6df8b8440f1fdbd0a5d17f89d531"  # the decoder the LoRAs were trained on

CONTROL_DT = 0.02  # 50 Hz
HISTORY_LEN = 10  # proprioception frames in ``policy``
FUTURE_STEPS = 10  # reference frames in ``tokenizer``

CAMERA_HZ = 25.0
DEPTH_HISTORY = 16  # token ring slots, 0.64 s at 25 Hz
DEPTH_SHAPE = (64, 96)  # optical-Z metres, 45 deg vertical FOV
POLICY_VFOV = math.radians(45.0)
THEIA_SIZE = 224
DEPTH_NEAR_M, DEPTH_FAR_M = 0.2, 6.0


def depth_to_theia_image(depth_m: np.ndarray) -> np.ndarray:
    """``(64, 96)`` metres -> uint8 ``(224, 224)``: 255 near, 1 far, 0 invalid, bilinearly resized.

    Depth and tokens are fp16-rounded because the training simulator rendered and cached them in fp16.
    """
    depth = torch.as_tensor(depth_m, dtype=torch.float32).half().float()
    valid = torch.isfinite(depth) & (depth >= DEPTH_NEAR_M) & (depth <= DEPTH_FAR_M)
    value = (1.0 + 254.0 * (DEPTH_FAR_M - depth) / (DEPTH_FAR_M - DEPTH_NEAR_M)).clamp(1.0, 255.0).round()
    gray = torch.where(valid, value, torch.zeros_like(value))
    image = interpolate(gray[None, None], size=(THEIA_SIZE, THEIA_SIZE), mode="bilinear", align_corners=False)
    return image.round().clamp(0, 255).to(torch.uint8)[0, 0].numpy()


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


class DepthDodgeController:
    """``RobotController`` for ``UnitreeG1``'s 50 Hz controller thread.

    Joint order: the policy was trained in mjlab, whose joint order is the Unitree SDK order
    (``G1_29_JointIndex``), and the SONIC deploy constants are stored in that order too, so no
    permutation is applied anywhere.
    """

    control_dt = CONTROL_DT

    def __init__(self):
        sonic_path = hf_hub_download(SONIC_REPO_ID, "model_decoder.onnx", revision=SONIC_REVISION)
        metadata = {p.key: p.value for p in onnx.load(sonic_path, load_external_data=False).metadata_props}
        constants = {
            k: np.array(json.loads(metadata[k]), dtype=np.float32)
            for k in ("kp", "kd", "default_angles", "action_scale")
        }
        self.kp, self.kd = constants["kp"], constants["kd"]
        self.default_angles, self.action_scale = constants["default_angles"], constants["action_scale"]

        def session(filename: str, threads: int) -> ort.InferenceSession:
            path = hf_hub_download(DODGE_REPO_ID, filename, revision=DODGE_REVISION)
            options = make_ort_session_options(intra_op_num_threads=threads, inter_op_num_threads=1)
            return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])

        # Theia is the heavy graph (~36 ms on a Jetson Orin CPU with 4 threads) and runs off the control thread.
        self.theia = session("theia_image.onnx", 4)
        self.perception = session("perception.onnx", 2)
        self.actor = session("actor.onnx", 2)

        self.black_tokens = self._tokenize(np.zeros(DEPTH_SHAPE, np.float32))
        self.action_ft: dict[str, type] = {}  # autonomous: takes over the action space with nothing
        self._ring_lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(NUM_MOTORS, np.float32)
        self.history = [deque(maxlen=HISTORY_LEN) for _ in range(5)]  # ang vel, q, dq, last action, gravity
        with self._ring_lock:
            self.ring = deque([self.black_tokens] * DEPTH_HISTORY, maxlen=DEPTH_HISTORY)
            self._ring_version = 0
        self._percept_version = -1
        self._percept = None
        self._ref_quat = None

    def _tokenize(self, depth_m: np.ndarray) -> np.ndarray:
        tokens = self.theia.run(None, {"image": depth_to_theia_image(depth_m)[None]})[0][0]
        return tokens.astype(np.float16).astype(np.float32)

    def observe_depth(self, depth_m: np.ndarray) -> None:
        """Push one ``(64, 96)`` depth frame in metres (0 = no return). Call at 25 Hz."""
        tokens = self._tokenize(depth_m)
        with self._ring_lock:
            ring = self.ring.copy()
            ring.append(tokens)
            self.ring = ring
            self._ring_version += 1

    def _get_percept(self) -> np.ndarray:
        with self._ring_lock:
            version = self._ring_version
            ring = np.stack(self.ring)[None] if version != self._percept_version else None
        if ring is not None:
            self._percept = self.perception.run(None, {"depth_tokens": ring})[0][0]
            self._percept_version = version
        return self._percept

    def _tokenizer(self, quat: np.ndarray) -> np.ndarray:
        """Held-stand reference: 5 frames of default pose then 5 of zero velocity, each followed by
        the 6-D body-frame reference orientation, anchored to the heading at reset."""
        if self._ref_quat is None:
            self._ref_quat = yaw_quat(quat)
        rel = quat_mul(np.array([quat[0], -quat[1], -quat[2], -quat[3]], np.float32), self._ref_quat)
        orientation = quat_to_matrix(rel / (np.linalg.norm(rel) + 1e-8))[:, :2].reshape(-1)
        rows = np.concatenate(
            [np.tile(self.default_angles, FUTURE_STEPS), np.zeros(FUTURE_STEPS * NUM_MOTORS, np.float32)]
        ).reshape(FUTURE_STEPS, 2 * NUM_MOTORS)
        return np.concatenate([rows, np.tile(orientation, (FUTURE_STEPS, 1))], axis=1).reshape(-1)

    def run_step(self, action: dict, lowstate) -> dict:
        q = np.array([lowstate.motor_state[j.value].q for j in G1_29_JointIndex], np.float32)
        dq = np.array([lowstate.motor_state[j.value].dq for j in G1_29_JointIndex], np.float32)
        quat = np.array(lowstate.imu_state.quaternion, np.float32)
        quat /= np.linalg.norm(quat) + 1e-8
        gyro = np.array(lowstate.imu_state.gyroscope, np.float32)

        frames = (gyro, q - self.default_angles, dq, self.last_action, get_gravity_orientation(quat))
        for hist, frame in zip(self.history, frames, strict=True):
            hist.extend([frame] * (HISTORY_LEN if not hist else 1))  # training fills history on reset

        tokenizer = self._tokenizer(quat)
        policy = np.concatenate([np.concatenate(list(h)) for h in self.history])
        conditioning = np.concatenate([tokenizer, self._get_percept()])
        residual = self.actor.run(
            None,
            {
                "tokenizer": tokenizer[None].astype(np.float32),
                "policy": policy[None].astype(np.float32),
                "conditioning": conditioning[None].astype(np.float32),
            },
        )[0][0].astype(np.float32)
        if not np.isfinite(residual).all():
            raise FloatingPointError("Non-finite dodge action")
        self.last_action = residual
        target = self.default_angles + self.action_scale * residual
        return {f"{j.name}.q": float(target[j.value]) for j in G1_29_JointIndex}


def make_depth_crop(height: int, width: int, vfov_deg: float):
    """Crop a wider depth image to the policy's 45 deg FOV and downsample to ``(64, 96)`` metres.

    Nearest-neighbour keeps holes as holes (0 = no return), like the simulator's rendered depth.
    """
    f = (height / 2) / math.tan(math.radians(vfov_deg) / 2)
    half_h = f * math.tan(POLICY_VFOV / 2)
    half_w = half_h * DEPTH_SHAPE[1] / DEPTH_SHAPE[0]
    y0, y1 = max(round(height / 2 - half_h), 0), min(round(height / 2 + half_h), height)
    x0, x1 = max(round(width / 2 - half_w), 0), min(round(width / 2 + half_w), width)

    def crop(depth_mm: np.ndarray) -> np.ndarray:
        roi = np.squeeze(depth_mm)[y0:y1, x0:x1]
        small = cv2.resize(roi, (DEPTH_SHAPE[1], DEPTH_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
        return small.astype(np.float32) * 1e-3

    return crop


def camera_loop(controller: DepthDodgeController, args, stop: threading.Event) -> None:
    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    camera = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=args.camera_serial,
            fps=60,
            width=480,
            height=270,
            use_rgb=False,
            use_depth=True,
        )
    )
    camera.connect()
    crop = make_depth_crop(270, 480, args.camera_vfov)
    next_t = time.monotonic()
    try:
        while not stop.is_set():
            depth_mm = camera.async_read_depth(timeout_ms=1000)
            # Sample the 60 Hz stream at the 25 Hz the ring was trained on.
            now = time.monotonic()
            if now < next_t:
                continue
            next_t = max(next_t + 1.0 / CAMERA_HZ, now)
            controller.observe_depth(crop(depth_mm))
    finally:
        camera.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", action="store_true", help="drive the real robot instead of the MuJoCo sim")
    parser.add_argument("--robot_ip", default="127.0.0.1", help="run_g1_server.py host; local on the Jetson")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "none"])
    parser.add_argument(
        "--camera_serial", default="", help="RealSense serial number or name; empty = first found"
    )
    parser.add_argument(
        "--camera_vfov", type=float, default=58.0, help="depth stream vertical FOV (D435i: 58)"
    )
    parser.add_argument("--onscreen", action="store_true", help="show the MuJoCo viewer (sim only)")
    parser.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = UnitreeG1Config(
        is_simulation=not args.real,
        robot_ip=args.robot_ip,
        end_effector="dummy",
        sim_publish_images=False,
        sim_onscreen=args.onscreen if not args.real else None,
    )
    robot = UnitreeG1(config)
    controller = DepthDodgeController()
    robot.controller = controller  # not in UnitreeG1's name registry, so attach it before connect()

    stop = threading.Event()
    camera_thread = None
    try:
        robot.connect()
        if args.camera == "realsense":
            camera_thread = threading.Thread(target=camera_loop, args=(controller, args, stop), daemon=True)
            camera_thread.start()

        # The sim holds the robot up on an elastic band; release it once the stand has settled.
        sim = getattr(robot.sim_env, "simulator", None)
        band = getattr(getattr(sim, "sim_env", None), "elastic_band", None)
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            if band is not None and time.monotonic() - start > 1.0:
                band.enable = False
                band = None
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)
        robot.disconnect()


if __name__ == "__main__":
    main()
