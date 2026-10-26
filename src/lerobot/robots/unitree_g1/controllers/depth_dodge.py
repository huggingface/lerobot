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

"""Depth-only dodgeball controller for the Unitree G1 (frozen SONIC + LoRA + Theia percept).

Runs the ONNX export of the ``dodge_theia`` policy: a frozen SONIC whole-body controller whose
decoder carries seven rank-16 LoRAs conditioned on a 128-d percept of the last 16 head-camera
depth frames. Three graphs, all from ``nepyope/g1_depth_dodge``:

- ``theia_image.onnx``   uint8 depth image ``[1,224,224]`` -> Theia-Tiny tokens ``[1,197,192]``
- ``perception.onnx``    16-slot token ring ``[1,16,197,192]`` -> ``percept[1,128]``
- ``actor.onnx``         ``tokenizer[640]``, ``policy[930]``, ``conditioning[768]`` -> ``actions[29]``

The reference stream is a held stand, so unlike :class:`SonicWholeBodyController` there is no
token input: the controller is autonomous once running. Depth arrives through
:meth:`observe_depth` at camera rate; with no camera the ring holds tokens of an all-black frame,
which is the report's "zeroed depth" ablation (the policy stands and never dodges).

Joint order: the policy was trained in mjlab, whose MuJoCo joint order is the Unitree SDK order
(``G1_29_JointIndex``), so no permutation is applied anywhere.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    NUM_MOTORS,
    G1_29_JointIndex,
    get_gravity_orientation,
    make_ort_session_options,
)
from ..unitree_g1 import RobotController
from .sonic_whole_body import DEFAULT_SONIC_REPO_ID, SonicWholeBodyController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz
HISTORY_LEN = 10  # proprioception frames in ``policy``
POLICY_DIM = HISTORY_LEN * (3 + 3 * NUM_MOTORS) + HISTORY_LEN * 3  # 930
FUTURE_STEPS = 10  # reference frames in ``tokenizer``
TOKENIZER_DIM = FUTURE_STEPS * (2 * NUM_MOTORS + 6)  # 640
PERCEPT_DIM = 128

DEPTH_HISTORY = 16  # token ring slots (0.64 s at 25 Hz)
DEPTH_SHAPE = (64, 96)  # what the policy was trained on
THEIA_SIZE = 224
THEIA_TOKENS = (197, 192)
DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0

DEFAULT_REPO_ID = "nepyope/g1_depth_dodge"
FILES = {"theia": "theia_image.onnx", "perception": "perception.onnx", "actor": "actor.onnx"}
# "dodge": LoRA'd actor conditioned on the depth percept. "sonic": lerobot's stock
# SonicWholeBodyController fed its neutral token (the idle stand), for A/B on the robot.
MODES = ("dodge", "sonic")


def depth_to_theia_image(
    depth_m: np.ndarray, near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M
) -> np.ndarray:
    """Optical-Z depth in metres ``(64, 96)`` -> uint8 ``(224, 224)`` in Theia's contract.

    255 is near, 1 is far, 0 is invalid (no return or out of range); then bilinear resize with
    ``align_corners=False`` and round. Mirrors ``dodge_theia.vision.tokenize`` (whose depth is
    fp16-rounded first, as the simulator did).
    """
    import torch
    from torch.nn.functional import interpolate

    depth = torch.as_tensor(np.asarray(depth_m), dtype=torch.float32).half().float()
    if depth.shape != DEPTH_SHAPE:
        raise ValueError(f"depth must be {DEPTH_SHAPE} metres, got {tuple(depth.shape)}")
    valid = torch.isfinite(depth) & (depth >= near) & (depth <= far)
    value = (1.0 + 254.0 * (far - depth) / (far - near)).clamp(1.0, 255.0).round()
    gray = torch.where(valid, value, torch.zeros_like(value))
    x = interpolate(gray[None, None], size=(THEIA_SIZE, THEIA_SIZE), mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)[0, 0].numpy()


def load_sonic_constants(repo_id: str = DEFAULT_SONIC_REPO_ID) -> dict[str, np.ndarray]:
    """kp / kd / default_angles / action_scale from the SONIC decoder's ONNX metadata (SDK joint
    order). The dodge policy was trained on these exact values."""
    path = hf_hub_download(repo_id=repo_id, filename="model_decoder.onnx")
    model = onnx.load(path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    keys = ("kp", "kd", "default_angles", "action_scale")
    missing = [k for k in keys if k not in metadata]
    if missing:
        raise ValueError(f"{repo_id} ONNX metadata is missing {missing}")
    return {k: np.array(json.loads(metadata[k]), dtype=np.float32) for k in keys}


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _yaw_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw-only part of a (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


class DepthDodgeController(RobotController):
    """Autonomous depth-dodge controller for UnitreeG1's background control thread.

    Each 50 Hz tick: push the lowstate into the 10-frame proprio history, build SONIC's
    ``policy[930]`` and held-stand ``tokenizer[640]`` streams, and run the LoRA'd actor with the
    current percept. Call :meth:`observe_depth` from the camera thread (25 Hz) to advance the
    16-slot Theia token ring; the percept is recomputed on the next tick.
    """

    control_dt = CONTROL_DT
    timing: dict[str, float]
    depth_ring: deque[np.ndarray]
    _ref_quat: np.ndarray | None

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        intra_op_threads: int = 2,
        theia_threads: int = 4,
        mode: str = "dodge",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        constants = load_sonic_constants()
        self.kp, self.kd = constants["kp"], constants["kd"]
        self.default_angles, self.action_scale = constants["default_angles"], constants["action_scale"]

        # Theia runs at camera rate on its own thread and is the heavy graph (~36 ms on a Jetson
        # Orin CPU with 4 threads); the 50 Hz actor/perception pair stays small like SONIC's decoder.
        def session(name: str, threads: int) -> ort.InferenceSession:
            so = make_ort_session_options(intra_op_num_threads=threads, inter_op_num_threads=1)
            return ort.InferenceSession(
                hf_hub_download(repo_id, FILES[name]), sess_options=so, providers=["CPUExecutionProvider"]
            )

        self.theia = session("theia", theia_threads)
        self.perception = session("perception", intra_op_threads)
        self.actor = session("actor", intra_op_threads)
        logger.info(f"Loaded depth dodge ONNX graphs from {repo_id}")

        # Stock SONIC for "sonic" mode: with no token in the action it seeds ``neutral_token``
        # and decodes a stand, exactly as a fresh SonicWholeBodyController does on the robot.
        self.sonic = SonicWholeBodyController()
        self._mode = mode

        # Tokens of an all-black frame (no camera / no return anywhere): the ring's seed.
        self.black_tokens = self._tokenize(np.zeros(DEPTH_SHAPE, np.float32))

        # No action input: SONIC's token is replaced by the held-stand reference built in-tick.
        self.action_ft: dict[str, type] = {}
        self.reset()
        logger.info("DepthDodgeController initialized")

    # ---- depth --------------------------------------------------------------------------

    def _tokenize(self, depth_m: np.ndarray) -> np.ndarray:
        image = depth_to_theia_image(depth_m)[None]
        tokens = self.theia.run(None, {"image": image})[0][0]
        # Training cached Theia features in fp16; match that rounding.
        return tokens.astype(np.float16).astype(np.float32)

    def observe_depth(self, depth_m: np.ndarray) -> None:
        """Ingest one ``(64, 96)`` depth frame in metres (0 = no return). Thread-safe enough for
        a single camera producer: the ring is swapped atomically as a whole."""
        t0 = time.perf_counter()
        self.observe_tokens(self._tokenize(depth_m))
        self.timing["theia_ms"] = 0.8 * self.timing["theia_ms"] + 0.2 * (time.perf_counter() - t0) * 1e3

    def observe_tokens(self, tokens: np.ndarray) -> None:
        """Ingest Theia tokens ``(197, 192)`` computed elsewhere (e.g. on the Jetson GPU by the
        depth_shm writer); pushes one ring slot exactly like :meth:`observe_depth`."""
        if tokens.shape != THEIA_TOKENS:
            raise ValueError(f"tokens must be {THEIA_TOKENS}, got {tuple(tokens.shape)}")
        ring = self.depth_ring.copy()
        ring.append(np.ascontiguousarray(tokens, dtype=np.float32))
        self.depth_ring = ring
        self._percept_stale = True
        self.ring_times.append(time.perf_counter())

    @property
    def ring_span_s(self) -> float:
        """Wall time covered by the 16-slot ring (training: 16 frames at 25 Hz = 0.64 s)."""
        return self.ring_times[-1] - self.ring_times[0] if len(self.ring_times) >= 2 else 0.0

    def _percept(self) -> np.ndarray:
        if self._percept_stale:
            ring = np.stack(list(self.depth_ring))[None]  # [1,16,197,192]
            self._cached_percept = self.perception.run(None, {"depth_tokens": ring})[0]
            self._percept_stale = False
        return self._cached_percept

    # ---- control ------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {value!r}")
        if value == "sonic" and self._mode != "sonic":
            # Fresh start for the stock controller: zero history + neutral token on its first tick.
            self.sonic.reset()
        if value == "dodge" and self._mode != "dodge":
            # The training reference is world-fixed, so the actor holds whatever heading it is
            # anchored to. Re-anchor to the current heading rather than the one at connect time.
            self._ref_quat = None
        self._mode = value

    def reset(self) -> None:
        """Drop proprio history, refill the depth ring from black, forget the heading reference."""
        self.sonic.reset()
        self.last_action = np.zeros(NUM_MOTORS, np.float32)
        self.h_ang: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.h_q: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.h_dq: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.h_act: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.h_grav: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self.depth_ring = deque([self.black_tokens] * DEPTH_HISTORY, maxlen=DEPTH_HISTORY)
        self.ring_times: deque[float] = deque(maxlen=DEPTH_HISTORY)
        # EMA timings (ms) for the run log: Theia per frame, actor+perception per tick, tick period.
        self.timing = {"theia_ms": 0.0, "tick_ms": 0.0, "period_ms": 0.0}
        self._last_tick_t: float | None = None
        self._percept_stale = True
        self._cached_percept = None
        self._ref_quat = None  # heading the stand reference is anchored to

    def _tokenizer(self, quat: np.ndarray) -> np.ndarray:
        """Held-stand reference: SONIC's flat chop of 10 future frames of (joint pos, joint vel)
        = 5 rows of default pose then 5 rows of zeros, each row followed by the 6-D
        body-frame reference orientation. Verified bit-for-bit against the training env."""
        if self._ref_quat is None:
            # Anchor the reference heading to where the robot stands at reset (the sim starts at
            # yaw 0, so this reproduces training there and keeps a real robot from turning back
            # to IMU yaw zero).
            self._ref_quat = _yaw_quat(quat)
        # rot_dif = quat_inv(robot) * ref; 6-D = first two rotation-matrix columns, row-major.
        rel = _quat_mul(np.array([quat[0], -quat[1], -quat[2], -quat[3]], np.float32), self._ref_quat)
        ori = _quat_to_matrix(rel / (np.linalg.norm(rel) + 1e-8))[:, :2].reshape(-1)

        chop = np.concatenate(
            [np.tile(self.default_angles, FUTURE_STEPS), np.zeros(FUTURE_STEPS * NUM_MOTORS)]
        )
        rows = chop.reshape(FUTURE_STEPS, 2 * NUM_MOTORS).astype(np.float32)
        return np.concatenate([rows, np.tile(ori, (FUTURE_STEPS, 1))], axis=1).reshape(-1)

    def run_step(self, action: dict, lowstate) -> dict:
        t0 = time.perf_counter()
        if self._last_tick_t is not None:
            self.timing["period_ms"] = 0.9 * self.timing["period_ms"] + 0.1 * (t0 - self._last_tick_t) * 1e3
        self._last_tick_t = t0
        try:
            return self._run_step(lowstate)
        finally:
            self.timing["tick_ms"] = 0.9 * self.timing["tick_ms"] + 0.1 * (time.perf_counter() - t0) * 1e3

    def _run_step(self, lowstate) -> dict:
        q = np.array([lowstate.motor_state[m.value].q for m in G1_29_JointIndex], np.float32)
        dq = np.array([lowstate.motor_state[m.value].dq for m in G1_29_JointIndex], np.float32)
        quat = np.array(lowstate.imu_state.quaternion, np.float32)
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        ang = np.array(lowstate.imu_state.gyroscope, np.float32)

        frames = (ang, q - self.default_angles, dq, self.last_action.copy(), get_gravity_orientation(quat))
        for hist, frame in zip(
            (self.h_ang, self.h_q, self.h_dq, self.h_act, self.h_grav), frames, strict=True
        ):
            if not hist:  # training env fills the whole history from the first frame on reset
                hist.extend([frame] * HISTORY_LEN)
            else:
                hist.append(frame)

        if self._mode == "sonic":
            # Stock SONIC on its neutral token; no token keys in ``action`` -> it holds the idle
            # latent. Its residual comes back in its own joint order; keep ours (SDK order) fed so
            # switching to dodge starts from a warm, consistent action history.
            targets = self.sonic.run_step({}, lowstate)
            self.last_action = self.sonic.last_action_mj[ISAACLAB_TO_MUJOCO].astype(np.float32)
            return targets

        policy = np.concatenate(
            [np.concatenate(list(h)) for h in (self.h_ang, self.h_q, self.h_dq, self.h_act, self.h_grav)]
        )
        tokenizer = self._tokenizer(quat)
        conditioning = np.concatenate([tokenizer, self._percept()[0]])
        residual = self.actor.run(
            None,
            {
                "tokenizer": tokenizer[None].astype(np.float32),
                "policy": policy[None].astype(np.float32),
                "conditioning": conditioning[None].astype(np.float32),
            },
        )[0][0].astype(np.float32)
        self.last_action = residual
        target = self.default_angles + residual * self.action_scale
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}
