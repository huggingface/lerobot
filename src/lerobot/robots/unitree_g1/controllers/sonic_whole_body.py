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

"""SONIC full-body controller for Unitree G1."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

from lerobot.utils.import_utils import _onnxruntime_available, require_package

from ..g1_utils import lowstate_to_obs
from .sonic_pipeline import (
    CONTROL_DT,
    DEBUG_PRINT_EVERY,
    DEFAULT_ANGLES,
    ENCODER_UPDATE_EVERY,
    LM,
    MOTION_SETS,
    MovementState,
    PlannerController,
    SonicPlanner,
    clamp_mode_params,
    compute_kp_kd,
    ort_providers,
    process_joystick,
    should_replan_request,
    snapshot_ms,
)

if TYPE_CHECKING or _onnxruntime_available:
    import onnxruntime as ort
else:
    ort = None

logger = logging.getLogger(__name__)

# Length of the flattened SONIC whole-body reference window
# (10 frames x 24 SMPL joints x 3 coords). Matches smpl_joints_10frame_step1.
SMPL_ACTION_DIM = 720
# Prefix for per-element SMPL floats carried on the teleop action dict.
SMPL_ACTION_PREFIX = "smpl."
# Optional per-frame SMPL root orientation (wxyz) for the mode-2 anchor.
ROOT_ACTION_DIM = 4
ROOT_ACTION_PREFIX = "root."


def _extract_smpl_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a (720,) SMPL window from ``smpl.{i}`` action keys, or None.

    The pico_headset teleoperator emits the whole-body reference as flat floats so
    it flows unchanged through the standard lerobot action pipeline.
    """
    if not action or f"{SMPL_ACTION_PREFIX}0" not in action:
        return None
    arr = np.fromiter(
        (float(action.get(f"{SMPL_ACTION_PREFIX}{i}", 0.0)) for i in range(SMPL_ACTION_DIM)),
        dtype=np.float32,
        count=SMPL_ACTION_DIM,
    )
    return arr


def _extract_root_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a (4,) SMPL root quaternion (wxyz) from ``root.{i}`` keys, or None."""
    if not action or f"{ROOT_ACTION_PREFIX}0" not in action:
        return None
    q = np.fromiter(
        (float(action.get(f"{ROOT_ACTION_PREFIX}{i}", 0.0)) for i in range(ROOT_ACTION_DIM)),
        dtype=np.float32,
        count=ROOT_ACTION_DIM,
    )
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return None
    return q / n


class SonicRuntime:
    """Shared SONIC control loop state (standalone demo + locomotion controller)."""

    def __init__(self, force_cpu: bool = False):
        require_package("onnxruntime", extra="unitree_g1")
        planner_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="planner_sonic.onnx")
        encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        providers = ort_providers(force_cpu=force_cpu)
        self.use_gpu = providers[0] == "CUDAExecutionProvider"
        so = ort.SessionOptions()
        so.log_severity_level = 3

        planner_sess = ort.InferenceSession(planner_path, sess_options=so, providers=providers)
        encoder_sess = ort.InferenceSession(encoder_path, sess_options=so, providers=providers)
        decoder_sess = ort.InferenceSession(decoder_path, sess_options=so, providers=providers)

        self.kp, self.kd = compute_kp_kd()
        self.ms = MovementState()
        self.planner = SonicPlanner(planner_sess, planner_path)
        self.controller = PlannerController(self.planner, encoder_sess, decoder_sess)

        motion = self.planner.initialize(DEFAULT_ANGLES, self.ms)
        self.controller.load_initial_motion(motion)
        self.planner.start_subprocess(self.controller, use_gpu=self.use_gpu)

        self.step = 0
        self.replan_timer = 0.0
        self.last_ms = snapshot_ms(self.ms)

    @property
    def pipeline(self):
        return self.controller

    def tick(self, obs: dict, *, debug: bool | None = None, use_joystick: bool = True) -> dict:
        if not obs:
            self.step += 1
            return {}

        if use_joystick:
            process_joystick(obs, self.ms, self.controller)
        clamp_mode_params(self.ms)

        if self.step > 0:
            self.replan_timer += CONTROL_DT
        if should_replan_request(self.ms, self.last_ms, self.replan_timer, self.step):
            self.planner.request_replan(self.controller.ref_cursor, self.ms)
            self.replan_timer = 0.0
            self.ms.needs_replan = False
            self.last_ms = snapshot_ms(self.ms)

        do_enc = self.step % ENCODER_UPDATE_EVERY == 0
        if debug is None:
            debug = self.step % DEBUG_PRINT_EVERY == 0
        action = self.controller.step(obs, update_encoder=do_enc, debug=debug)

        result = self.planner.try_get_new_motion()
        if result:
            self.controller.blend_new_motion(*result)

        self.controller.advance_cursor()
        self.step += 1
        return action

    def reset(self):
        self.ms = MovementState()
        self.controller.reinit_heading = True
        self.controller.playing = True
        self.step = 0
        self.replan_timer = 0.0
        self.last_ms = snapshot_ms(self.ms)

    def shutdown(self):
        self.planner.stop_subprocess()


class SonicWholeBodyController:
    """Full-body SONIC controller for UnitreeG1's background controller thread."""

    control_dt = CONTROL_DT
    full_body = True

    def __init__(self, force_cpu: bool = False):
        logger.info("Loading SONIC whole-body controller...")
        self._runtime = SonicRuntime(force_cpu=force_cpu)
        self.kp = self._runtime.kp
        self.kd = self._runtime.kd
        self.controller = self._runtime.controller
        self.ms = self._runtime.ms

        # Optional: subscribe directly to the rt/smpl headset stream so full-body
        # teleop works with ANY teleoperator (e.g. --teleop.type=unitree_g1 for the
        # estop/joystick) before the dedicated pico_headset teleop exists. Enable
        # with SONIC_SMPL_STREAM=1; override endpoint via SONIC_SMPL_HOST/PORT.
        self._smpl_stream = None
        if os.environ.get("SONIC_SMPL_STREAM", "0") not in ("0", "", "false", "False"):
            self._init_smpl_stream()

        logger.info(
            "SONIC ready: %s (default mode: %s, smpl_stream=%s)",
            MOTION_SETS[0][0],
            LM(self.ms.mode).name,
            self._smpl_stream is not None,
        )

    def _init_smpl_stream(self) -> None:
        # Lazy import so the zmq dependency is only required when streaming is on.
        from lerobot.robots.unitree_g1.smpl_stream import (
            DEFAULT_SMPL_HOST,
            DEFAULT_SMPL_PORT,
            SmplStream,
        )

        host = os.environ.get("SONIC_SMPL_HOST", DEFAULT_SMPL_HOST)
        port = int(os.environ.get("SONIC_SMPL_PORT", DEFAULT_SMPL_PORT))
        self._smpl_stream = SmplStream(host=host, port=port)
        logger.info("SONIC subscribed to rt/smpl @ tcp://%s:%d", host, port)

    def _enter_wholebody(self) -> None:
        """Switch into SMPL whole-body tracking (encode_mode 2)."""
        self.controller.encode_mode = 2
        self.controller.reinit_heading = True
        logger.info("SONIC: SMPL stream active -> whole-body tracking (mode 2)")

    def _exit_wholebody(self) -> None:
        """Revert to locomotion/standing (encode_mode 0) after SMPL is lost.

        Mirrors the 'M' toggle in sonic.py so the handoff is clean: the robot holds
        a standing reference and (if a joystick teleop is attached) can be driven.
        """
        self.controller.encode_mode = 0
        self.controller.playing = True
        self.controller.reinit_heading = True
        self.ms.needs_replan = True
        logger.warning("SONIC: SMPL stream lost/stale -> reverting to locomotion (standing)")

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}
        obs = lowstate_to_obs(lowstate)

        # Prefer SMPL delivered via the teleop action (pico_headset). Fall back to a
        # direct rt/smpl subscription when SONIC_SMPL_STREAM is enabled. A stale
        # stream (headset silent past its timeout) is treated as "no SMPL" so the
        # robot doesn't stay frozen tracking the last pose.
        smpl = _extract_smpl_from_action(action)
        root_quat = _extract_root_from_action(action)
        if smpl is None and self._smpl_stream is not None:
            window = self._smpl_stream.step()
            if self._smpl_stream.has_data and not self._smpl_stream.is_stale:
                smpl = window
                root_quat = np.asarray(self._smpl_stream.root_quat, np.float32)

        if smpl is not None:
            # Full-body whole-body tracking: SMPL drives the reference, not joystick.
            if self.controller.encode_mode != 2:
                self._enter_wholebody()
            self.controller.smpl_joints_10frame_step1 = smpl
            # Root orientation (if provided) steers the mode-2 anchor/heading.
            self.controller.smpl_root_quat = root_quat
            return self._runtime.tick(obs, debug=False, use_joystick=False)

        # No (or stale) SMPL: fall back to locomotion so the robot stays balanced.
        if self.controller.encode_mode == 2:
            self.controller.smpl_root_quat = None
            self._exit_wholebody()
        return self._runtime.tick(obs, debug=False)

    def reset(self):
        self._runtime.reset()

    def shutdown(self):
        if self._smpl_stream is not None:
            self._smpl_stream.close()
        self._runtime.shutdown()
