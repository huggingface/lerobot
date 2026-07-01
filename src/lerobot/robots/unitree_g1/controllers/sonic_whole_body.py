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

import logging

import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
    CONTROL_DT,
    DEBUG_PRINT_EVERY,
    DEFAULT_ANGLES,
    ENCODER_UPDATE_EVERY,
    LM,
    MOTION_SETS,
    MovementState,
    PlannerController,
    SonicPlanner,
    _ort_providers,
    _snapshot_ms,
    clamp_mode_params,
    compute_kp_kd,
    lowstate_to_obs,
    process_joystick,
    should_replan_request,
)

logger = logging.getLogger(__name__)


class SonicRuntime:
    """Shared SONIC control loop state (standalone demo + locomotion controller)."""

    def __init__(self, force_cpu: bool = False):
        planner_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="planner_sonic.onnx")
        encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        providers = _ort_providers(force_cpu=force_cpu)
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
        self.last_ms = _snapshot_ms(self.ms)

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
            self.last_ms = _snapshot_ms(self.ms)

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
        self.last_ms = _snapshot_ms(self.ms)

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
        logger.info(
            "SONIC ready: %s (default mode: %s)",
            MOTION_SETS[0][0],
            LM(self.ms.mode).name,
        )

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}
        obs = lowstate_to_obs(lowstate)
        return self._runtime.tick(obs, debug=False)

    def reset(self):
        self._runtime.reset()

    def shutdown(self):
        self._runtime.shutdown()
