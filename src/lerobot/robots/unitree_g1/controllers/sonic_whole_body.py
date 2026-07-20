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
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

from lerobot.utils.import_utils import _onnxruntime_available, require_package

from ..g1_utils import (
    MUJOCO_TO_ISAACLAB,
    WB_ACTION_DIM,
    G1_29_JointIndex,
    lowstate_to_obs,
    wb_action_key,
)
from .sonic_pipeline import (
    CONTROL_DT,
    DEFAULT_ANGLES,
    ENCODER_UPDATE_EVERY,
    PlannerController,
    compute_kp_kd,
    make_ort_session_options,
    ort_providers,
)

if TYPE_CHECKING or _onnxruntime_available:
    import onnxruntime as ort
else:
    ort = None

logger = logging.getLogger(__name__)

# Startup blend duration: over the first control ticks, linearly interpolate every joint
# from the robot's initial measured pose into the policy's commanded target, so control
# eases in without a snap on the first command.
INIT_RAMP_S = 3.0


def _extract_wb34_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a dense (34,) whole-body command from ``wb.{i}.pos`` keys, or None.

    This is the OpenHLM / pi0.5 joint-based interface: one 34-D vector per tick
    (sentinel: presence of ``wb.0.pos``) carrying absolute joint targets in real
    units. The ``.pos`` suffix lets these flow through ``lerobot-rollout`` as normal
    joint-position action features.
    """
    if not action or wb_action_key(0) not in action:
        return None
    return np.fromiter(
        (float(action.get(wb_action_key(i), 0.0)) for i in range(WB_ACTION_DIM)),
        dtype=np.float32,
        count=WB_ACTION_DIM,
    )


def _wb34_to_reference(wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 34-D OpenHLM whole-body command to a SONIC mode-0 reference.

    Returns ``(ref29, anchor_quat)`` where ``ref29`` is the 29 joint targets in
    IsaacLab order (what SONIC's ``motion_joint_positions`` expects) and
    ``anchor_quat`` (wxyz) encodes the root roll/pitch (yaw=0).

    OpenHLM layout : [L-arm 0:7, L-grip 7, R-arm 8:15, R-grip 15,
                      L-leg 16:22, R-leg 22:28, waist 28:31, root rp+yaw 31:34]
    The 29 joints are first assembled in MuJoCo / Unitree-SDK order
    ([L-leg 0:6, R-leg 6:12, waist 12:15, L-arm 15:22, R-arm 22:29] — the
    ``G1_29_JointIndex`` grouping OpenHLM uses), then permuted to IsaacLab order via
    ``MUJOCO_TO_ISAACLAB``. Grippers (7, 15) and yaw-rate (33) are not part of the
    29-DoF SONIC reference.
    """
    ref_mj = np.zeros(29, np.float32)  # MuJoCo / Unitree-SDK grouped order
    ref_mj[0:6] = wb[16:22]   # left leg
    ref_mj[6:12] = wb[22:28]  # right leg
    ref_mj[12:15] = wb[28:31]  # waist
    ref_mj[15:22] = wb[0:7]   # left arm
    ref_mj[22:29] = wb[8:15]  # right arm
    ref = ref_mj[MUJOCO_TO_ISAACLAB].astype(np.float32)  # -> IsaacLab order for SONIC
    roll, pitch = float(wb[31]), float(wb[32])
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    anchor = np.array([cr * cp, sr * cp, cr * sp, sr * sp], np.float32)  # Rx(roll)·Ry(pitch)
    return ref, anchor


class SonicRuntime:
    """Loads the SONIC encoder/decoder ONNX models and owns the controller.

    No motion planner: the reference motion buffer is written directly each tick by
    :class:`SonicWholeBodyController` from the incoming 34-D whole-body command.
    """

    def __init__(self, force_cpu: bool = False):
        require_package("onnxruntime", extra="unitree_g1")
        encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        providers = ort_providers(force_cpu=force_cpu)
        self.use_gpu = providers[0] == "CUDAExecutionProvider"
        so = make_ort_session_options()

        encoder_sess = ort.InferenceSession(encoder_path, sess_options=so, providers=providers)
        decoder_sess = ort.InferenceSession(decoder_path, sess_options=so, providers=providers)

        self.kp, self.kd = compute_kp_kd()
        self.controller = PlannerController(encoder_sess, decoder_sess)

    @property
    def pipeline(self):
        return self.controller

    def reset(self):
        self.controller.reinit_heading = True
        self.controller.playing = True

    def shutdown(self):
        pass


class SonicWholeBodyController:
    """Full-body SONIC controller for UnitreeG1's background controller thread."""

    control_dt = CONTROL_DT
    full_body = True
    # Advertise a dense 34-D whole-body action space (OpenHLM / pi0.5) so the robot
    # exposes ``wb.{i}.pos`` action features and ``lerobot-rollout`` can drive it
    # directly with a 34-D VLA policy.
    wb_action = True

    def __init__(self, force_cpu: bool = False):
        logger.info("Loading SONIC whole-body controller...")
        self._runtime = SonicRuntime(force_cpu=force_cpu)
        self.kp = self._runtime.kp
        self.kd = self._runtime.kd
        self.controller = self._runtime.controller

        # Startup blend: ease from the robot's initial pose into the first commanded
        # policy targets over INIT_RAMP_S (captured on the first control tick).
        self._init_ramp_steps = max(1, round(INIT_RAMP_S / CONTROL_DT))
        self._init_step = 0
        self._start_pose: dict[str, float] = {}

        # Tick counter for the dense whole-body (OpenHLM, mode-0) path's encoder cadence.
        self._wb_step = 0
        # Rolling 50-frame reference trajectory (ref29 + anchor quat) built from the
        # stream of per-tick whole-body commands, fed to the encoder as a batch.
        self._wb_traj: deque[np.ndarray] = deque(maxlen=50)
        self._wb_quat_traj: deque[np.ndarray] = deque(maxlen=50)

        logger.info("SONIC ready (encoder/decoder, 34-D whole-body command path)")

    def _run_wholebody34(self, obs: dict, wb: np.ndarray) -> dict:
        """Feed a dense 34-D OpenHLM whole-body command as the mode-0 encoder reference.

        The 29 joint targets are held across the encoder lookahead window (zero
        velocity) and the root roll/pitch set the anchor orientation, then the
        encoder/decoder run directly (planner bypassed). One command per tick, so the
        VLA's commanded pose is what SONIC tracks.
        """
        ref, anchor = _wb34_to_reference(wb)
        c = self.controller
        if c.encode_mode != 0:
            c.encode_mode = 0
            c.reinit_heading = True
        # Capture the heading/anchor reference on the first whole-body tick. The
        # controller only latches ``init_ref_quat`` (and the base heading) inside
        # ``step()`` when ``first_motion or reinit_heading`` — but it already boots in
        # mode 0, so the mode-switch guard above misses the very first command and the
        # anchor would stay identity. This mirrors the GEAR reference, which seeds
        # ``init_ref_quat`` from the first anchor. Must run before the buffers below so
        # ``step()`` latches ``motion_body_quats[0]`` = this tick's anchor.
        if self._wb_step == 0:
            c.reinit_heading = True

        # Accumulate the per-tick commands into a rolling 50-frame reference
        # trajectory so the encoder's 10-frame, step-5 lookahead sees an actual
        # motion sequence (with velocities) instead of one repeated pose. 50 frames
        # == chunk horizon == 10 lookahead frames × step 5.
        self._wb_traj.append(ref)
        self._wb_quat_traj.append(anchor)
        traj = np.asarray(self._wb_traj, np.float32)  # (L, 29), oldest -> newest
        quats = np.asarray(self._wb_quat_traj, np.float32)  # (L, 4)
        n = len(traj)
        # Per-frame velocities from finite differences (rad/s at the control rate).
        vel = np.zeros_like(traj)
        if n > 1:
            vel[1:] = (traj[1:] - traj[:-1]) / CONTROL_DT
            vel[0] = vel[1]
        with c.motion_lock:
            c.motion_joint_positions[:n] = traj
            c.motion_joint_velocities[:n] = vel
            c.motion_body_quats[:n] = quats
            c.motion_body_pos[:n] = 0.0
            c.motion_timesteps = n
            c.ref_cursor = 0
        c.playing = True
        do_enc = self._wb_step % ENCODER_UPDATE_EVERY == 0
        out = c.step(obs, update_encoder=do_enc, debug=False)
        if self._wb_step % 25 == 0:
            tgt = np.array([out[f"{m.name}.q"] for m in G1_29_JointIndex], np.float32)
            logger.info(
                "[WB34] step=%d |ref|mean=%.3f |target|mean=%.3f target_std=%.3f init_ref_quat=%s",
                self._wb_step,
                float(np.abs(ref).mean()),
                float(np.abs(tgt).mean()),
                float(tgt.std()),
                np.round(c.init_ref_quat, 3).tolist(),
            )
        self._wb_step += 1
        return out

    def _startup_blend(self, obs: dict, out: dict) -> dict:
        """Ease into policy control at startup: for the first ``INIT_RAMP_S`` seconds,
        interpolate between the robot's pose captured on the first tick and the policy's
        live commanded target, so the handoff has no snap.

        ``out`` is the policy's ``<joint>.q`` target dict for this tick; the blend ratio
        climbs 0->1 over the ramp, after which the raw policy target passes through.
        """
        if self._init_step >= self._init_ramp_steps or not out:
            return out
        if self._init_step == 0:
            # Capture the robot's actual pose as the interpolation start point.
            self._start_pose = {
                f"{m.name}.q": float(obs.get(f"{m.name}.q", DEFAULT_ANGLES[m.value]))
                for m in G1_29_JointIndex
            }
        self._init_step += 1
        ratio = min(1.0, self._init_step / self._init_ramp_steps)
        blended = {
            k: self._start_pose.get(k, float(tgt)) * (1.0 - ratio) + float(tgt) * ratio
            for k, tgt in out.items()
        }
        if self._init_step >= self._init_ramp_steps:
            logger.info("SONIC startup blend complete -> full policy control")
        return blended

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}
        obs = lowstate_to_obs(lowstate)

        # Dense 34-D whole-body command (OpenHLM / pi0.5 joint interface): a single
        # vector per tick drives the mode-0 encoder reference directly. Until the
        # policy produces one, hold (no command) so the robot keeps its last target.
        wb = _extract_wb34_from_action(action)
        if wb is None:
            self._wb_miss = getattr(self, "_wb_miss", 0) + 1
            if self._wb_miss % 50 == 1:
                akeys = [k for k in action if isinstance(k, str)]
                logger.info(
                    "[WB34] no wb.*.pos in action this tick (miss=%d). action keys sample: %s",
                    self._wb_miss,
                    akeys[:8],
                )
            return {}
        return self._startup_blend(obs, self._run_wholebody34(obs, wb))

    def reset(self):
        self._runtime.reset()
        self._init_step = 0  # re-run the startup blend after a reset
        self._start_pose = {}
        self._wb_step = 0
        self._wb_traj.clear()
        self._wb_quat_traj.clear()

    def shutdown(self):
        self._runtime.shutdown()
