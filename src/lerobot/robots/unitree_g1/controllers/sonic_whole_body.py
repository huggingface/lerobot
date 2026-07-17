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
import math
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

from lerobot.teleoperators.pico_headset.smpl_constants import (
    LOCO_AXES_PREFIX,
    LOCO_BTN_PREFIX,
    LOCO_N_AXES,
    LOCO_N_BTN,
    ROOT_ACTION_DIM,
    ROOT_ACTION_PREFIX,
    SMPL_ACTION_PREFIX,
    SMPL_OBS_DIM as SMPL_ACTION_DIM,
    VR3_ORN_DIM,
    VR3_ORN_PREFIX,
    VR3_POS_DIM,
    VR3_POS_PREFIX,
)
from lerobot.utils.import_utils import _onnxruntime_available, require_package

from ..g1_utils import KEYBOARD_KEYS_FIELD, G1_29_JointIndex, lowstate_to_obs
from .sonic_pipeline import (
    CONTROL_DT,
    DEBUG_PRINT_EVERY,
    DEFAULT_ANGLES,
    DEFAULT_HEIGHT,
    ENCODER_UPDATE_EVERY,
    LM,
    MOTION_SETS,
    MovementState,
    PlannerController,
    SonicPlanner,
    apply_pico_loco_axes,
    clamp_mode_params,
    compute_kp_kd,
    make_ort_session_options,
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

# Startup blend duration: over the first control ticks, linearly interpolate every joint
# from the robot's initial measured pose into the policy's commanded target, so control
# eases in without a snap on the first command.
INIT_RAMP_S = 3.0


def _extract_smpl_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a (720,) SMPL window from ``smpl.{i}`` action keys, or None.

    The pico_headset teleoperator emits the whole-body reference as flat floats so
    it flows unchanged through the standard lerobot action pipeline.
    """
    # The keys are smpl.0 .. smpl.719; presence of the first element (smpl.0) is the
    # sentinel that a full SMPL window was sent this tick. If it's absent, there's no
    # whole-body reference, so bail out.
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


def _extract_vr3_from_action(action: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Reassemble the 3-point VR targets from ``vr3_pos.{i}`` / ``vr3_orn.{i}`` keys.

    Returns ``(pos (9,), orn (12,))`` for the [l-wrist, r-wrist, neck] keypoints, or
    None when no VR3 reference was sent this tick. Presence of ``vr3_pos.0`` is the
    sentinel that a full 3-point frame is available (mirrors the SMPL sentinel).
    """
    if not action or f"{VR3_POS_PREFIX}0" not in action:
        return None
    pos = np.fromiter(
        (float(action.get(f"{VR3_POS_PREFIX}{i}", 0.0)) for i in range(VR3_POS_DIM)),
        dtype=np.float32,
        count=VR3_POS_DIM,
    )
    orn = np.fromiter(
        (float(action.get(f"{VR3_ORN_PREFIX}{i}", 0.0)) for i in range(VR3_ORN_DIM)),
        dtype=np.float32,
        count=VR3_ORN_DIM,
    )
    return pos, orn


def _extract_loco_from_action(action: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Reassemble controller-stick locomotion from ``loco_axes.{i}`` / ``loco_btn.{i}``.

    Returns ``(axes (4,) = [lx, ly, rx, ry], buttons (4,) = [A, B, X, Y])`` or None
    when no locomotion state was sent this tick (sentinel: ``loco_axes.0``).
    """
    if not action or f"{LOCO_AXES_PREFIX}0" not in action:
        return None
    axes = np.fromiter(
        (float(action.get(f"{LOCO_AXES_PREFIX}{i}", 0.0)) for i in range(LOCO_N_AXES)),
        dtype=np.float32,
        count=LOCO_N_AXES,
    )
    buttons = np.fromiter(
        (float(action.get(f"{LOCO_BTN_PREFIX}{i}", 0.0)) for i in range(LOCO_N_BTN)),
        dtype=np.float32,
        count=LOCO_N_BTN,
    )
    return axes, buttons


class SonicRuntime:
    """Shared SONIC control loop state (standalone demo + locomotion controller)."""

    def __init__(self, force_cpu: bool = False):
        require_package("onnxruntime", extra="unitree_g1")
        planner_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="planner_sonic.onnx")
        encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        providers = ort_providers(force_cpu=force_cpu)
        self.use_gpu = providers[0] == "CUDAExecutionProvider"
        so = make_ort_session_options()

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

    def __init__(
        self,
        force_cpu: bool = False,
        *,
        enable_smpl_root: bool = False,
        root_smoothing_alpha: float = 0.15,
        enable_smpl_stream: bool = False,
        smpl_host: str | None = None,
        smpl_port: int | None = None,
    ):
        logger.info("Loading SONIC whole-body controller...")
        self._runtime = SonicRuntime(force_cpu=force_cpu)
        self.kp = self._runtime.kp
        self.kd = self._runtime.kd
        self.controller = self._runtime.controller
        self.ms = self._runtime.ms

        # When True, the per-frame SMPL root quaternion steers the mode-2 anchor.
        # Off by default: even with smoothing this changes the anchor/heading and is
        # untested on hardware, so it stays opt-in. When enabled, the raw per-frame
        # root quat (from a 30 Hz dataset resampled to a 50 Hz loop) is spherically
        # smoothed by :meth:`_smooth_root_quat` before it reaches the anchor, which
        # removes the root-acceleration spikes (NaN QACC at DOF 0) the unsmoothed
        # trajectory caused. ``root_smoothing_alpha`` in (0, 1] is the per-tick blend
        # toward the incoming quat (smaller = smoother/laggier, 1 = no smoothing).
        self.enable_smpl_root = enable_smpl_root
        self._root_smoothing_alpha = float(np.clip(root_smoothing_alpha, 1e-3, 1.0))
        self._smoothed_root_quat: np.ndarray | None = None

        # Tracks the previous keyboard held-key set so discrete controls (mode,
        # motion set, replan, e-stop, WASD direction) fire once per physical press
        # instead of every 50 Hz tick while the key is held.
        self._prev_keys: set[str] = set()
        # Edge state for the PICO A+B / X+Y locomotion-mode cycle (3-point teleop).
        self._prev_loco_mode_pair: tuple[bool, bool] = (False, False)

        # Startup blend: ease from the robot's initial pose into the first commanded
        # policy targets over INIT_RAMP_S (captured on the first control tick).
        self._init_ramp_steps = max(1, round(INIT_RAMP_S / CONTROL_DT))
        self._init_step = 0
        self._start_pose: dict[str, float] = {}

        # Optional: subscribe directly to the rt/smpl headset stream so full-body
        # teleop works with ANY teleoperator (e.g. --teleop.type=unitree_g1 for the
        # estop/joystick) before the dedicated pico_headset teleop exists.
        self._smpl_host = smpl_host
        self._smpl_port = smpl_port
        self._smpl_stream = None
        if enable_smpl_stream:
            self._init_smpl_stream()

        logger.info(
            "SONIC ready: %s (default mode: %s, smpl_stream=%s)",
            MOTION_SETS[0][0],
            LM(self.ms.mode).name,
            self._smpl_stream is not None,
        )

    def _init_smpl_stream(self) -> None:
        # Lazy import so the zmq dependency is only required when streaming is on.
        from lerobot.teleoperators.pico_headset.smpl_stream import (
            DEFAULT_SMPL_HOST,
            DEFAULT_SMPL_PORT,
            SmplStream,
        )

        host = self._smpl_host or DEFAULT_SMPL_HOST
        port = self._smpl_port or DEFAULT_SMPL_PORT
        self._smpl_stream = SmplStream(host=host, port=port)
        logger.info("SONIC subscribed to rt/smpl @ tcp://%s:%d", host, port)

    def _enter_wholebody(self) -> None:
        """Switch into SMPL whole-body tracking (encode_mode 2)."""
        self.controller.encode_mode = 2
        self.controller.reinit_heading = True
        logger.info("SONIC: SMPL stream active -> whole-body tracking (mode 2)")

    def _enter_3point(self) -> None:
        """Switch into 3-point VR upper-body teleop (encode_mode 1).

        The upper body tracks the VR wrist/neck targets while the lower body /
        locomotion keeps running off the planner (joystick/keyboard-driven).
        """
        self.controller.encode_mode = 1
        self.controller.playing = True
        self.controller.reinit_heading = True
        self.ms.needs_replan = True
        logger.info("SONIC: 3-point VR active -> upper-body tracking + planner locomotion (mode 1)")

    def _exit_wholebody(self) -> None:
        """Revert to locomotion/standing (encode_mode 0) after a teleop reference is lost.

        Mirrors the 'M' toggle in sonic.py so the handoff is clean: the robot holds
        a standing reference and (if a joystick teleop is attached) can be driven.
        """
        self.controller.encode_mode = 0
        self.controller.playing = True
        self.controller.reinit_heading = True
        self.ms.needs_replan = True
        logger.warning("SONIC: teleop reference lost/stale -> reverting to locomotion (standing)")

    def _process_keyboard(self, action: dict | None) -> None:
        """Translate a native KeyboardTeleop's held-key set into MovementState.

        Mirrors the standalone SONIC demo's keyboard mapping so locomotion (mode 0/1)
        can be driven with ``--teleop.type=keyboard`` instead of the PICO SMPL stream.
        Discrete controls act on newly-pressed keys (edge-detected against the previous
        tick); inherently-continuous controls (facing turn, height, speed) integrate a
        small per-tick delta while the key is held so they feel smooth at 50 Hz.

        Controls: WASD move, Q/E turn, 1-8 select mode, 9/0 speed down/up,
        -/= height down/up, R replan, Space emergency-stop -> IDLE.
        """
        if action is None:
            return
        keys = action.get(KEYBOARD_KEYS_FIELD)
        if keys is None:
            return  # No KeyboardTeleop attached; leave joystick/SMPL paths untouched.

        ms, controller = self.ms, self.controller
        held = {k.lower() if isinstance(k, str) and len(k) == 1 else k for k in keys}
        prev = self._prev_keys
        pressed = held - prev  # newly-pressed this tick (edge)
        self._prev_keys = held

        # ── Discrete: fire once per press ────────────────────────────────────
        if "space" in pressed:
            ms.mode = LM.IDLE
            ms.speed = ms.height = -1.0
            ms.has_movement = False
            ms.needs_replan = True
            controller.playing = False
            controller.reinit_heading = True
            logger.info("SONIC keyboard: EMERGENCY STOP -> IDLE")
        if "r" in pressed:
            ms.needs_replan = True
        if "n" in pressed or "p" in pressed:
            step = 1 if "n" in pressed else -1
            ms.motion_set_idx = (ms.motion_set_idx + step) % len(MOTION_SETS)
            logger.info("SONIC keyboard: motion set -> %s", MOTION_SETS[ms.motion_set_idx][0])
        for digit in ("1", "2", "3", "4", "5", "6", "7", "8"):
            if digit in pressed:
                idx = int(digit) - 1
                modes = MOTION_SETS[ms.motion_set_idx][1]
                if 0 <= idx < len(modes):
                    ms.mode = modes[idx]
                    ms.has_movement = False
                    ms.needs_replan = True
                    controller.playing = True
                    controller.reinit_heading = True
                    logger.info("SONIC keyboard: mode -> %s", LM(ms.mode).name)
        # WASD sets the movement direction relative to current facing (press to set,
        # Space to stop) to match the standalone demo.
        if "w" in pressed:
            ms.movement_angle = ms.facing_angle
        elif "s" in pressed:
            ms.movement_angle = ms.facing_angle + math.pi
        elif "a" in pressed:
            ms.movement_angle = ms.facing_angle + math.pi / 2
        elif "d" in pressed:
            ms.movement_angle = ms.facing_angle - math.pi / 2
        if pressed & {"w", "a", "s", "d"}:
            ms.has_movement = True
            ms.needs_replan = True

        # ── Continuous: integrate a small delta while held ───────────────────
        if "q" in held:
            ms.facing_angle += 0.02
            controller.delta_heading += 0.02
        if "e" in held:
            ms.facing_angle -= 0.02
            controller.delta_heading -= 0.02
        if "0" in held:
            ms.speed = min(5.0, (ms.speed if ms.speed >= 0 else 1.0) + 0.02)
        if "9" in held:
            ms.speed = max(0.0, (ms.speed if ms.speed >= 0 else 1.0) - 0.02)
        if "=" in held:
            ms.height = min(1.0, (ms.height if ms.height >= 0 else DEFAULT_HEIGHT) + 0.005)
        if "-" in held:
            ms.height = max(0.1, (ms.height if ms.height >= 0 else DEFAULT_HEIGHT) - 0.005)

    def _process_pico_loco(self, axes: np.ndarray, buttons: np.ndarray) -> None:
        """Drive locomotion from the PICO controller sticks/buttons (encode_mode 1).

        Mirrors gear_sonic's ``PlannerLoop`` VR-3PT tick: left/right sticks steer
        movement/facing/speed via :func:`apply_pico_loco_axes` (the faithful gear_sonic
        yaw-accumulator + mode-dependent speed curves, not the keyboard-parity map), and
        A+B / X+Y edge-cycle the locomotion mode within the current motion set.
        """
        lx, ly, rx, ry = (float(v) for v in axes)
        apply_pico_loco_axes(lx, ly, rx, ry, self.ms)

        # Mode cycling: step linearly through the LocomotionMode enum (A+B = next,
        # X+Y = previous), exactly like gear_sonic's PlannerLoop — so the operator can
        # reach squat/kneel/crawl, not just the modes in one UI motion set.
        a, b, x, y = (v > 0.5 for v in buttons)
        ab_now, xy_now = (a and b), (x and y)
        ab_prev, xy_prev = self._prev_loco_mode_pair
        mode = int(self.ms.mode)
        if ab_now and not ab_prev:
            mode = min(int(LM.INJURED_WALK), mode + 1)
        elif xy_now and not xy_prev:
            mode = max(int(LM.IDLE), mode - 1)
        if mode != int(self.ms.mode):
            self.ms.mode = LM(mode)
            self.ms.needs_replan = True
            self.controller.playing = True
            logger.info("SONIC 3-point: locomotion mode -> %s", LM(self.ms.mode).name)
        self._prev_loco_mode_pair = (ab_now, xy_now)

    def _smooth_root_quat(self, root_quat: np.ndarray | None) -> np.ndarray | None:
        """Spherically smooth the per-frame SMPL root quaternion (mode-2 anchor).

        The reference root trajectory is authored at ~30 Hz and consumed at 50 Hz, so
        the raw per-tick quat steps unevenly and injects root-acceleration spikes into
        the anchor. This keeps a persistent estimate and shortest-path nlerp-slerps it
        toward each incoming (unit) quat by ``root_smoothing_alpha``, yielding a
        continuous, rate-matched heading. Quaternions are scalar-first (w, x, y, z).
        Returns ``None`` (leaving the anchor self-driven) for an invalid/zero input.
        """
        if root_quat is None:
            self._smoothed_root_quat = None
            return None
        q = np.asarray(root_quat, np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return self._smoothed_root_quat
        q = q / n
        if self._smoothed_root_quat is None:
            self._smoothed_root_quat = q
        else:
            prev = self._smoothed_root_quat
            if np.dot(prev, q) < 0.0:  # shortest-path: quats double-cover SO(3)
                q = -q
            blended = prev + self._root_smoothing_alpha * (q - prev)
            self._smoothed_root_quat = blended / (np.linalg.norm(blended) + 1e-12)
        return self._smoothed_root_quat.astype(np.float32)

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

        # Keyboard teleop (native KeyboardTeleop) drives the same locomotion intent
        # the joystick does; applied before the SMPL check so whole-body tracking
        # still takes priority when a headset stream is present.
        self._process_keyboard(action)

        # Prefer SMPL delivered via the teleop action (pico_headset). Fall back to a
        # direct rt/smpl subscription when enabled (enable_smpl_stream). A stale
        # stream (headset silent past its timeout) is treated as "no SMPL" so the
        # robot doesn't stay frozen tracking the last pose.
        smpl = _extract_smpl_from_action(action)
        root_quat = _extract_root_from_action(action)
        vr3 = _extract_vr3_from_action(action)
        loco = _extract_loco_from_action(action)
        if smpl is None and vr3 is None and self._smpl_stream is not None:
            window = self._smpl_stream.step()
            if self._smpl_stream.has_data and not self._smpl_stream.is_stale:
                smpl = window
                root_quat = np.asarray(self._smpl_stream.root_quat, np.float32)
            # VR3 is independent of the SMPL window: the controller-state source
            # (head + controllers only) sends 3-point targets with no SMPL frame.
            elif self._smpl_stream.has_fresh_vr3:
                vr3 = (self._smpl_stream.vr3_pos, self._smpl_stream.vr3_orn)
                if self._smpl_stream.has_fresh_loco:
                    loco = (self._smpl_stream.loco_axes, self._smpl_stream.loco_buttons)

        if smpl is not None:
            # Full-body whole-body tracking: SMPL drives the reference, not joystick.
            if self.controller.encode_mode != 2:
                self._enter_wholebody()
            self.controller.smpl_joints_10frame_step1 = smpl
            # Root orientation steers the mode-2 anchor/heading, but only when
            # explicitly enabled (see enable_smpl_root); the raw per-frame quat is
            # spherically smoothed first so the 30->50 Hz resample doesn't spike the
            # anchor. Disabled -> anchor stays self-driven.
            self.controller.smpl_root_quat = (
                self._smooth_root_quat(root_quat) if self.enable_smpl_root else None
            )
            out = self._runtime.tick(obs, debug=False, use_joystick=False)
        elif vr3 is not None:
            # 3-point VR teleop: upper body tracks the wrist/neck targets; the lower
            # body / locomotion keeps running off the planner, so the joystick (and
            # keyboard) still steer walking/turning underneath.
            if self.controller.encode_mode != 1:
                self._enter_3point()
            self.controller.vr_3point_local_target = vr3[0]
            self.controller.vr_3point_local_orn_target = vr3[1]
            # Replicate the original encode_mode-1 handling: when the PICO controller
            # sticks are forwarded, drive locomotion from them directly (and skip the
            # wireless-remote joystick read). Otherwise leave the remote/keyboard path.
            if loco is not None:
                self._process_pico_loco(loco[0], loco[1])
                out = self._runtime.tick(obs, debug=False, use_joystick=False)
            else:
                out = self._runtime.tick(obs, debug=False, use_joystick=True)
        else:
            # No (or stale) teleop reference: fall back to locomotion so the robot stays balanced.
            if self.controller.encode_mode != 0:
                self.controller.smpl_root_quat = None
                self._smoothed_root_quat = None
                self._exit_wholebody()
            out = self._runtime.tick(obs, debug=False)

        # Startup interpolation: blend from the robot's initial pose into the policy's
        # commanded target over INIT_RAMP_S, regardless of mode.
        return self._startup_blend(obs, out)

    def reset(self):
        self._runtime.reset()
        self._init_step = 0  # re-run the startup blend after a reset
        self._start_pose = {}
        self._smoothed_root_quat = None

    def shutdown(self):
        if self._smpl_stream is not None:
            self._smpl_stream.close()
        self._runtime.shutdown()
