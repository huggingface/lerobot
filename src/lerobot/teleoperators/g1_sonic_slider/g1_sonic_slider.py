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

"""Pygame SONIC test UI: foot xyz (leg IK) + waist/arm joint sliders."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from functools import cached_property

import numpy as np

from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from .config_g1_sonic_slider import G1SonicSliderTeleopConfig
from .joint_limits import JOINT_HI, JOINT_LO, JOINT_NAMES

logger = logging.getLogger(__name__)

NUM_JOINTS = 29
LEG_JOINT_COUNT = 12
UPPER_BODY_INDICES = list(range(LEG_JOINT_COUNT, NUM_JOINTS))
NUM_FOOT_SLIDERS = 6
FOOT_LABELS = ("L foot X", "L foot Y", "L foot Z", "R foot X", "R foot Y", "R foot Z")

# Pelvis-frame standing foot centers (m) if Pinocchio FK is unavailable at startup.
_FALLBACK_LEFT_FOOT = np.array([0.02, 0.12, -0.76], dtype=np.float32)
_FALLBACK_RIGHT_FOOT = np.array([0.02, -0.12, -0.76], dtype=np.float32)

HEADER_H = 56
LABEL_W = 148
MARGIN = 10
KNOB_W = 10


def _leg_ik_process(target_q: mp.Queue, result_q: mp.Queue, stop_evt) -> None:
    """Child process: build the leg IK once, then solve for the latest foot target.

    The IPOPT/CasADi solve holds the GIL, so it must run in a separate *process*
    (not a thread) to keep the teleop UI loop responsive.
    """
    import numpy as _np

    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES as _DEFAULTS
    from lerobot.robots.unitree_g1.g1_kinematics import G1_29_LegIK

    try:
        ik = G1_29_LegIK()
        q_legs = _DEFAULTS[:LEG_JOINT_COUNT].astype(_np.float64)
        ik.cache_default_orientation(q_legs)
        left_pos, right_pos = ik.foot_positions(q_legs)
    except Exception as e:  # noqa: BLE001
        result_q.put(("error", str(e)))
        return

    current = q_legs.copy()
    result_q.put(("ready", _np.concatenate([left_pos, right_pos]).astype(_np.float64)))

    while not stop_evt.is_set():
        try:
            target = target_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if target is None:
            break
        # Drain to the most recent target so we never solve stale slider positions.
        while True:
            try:
                newer = target_q.get_nowait()
            except queue.Empty:
                break
            if newer is None:
                target = None
                break
            target = newer
        if target is None:
            break
        target = _np.asarray(target, dtype=_np.float64)
        # Fast damped-least-squares IK: sub-ms per step, warm-started from the last
        # solution so the legs track the sliders in real time.
        leg_q = ik.solve_ik_dls(target[:3], target[3:], current_leg_q_g1=current)
        current = _np.asarray(leg_q, dtype=_np.float64)
        result_q.put(("q", current.copy()))


class G1SonicSliderTeleop(Teleoperator):
    """Foot xyz + waist/arm sliders feeding SONIC encoder mode 0."""

    config_class = G1SonicSliderTeleopConfig
    name = "g1_sonic_slider"

    def __init__(self, config: G1SonicSliderTeleopConfig):
        super().__init__(config)
        self.config = config
        self._values = DEFAULT_ANGLES.astype(np.float32).copy()
        self._foot_xyz = np.zeros(6, dtype=np.float32)
        self._foot_lo = np.full(6, -0.5, dtype=np.float32)
        self._foot_hi = np.full(6, 0.5, dtype=np.float32)
        self._scroll_y = 0
        self._drag_joint: int | None = None
        self._drag_foot: int | None = None
        self._connected = False
        self._pygame = None
        self._screen = None
        self._font = None
        self._small_font = None
        self._clock = None
        # Leg IK runs in a separate process: the IPOPT/CasADi solve holds the GIL,
        # so a thread would still stall the UI loop. We publish the latest foot
        # target and read back the newest solution over queues, never blocking.
        self._ik_proc: mp.process.BaseProcess | None = None
        self._ik_target_q: mp.Queue | None = None
        self._ik_result_q: mp.Queue | None = None
        self._ik_stop_evt = None
        self._ik_ready = False
        self._standing_foot_xyz: np.ndarray | None = None
        self._leg_ik_ok = False
        self._leg_ik_error: str | None = None
        self._foot_divider_x = config.foot_panel_width

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.q": float for name in JOINT_NAMES}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def _set_foot_limits_from_positions(self, left_pos: np.ndarray, right_pos: np.ndarray) -> None:
        """Slider range = FK foot position at standing pose ± foot_xyz_margin (meters, pelvis frame)."""
        margin = np.array(self.config.foot_xyz_margin, dtype=np.float32)
        for i, pos in enumerate((left_pos, right_pos)):
            base = i * 3
            self._foot_xyz[base : base + 3] = pos.astype(np.float32)
            self._foot_lo[base : base + 3] = pos.astype(np.float32) - margin
            self._foot_hi[base : base + 3] = pos.astype(np.float32) + margin

    def _set_fallback_foot_limits(self) -> None:
        self._set_foot_limits_from_positions(_FALLBACK_LEFT_FOOT, _FALLBACK_RIGHT_FOOT)

    def _init_leg_ik(self) -> None:
        if not self.config.use_leg_ik:
            return
        # Fallback limits until the solver process reports standing FK foot positions.
        self._set_fallback_foot_limits()
        try:
            # Metadata name for the Pinocchio distribution is "pin" (conda-forge/PyPI),
            # not "pinocchio", which is only the importable module name.
            require_package("pin", extra="unitree_g1", import_name="pinocchio")
            require_package("casadi", extra="unitree_g1", import_name="casadi")
        except Exception as e:
            self._leg_ik_ok = False
            self._leg_ik_error = str(e)
            logger.warning("Leg IK unavailable (%s); foot sliders shown but legs use joint values", e)
            return

        # Use "spawn": the parent already holds CUDA/pygame/threads and forking that
        # state into the solver is unsafe.
        ctx = mp.get_context("spawn")
        self._ik_target_q = ctx.Queue(maxsize=2)
        self._ik_result_q = ctx.Queue()
        self._ik_stop_evt = ctx.Event()
        self._ik_proc = ctx.Process(
            target=_leg_ik_process,
            args=(self._ik_target_q, self._ik_result_q, self._ik_stop_evt),
            name="g1-leg-ik",
            daemon=True,
        )
        self._ik_proc.start()
        self._leg_ik_ok = True
        self._leg_ik_error = None
        logger.info("Leg IK solver process starting (foot limits update once standing FK is ready)...")

    def _publish_ik_target(self) -> None:
        if self._ik_target_q is None:
            return
        # Keep only the newest target; drop if the child is momentarily behind.
        try:
            self._ik_target_q.put_nowait(self._foot_xyz.copy())
        except queue.Full:
            pass

    def _pump_ik_results(self) -> None:
        if self._ik_result_q is None:
            return
        while True:
            try:
                kind, payload = self._ik_result_q.get_nowait()
            except queue.Empty:
                break
            if kind == "q":
                self._values[:LEG_JOINT_COUNT] = np.asarray(payload, dtype=np.float32)
            elif kind == "ready":
                payload = np.asarray(payload, dtype=np.float32)
                self._standing_foot_xyz = payload.copy()
                self._set_foot_limits_from_positions(payload[:3], payload[3:])
                self._ik_ready = True
                logger.info(
                    "Leg IK ready — foot slider limits from standing FK ± %s m (pelvis frame)",
                    self.config.foot_xyz_margin,
                )
            elif kind == "error":
                self._leg_ik_ok = False
                self._leg_ik_error = str(payload)
                logger.warning(
                    "Leg IK unavailable (%s); foot sliders shown but legs use joint values", payload
                )

    def _stop_ik_process(self) -> None:
        if self._ik_proc is None:
            return
        try:
            if self._ik_stop_evt is not None:
                self._ik_stop_evt.set()
            if self._ik_target_q is not None:
                try:
                    self._ik_target_q.put_nowait(None)
                except queue.Full:
                    pass
            self._ik_proc.join(timeout=2.0)
            if self._ik_proc.is_alive():
                self._ik_proc.terminate()
        finally:
            self._ik_proc = None

    def connect(self, calibrate: bool = True) -> None:
        require_package("pygame", extra="pygame-dep", import_name="pygame")
        import pygame

        self._foot_divider_x = self.config.foot_panel_width
        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("G1 SONIC — foot IK + upper-body sliders")
        self._screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        self._font = pygame.font.SysFont("dejavusans", 15)
        self._small_font = pygame.font.SysFont("dejavusans", 12)
        self._clock = pygame.time.Clock()
        self._init_leg_ik()
        self._connected = True
        logger.info("G1 sonic slider UI ready (R=reset, wheel=scroll, Esc=quit)")

    def configure(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def _reset_pose(self) -> None:
        self._values[:] = DEFAULT_ANGLES
        if self._leg_ik_ok and self._standing_foot_xyz is not None:
            self._foot_xyz[:] = self._standing_foot_xyz
            self._publish_ik_target()

    def _foot_row_rect(self, foot_idx: int) -> tuple[int, int, int, int]:
        y = HEADER_H + foot_idx * self.config.row_height
        track_x = MARGIN + 88
        track_w = self.config.foot_panel_width - track_x - MARGIN
        return track_x, y, track_w, self.config.row_height

    def _joint_row_rect(self, ui_idx: int) -> tuple[int, int, int, int]:
        joint_idx = UPPER_BODY_INDICES[ui_idx] if self._leg_ik_ok else ui_idx
        row = ui_idx
        y = HEADER_H + row * self.config.row_height - self._scroll_y
        track_x = self._foot_divider_x + MARGIN + LABEL_W
        track_w = self.config.slider_width
        return track_x, y, track_w, self.config.row_height, joint_idx

    def _value_from_track(self, lo: float, hi: float, mouse_x: int, track_x: int, track_w: int) -> float:
        t = (mouse_x - track_x) / max(track_w, 1)
        t = float(np.clip(t, 0.0, 1.0))
        return lo + t * (hi - lo)

    def _knob_x(self, val: float, lo: float, hi: float, track_x: int, track_w: int) -> int:
        span = hi - lo if hi > lo else 1.0
        t = (val - lo) / span
        return int(track_x + t * track_w)

    def _handle_events(self) -> bool:
        pygame = self._pygame
        num_joint_rows = len(UPPER_BODY_INDICES) if self._leg_ik_ok else NUM_JOINTS
        max_scroll = max(0, num_joint_rows * self.config.row_height - self.config.window_height + HEADER_H)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._reset_pose()
            if event.type == pygame.MOUSEWHEEL:
                self._scroll_y = int(np.clip(self._scroll_y - event.y * self.config.scroll_step, 0, max_scroll))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if self.config.use_leg_ik and mx < self._foot_divider_x:
                    for fi in range(NUM_FOOT_SLIDERS):
                        rx, ry, rw, rh = self._foot_row_rect(fi)
                        if ry + 4 <= my <= ry + rh - 4 and rx <= mx <= rx + rw:
                            self._drag_foot = fi
                            self._foot_xyz[fi] = self._value_from_track(
                                float(self._foot_lo[fi]),
                                float(self._foot_hi[fi]),
                                mx,
                                rx,
                                rw,
                            )
                            break
                else:
                    for ui in range(num_joint_rows):
                        rx, ry, rw, rh, ji = self._joint_row_rect(ui)
                        if ry + rh < HEADER_H or ry > self.config.window_height:
                            continue
                        if ry + 4 <= my <= ry + rh - 4 and rx <= mx <= rx + rw:
                            self._drag_joint = ji
                            self._values[ji] = self._value_from_track(
                                float(JOINT_LO[ji]), float(JOINT_HI[ji]), mx, rx, rw
                            )
                            break
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._drag_foot = None
                self._drag_joint = None
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self._drag_foot is not None:
                    rx, _, rw, _ = self._foot_row_rect(self._drag_foot)
                    self._foot_xyz[self._drag_foot] = self._value_from_track(
                        float(self._foot_lo[self._drag_foot]),
                        float(self._foot_hi[self._drag_foot]),
                        mx,
                        rx,
                        rw,
                    )
                elif self._drag_joint is not None:
                    for ui in range(num_joint_rows):
                        rx, ry, rw, rh, ji = self._joint_row_rect(ui)
                        if ji == self._drag_joint:
                            self._values[ji] = self._value_from_track(
                                float(JOINT_LO[ji]), float(JOINT_HI[ji]), mx, rx, rw
                            )
                            break
        return True

    def _draw_foot_panel(self) -> None:
        pygame = self._pygame
        screen = self._screen
        panel_title = self._small_font.render("Foot IK (pelvis)", True, (180, 200, 255))
        screen.blit(panel_title, (MARGIN, 38))
        if self._leg_ik_error:
            err = self._leg_ik_error if len(self._leg_ik_error) < 42 else self._leg_ik_error[:39] + "..."
            screen.blit(self._small_font.render(f"IK off: {err}", True, (255, 120, 120)), (MARGIN, 50))
        pygame.draw.line(
            screen,
            (60, 60, 70),
            (self._foot_divider_x - 1, HEADER_H - 4),
            (self._foot_divider_x - 1, self.config.window_height),
            1,
        )

        for fi, label in enumerate(FOOT_LABELS):
            rx, ry, rw, rh = self._foot_row_rect(fi)
            lo, hi = float(self._foot_lo[fi]), float(self._foot_hi[fi])
            txt = self._small_font.render(label, True, (190, 210, 230))
            screen.blit(txt, (MARGIN, ry + 4))
            track_y = ry + rh // 2 - 2
            pygame.draw.rect(screen, (45, 55, 70), (rx, track_y, rw, 4), border_radius=2)
            kx = self._knob_x(float(self._foot_xyz[fi]), lo, hi, rx, rw)
            pygame.draw.rect(screen, (70, 140, 220), (rx, track_y, max(0, kx - rx), 4), border_radius=2)
            pygame.draw.rect(screen, (200, 225, 255), (kx - KNOB_W // 2, track_y - 5, KNOB_W, 14), border_radius=3)
            val_txt = self._small_font.render(f"{self._foot_xyz[fi]:+.3f}", True, (150, 200, 220))
            screen.blit(val_txt, (rx + rw + 4, ry + 4))

    def _draw_joint_panel(self) -> None:
        pygame = self._pygame
        screen = self._screen
        num_joint_rows = len(UPPER_BODY_INDICES) if self._leg_ik_ok else NUM_JOINTS
        joint_title = "Waist + arms" if self._leg_ik_ok else "All joints"
        screen.blit(
            self._small_font.render(joint_title, True, (180, 180, 190)),
            (self._foot_divider_x + MARGIN, 38),
        )

        for ui in range(num_joint_rows):
            rx, ry, rw, rh, ji = self._joint_row_rect(ui)
            if ry + rh < HEADER_H or ry > self.config.window_height:
                continue
            short = JOINT_NAMES[ji].removeprefix("k")
            label = self._small_font.render(f"{ji:02d} {short}", True, (200, 200, 210))
            screen.blit(label, (self._foot_divider_x + MARGIN, ry + 4))
            track_y = ry + rh // 2 - 2
            lo, hi = float(JOINT_LO[ji]), float(JOINT_HI[ji])
            pygame.draw.rect(screen, (55, 55, 65), (rx, track_y, rw, 4), border_radius=2)
            kx = self._knob_x(float(self._values[ji]), lo, hi, rx, rw)
            pygame.draw.rect(screen, (80, 160, 255), (rx, track_y, max(0, kx - rx), 4), border_radius=2)
            pygame.draw.rect(screen, (220, 235, 255), (kx - KNOB_W // 2, track_y - 5, KNOB_W, 14), border_radius=3)
            val_txt = self._small_font.render(f"{self._values[ji]:+.3f}", True, (170, 220, 170))
            screen.blit(val_txt, (rx + rw + 8, ry + 4))

    def _draw(self) -> None:
        pygame = self._pygame
        screen = self._screen
        screen.fill((28, 28, 32))
        title = self._font.render("G1 reference → SONIC encoder mode 0", True, (230, 230, 235))
        hint = self._small_font.render("Foot xyz (left) · waist/arms (right) · R reset · Esc quit", True, (150, 150, 160))
        screen.blit(title, (MARGIN, 10))
        screen.blit(hint, (MARGIN, 28))
        if self.config.use_leg_ik:
            self._draw_foot_panel()
        self._draw_joint_panel()
        pygame.display.flip()

    def get_action(self) -> dict[str, float]:
        if not self._connected:
            return {f"{name}.q": float(self._values[i]) for i, name in enumerate(JOINT_NAMES)}
        if not self._handle_events():
            raise KeyboardInterrupt("G1 sonic slider window closed")
        if self._leg_ik_ok:
            # Read the newest solution from the solver process and hand it the latest
            # foot target — never block the teleop loop on the IPOPT solve.
            self._pump_ik_results()
            self._publish_ik_target()
        self._draw()
        self._clock.tick(60)
        return {f"{name}.q": float(self._values[i]) for i, name in enumerate(JOINT_NAMES)}

    def send_feedback(self, feedback: dict) -> None:
        del feedback

    def disconnect(self) -> None:
        self._stop_ik_process()
        if self._pygame is not None:
            self._pygame.quit()
        self._connected = False
        self._screen = None
