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

"""PICO full-body headset teleoperator (live SMPL -> SONIC whole-body)."""

import logging
from typing import Any

from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .config_pico_headset import PicoHeadsetConfig
from .smpl_constants import (
    LOCO_AXES_PREFIX,
    LOCO_BTN_PREFIX,
    LOCO_N_AXES,
    LOCO_N_BTN,
    ROOT_ACTION_DIM,
    ROOT_ACTION_PREFIX,
    SMPL_ACTION_PREFIX,
    SMPL_OBS_DIM,
    VR3_ORN_DIM,
    VR3_ORN_PREFIX,
    VR3_POS_DIM,
    VR3_POS_PREFIX,
)
from .smpl_stream import SmplStream

logger = logging.getLogger(__name__)


class PicoHeadset(Teleoperator):
    """Streams full-body SMPL from a PICO headset as a SONIC whole-body reference.

    Subscribes to the ``rt/smpl`` ZMQ channel and, once real frames are flowing,
    emits the 720-element encoder window as ``smpl.{i}`` floats. Before the first
    frame arrives it emits no SMPL keys, so the robot stays in safe locomotion mode
    rather than tracking a zero pose.
    """

    config_class = PicoHeadsetConfig
    name = "pico_headset"

    def __init__(self, config: PicoHeadsetConfig):
        super().__init__(config)
        self.config = config
        self._stream: SmplStream | None = None

    @property
    def action_features(self) -> dict:
        if self.config.mode == "vr3":
            feats = {f"{VR3_POS_PREFIX}{i}": float for i in range(VR3_POS_DIM)}
            feats.update({f"{VR3_ORN_PREFIX}{i}": float for i in range(VR3_ORN_DIM)})
            # Controller-stick locomotion travels alongside the VR targets.
            feats.update({f"{LOCO_AXES_PREFIX}{i}": float for i in range(LOCO_N_AXES)})
            feats.update({f"{LOCO_BTN_PREFIX}{i}": float for i in range(LOCO_N_BTN)})
            return feats
        feats = {f"{SMPL_ACTION_PREFIX}{i}": float for i in range(SMPL_OBS_DIM)}
        feats.update({f"{ROOT_ACTION_PREFIX}{i}": float for i in range(ROOT_ACTION_DIM)})
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._stream is not None

    def connect(self, calibrate: bool = True) -> None:
        if self._stream is not None:
            raise RuntimeError(f"{self} already connected")
        self._stream = SmplStream(
            host=self.config.smpl_host,
            port=self.config.smpl_port,
            stale_after_s=self.config.stale_after_s,
        )
        logger.info(
            "PicoHeadset subscribed to rt/smpl @ tcp://%s:%d",
            self.config.smpl_host,
            self.config.smpl_port,
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Calibration happens on the headset / pico_manager side, not here.
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if self._stream is None:
            raise RuntimeError(f"{self} is not connected")
        window = self._stream.step()
        # Emit a reference only while the headset is actively streaming: hold back
        # before the first frame (so the controller doesn't track an all-zero
        # collapsed pose) and once the stream goes stale (so the controller falls
        # back to a safe standing/locomotion mode instead of freezing on the last
        # pose).
        if self.config.mode == "vr3":
            # Sparse 3-point upper-body teleop (encode_mode 1). Gated on fresh vr3_*
            # frames only (independent of the SMPL window), so the controller-state
            # source (head + controllers, no body tracking) works. Emit nothing
            # otherwise and stay in locomotion.
            if not self._stream.has_fresh_vr3:
                return {}
            action = {f"{VR3_POS_PREFIX}{i}": float(v) for i, v in enumerate(self._stream.vr3_pos)}
            action.update({f"{VR3_ORN_PREFIX}{i}": float(v) for i, v in enumerate(self._stream.vr3_orn)})
            # Forward controller-stick locomotion when present, so the planner can
            # steer walking/turning under the upper-body tracking (encode_mode 1).
            if self._stream.has_fresh_loco:
                action.update(
                    {f"{LOCO_AXES_PREFIX}{i}": float(v) for i, v in enumerate(self._stream.loco_axes)}
                )
                action.update(
                    {f"{LOCO_BTN_PREFIX}{i}": float(v) for i, v in enumerate(self._stream.loco_buttons)}
                )
            return action
        if not self._stream.has_data or self._stream.is_stale:
            return {}
        action = {f"{SMPL_ACTION_PREFIX}{i}": float(v) for i, v in enumerate(window)}
        action.update({f"{ROOT_ACTION_PREFIX}{i}": float(v) for i, v in enumerate(self._stream.root_quat)})
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
