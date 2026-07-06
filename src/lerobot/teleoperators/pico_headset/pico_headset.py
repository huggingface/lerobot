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

from lerobot.robots.unitree_g1.smpl_stream import SMPL_OBS_DIM, SmplStream
from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .config_pico_headset import PicoHeadsetConfig

logger = logging.getLogger(__name__)

# Flat action keys carrying the 720-vec SONIC whole-body reference window. Kept as
# scalar floats so the reference flows unchanged through the standard lerobot action
# pipeline; SonicWholeBodyController reassembles them into smpl_joints_10frame_step1.
SMPL_ACTION_PREFIX = "smpl."


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
        return {f"{SMPL_ACTION_PREFIX}{i}": float for i in range(SMPL_OBS_DIM)}

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
        # Hold back until the headset is actually streaming, so the controller does
        # not switch to whole-body tracking of an all-zero (collapsed) pose.
        if not self._stream.has_data:
            return {}
        return {f"{SMPL_ACTION_PREFIX}{i}": float(v) for i, v in enumerate(window)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
