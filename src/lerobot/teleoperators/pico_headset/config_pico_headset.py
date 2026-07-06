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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico_headset")
@dataclass
class PicoHeadsetConfig(TeleoperatorConfig):
    """PICO full-body headset teleop: live SMPL over the rt/smpl ZMQ stream.

    Consumes the ``rt/smpl`` channel published by the GEAR PICO manager
    (``gear_sonic/scripts/pico_manager_thread_server.py``) and emits the whole-body
    SONIC reference window (``encode_mode == 2``) for SonicWholeBodyController.
    """

    smpl_host: str = "127.0.0.1"
    """Host of the pico_manager rt/smpl publisher (the laptop bridging the PICO)."""
    smpl_port: int = 5560
    """Port of the rt/smpl publisher."""
    stale_after_s: float = 0.5
    """Warn if no fresh headset frame arrives within this many seconds."""
