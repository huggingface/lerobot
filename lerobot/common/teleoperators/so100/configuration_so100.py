#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig

# TODO(pepijn): Do this differently
BASE_CALIBRATION_DIR = Path(".cache/calibration/so100")
BASE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


@TeleoperatorConfig.register_subclass("so100")
@dataclass
class SO100TeleopConfig(TeleoperatorConfig):
    # Port to connect to the teloperator
    port: str = "/dev/tty.usbmodem58760430821"

    calibration_fpath: Path = field(default=BASE_CALIBRATION_DIR / "follower_calibration.json")
