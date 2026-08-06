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

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import TeleoperatorConfig


class PhoneOS(Enum):
    """Which phone platform a `Phone` teleoperator talks to, selecting its backend implementation.

    **Attributes**:
        - **ANDROID** (`str`) -- WebXR-based backend (`AndroidPhone`), driven through the `teleop` Python
          package.
        - **IOS** (`str`) -- ARKit-based backend (`IOSPhone`), driven through the HEBI Mobile I/O app.
    """

    ANDROID = "android"
    IOS = "ios"


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneConfig(TeleoperatorConfig):
    """Configuration for the [`~teleoperators.phone.Phone`] teleoperator.

    Args:
        phone_os (`PhoneOS`, *optional*, defaults to `PhoneOS.IOS`):
            Which phone platform and backend to use. `PhoneOS.IOS` talks to the HEBI Mobile I/O app over
            ARKit; `PhoneOS.ANDROID` talks to a browser WebXR session over the `teleop` package.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.

    Note:
        `camera_offset` is a fixed class attribute, not a constructor argument, so it currently cannot be
        overridden per instance or from the command line. It defaults to the offset between an iPhone 14
        Pro's camera and the phone's physical center (2cm lateral, 4cm vertical) and is applied to
        translate the ARKit/WebXR camera pose into the phone's own frame.

    Example:
        ```python
        >>> from lerobot.teleoperators.phone import PhoneConfig
        >>> from lerobot.teleoperators.phone.config_phone import PhoneOS
        >>> config = PhoneConfig(phone_os=PhoneOS.ANDROID)
        >>> config.phone_os
        <PhoneOS.ANDROID: 'android'>
        ```
    """

    phone_os: PhoneOS = PhoneOS.IOS
    camera_offset = np.array(
        [0.0, -0.02, 0.04]
    )  # iPhone 14 Pro camera is 2cm off center and 4cm above center
