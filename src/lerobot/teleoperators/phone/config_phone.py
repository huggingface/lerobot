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
    ANDROID = "android"
    IOS = "ios"


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneConfig(TeleoperatorConfig):
    phone_os: PhoneOS = PhoneOS.IOS
    camera_offset = np.array(
        [0.0, -0.02, 0.04]
    )  # iPhone 14 Pro camera is 2cm off center and 4cm above center
