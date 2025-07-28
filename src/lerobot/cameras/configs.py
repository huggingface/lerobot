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

import abc
from dataclasses import dataclass
from enum import Enum

import draccus


class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"


class Cv2Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


@dataclass(kw_only=True)
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    fps: int | None = None
    width: int | None = None
    height: int | None = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
