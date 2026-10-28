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

from ..configs import CameraConfig, ColorMode

__all__ = ["ColorMode", "GripetteCameraConfig"]


@CameraConfig.register_subclass("gripette")
@dataclass
class GripetteCameraConfig(CameraConfig):
    """JPEG stream from a Pollen Gripette gRPC server.

    ``addr`` is the server target, for example ``"172.18.7.107:50051"``. Width, height, and fps
    default to the server's 648x486 stream at 10 Hz so a robot can declare the feature shape
    before connecting. Frames are resized when the server image differs.
    """

    addr: str
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        if not self.addr:
            raise ValueError("`addr` cannot be empty.")
        self.color_mode = ColorMode(self.color_mode)
        for name, default in (("fps", 10), ("width", 648), ("height", 486)):
            if getattr(self, name) is None:
                setattr(self, name, default)
