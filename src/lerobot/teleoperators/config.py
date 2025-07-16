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
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class TeleoperatorConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different teleoperators of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    @classmethod
    def get_known_choices(cls):
        choices = super().get_known_choices()
        for arg in sys.argv:
            if arg.startswith("--teleop.type="):
                class_path = arg.split("=")[1]
                if '.' in class_path:
                    module_path, _ = class_path.rsplit('.', 1)
                    __import__(module_path)

        return choices

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
