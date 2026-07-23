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
from typing import Literal

import draccus
import pytest
from draccus.utils import DecodingError

import lerobot.configs.parser  # noqa: F401


def test_literal_field_accepts_valid_value():
    @dataclass
    class Config:
        mode: Literal["train", "eval", "record"] = "train"

    cfg = draccus.parse(Config, args=["--mode=eval"])
    assert cfg.mode == "eval"


def test_literal_field_rejects_invalid_value():
    @dataclass
    class Config:
        mode: Literal["train", "eval", "record"] = "train"

    with pytest.raises(DecodingError, match="bogus"):
        draccus.parse(Config, args=["--mode=bogus"])
