#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


@TeleoperatorConfig.register_subclass("bi_yam_leader")
@dataclass
class BiYamLeaderConfig(TeleoperatorConfig):
    # Server ports for left and right arm leaders
    # These should be different from the follower ports
    # Note: You'll need to run separate server processes for the leader arms
    # that expose their state for reading (see i2rt minimum_gello.py)
    left_arm_port: int = 5002
    right_arm_port: int = 5001

    # Server host (usually localhost for local setup)
    server_host: str = "localhost"
