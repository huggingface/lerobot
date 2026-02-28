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

from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import (
    OpenArmFollowerConfig,
    OpenArmFollowerConfigBase,
)
from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
from lerobot.teleoperators.openarm_leader.config_openarm_leader import (
    OpenArmLeaderConfig,
    OpenArmLeaderConfigBase,
)
from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig


def test_openarm_follower_default_id():
    cfg = OpenArmFollowerConfig(port="can1", side="left")
    assert cfg.id == "openarm_follower_left_can1"


def test_openarm_follower_keeps_explicit_id():
    cfg = OpenArmFollowerConfig(id="my_arm", port="can1", side="left")
    assert cfg.id == "my_arm"


def test_openarm_leader_default_id():
    cfg = OpenArmLeaderConfig(port="can3")
    assert cfg.id == "openarm_leader_can3"


def test_openarm_mini_default_id():
    cfg = OpenArmMiniConfig(port_right="/dev/ttyACM0", port_left="/dev/ttyACM1")
    assert cfg.id == "openarm_mini_dev_ttyACM0_dev_ttyACM1"


def test_bi_openarm_follower_default_id():
    cfg = BiOpenArmFollowerConfig(
        left_arm_config=OpenArmFollowerConfigBase(port="can1", side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port="can0", side="right"),
    )
    assert cfg.id == "bi_openarm_follower_can1_can0"


def test_bi_openarm_leader_default_id():
    cfg = BiOpenArmLeaderConfig(
        left_arm_config=OpenArmLeaderConfigBase(port="can3"),
        right_arm_config=OpenArmLeaderConfigBase(port="can2"),
    )
    assert cfg.id == "bi_openarm_leader_can3_can2"
