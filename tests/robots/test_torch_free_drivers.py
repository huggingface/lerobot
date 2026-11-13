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

"""Robot and camera drivers, and the loop helpers, load without PyTorch.

The Unitree G1 is the exception: its config imports the simulation environments, which need PyTorch.
"""

import pytest

from tests.torch_free import assert_runs_without_torch

_IMPORTS = """
import lerobot.cameras.opencv
import lerobot.robots.bi_openarm_follower.bi_openarm_follower
import lerobot.robots.bi_rebot_b601_follower.bi_rebot_b601_follower
import lerobot.robots.bi_so_follower.bi_so_follower
import lerobot.robots.bi_yam_follower.bi_yam_follower
import lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus
import lerobot.robots.hope_jr.hope_jr_arm
import lerobot.robots.hope_jr.hope_jr_hand
import lerobot.robots.koch_follower.koch_follower
import lerobot.robots.lekiwi.lekiwi
import lerobot.robots.lekiwi.lekiwi_client
import lerobot.robots.omx_follower.omx_follower
import lerobot.robots.openarm_follower.openarm_follower
import lerobot.robots.reachy2.robot_reachy2
import lerobot.robots.rebot_b601_follower.rebot_b601_follower
import lerobot.robots.so_follower.so_follower
import lerobot.robots.yam_follower.yam_follower
import lerobot.teleoperators.bi_openarm_leader.bi_openarm_leader
import lerobot.teleoperators.bi_openarm_mini.bi_openarm_mini
import lerobot.teleoperators.bi_rebot_102_leader.bi_rebot_102_leader
import lerobot.teleoperators.bi_so_leader.bi_so_leader
import lerobot.teleoperators.gamepad.teleop_gamepad
import lerobot.teleoperators.keyboard.teleop_keyboard
import lerobot.teleoperators.openarm_leader.openarm_leader
import lerobot.teleoperators.openarm_mini.openarm_mini
import lerobot.teleoperators.phone.teleop_phone
import lerobot.teleoperators.rebot_102_leader.rebot_102_leader
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.utils import init_logging
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import policy_features, return_to_position
"""


def test_drivers_and_loop_helpers_load_without_torch():
    assert_runs_without_torch(_IMPORTS)


# Only making a robot needs its motor library, so the import check above runs in every test tier.
@pytest.mark.hardware
def test_a_robot_is_made_without_torch(tmp_path):
    pytest.importorskip("rustypot", reason="rustypot is required (install lerobot[serial-motors])")
    assert_runs_without_torch(
        _IMPORTS
        + """
import sys
from pathlib import Path

# Not the user's calibration files, which this test does not check.
config = SO101FollowerConfig(port="/dev/null", calibration_dir=Path(sys.argv[1]))
robot = make_robot_from_config(config)
policy_features(robot, None)
""",
        str(tmp_path),
    )
