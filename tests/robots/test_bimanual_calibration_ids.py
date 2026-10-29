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

# Example of running a specific test:
# ```bash
# pytest tests/robots/test_bimanual_calibration_ids.py
# ```

import pytest

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig


def _bi_so_follower(calibration_dir):
    return BiSOFollower(
        BiSOFollowerConfig(
            calibration_dir=calibration_dir,
            left_arm_config=SOFollowerConfig(port="/dev/null0"),
            right_arm_config=SOFollowerConfig(port="/dev/null1"),
        )
    )


def _bi_so_leader(calibration_dir):
    return BiSOLeader(
        BiSOLeaderConfig(
            calibration_dir=calibration_dir,
            left_arm_config=SOLeaderConfig(port="/dev/null0"),
            right_arm_config=SOLeaderConfig(port="/dev/null1"),
        )
    )


@pytest.mark.parametrize("make_device", [_bi_so_follower, _bi_so_leader], ids=["follower", "leader"])
def test_arms_do_not_share_a_calibration_file_without_an_explicit_id(make_device, tmp_path):
    """Both arms used to fall back to `id=None`, i.e. the same `None.json`.

    Calibrating a pair then left the right arm's homing offsets and limits in the file the
    left arm reads back, so the left arm was programmed with the right arm's calibration.
    """
    device = make_device(tmp_path)

    assert device.left_arm.id == "left"
    assert device.right_arm.id == "right"
    assert device.left_arm.calibration_fpath != device.right_arm.calibration_fpath


@pytest.mark.parametrize("make_device", [_bi_so_follower, _bi_so_leader], ids=["follower", "leader"])
def test_explicit_id_still_prefixes_each_arm(make_device, tmp_path):
    """An explicit id keeps the documented `<id>_left` / `<id>_right` naming."""
    device = make_device(tmp_path)
    device.config.id = "pair"
    rebuilt = type(device)(device.config)

    assert rebuilt.left_arm.id == "pair_left"
    assert rebuilt.right_arm.id == "pair_right"
    assert rebuilt.left_arm.calibration_fpath != rebuilt.right_arm.calibration_fpath
