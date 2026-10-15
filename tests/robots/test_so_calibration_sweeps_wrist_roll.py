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

"""`wrist_roll` is swept like every other joint, not assigned `0-4095` on faith.

Calibration used to exclude it and write the full range outright, on the assumption that it turns
freely. Measured on this arm 2026-09-03: it reaches 113-3981 (340 deg) and is stopped from both
directions by a corner of the printed gripper housing. The assumption made the joint with the
tightest travel on the arm the one joint no position check could fence.

These tests are what the hardware measurement cannot be: a check that runs on every commit. They
assert the two halves that matter -- that the sweep is asked about every motor, and that whatever
it reports for `wrist_roll` is what gets written.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.so_leader import SOLeader

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# The travel this arm actually has, measured by hand with torque off on 2026-09-03.
SWEPT_MINS = dict.fromkeys(JOINTS, 800) | {"wrist_roll": 113}
SWEPT_MAXES = dict.fromkeys(JOINTS, 3200) | {"wrist_roll": 3981}


def _stub(cls):
    bus = MagicMock()
    bus.motors = {
        name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(JOINTS)
    }
    bus.model_resolution_table = {"sts3215": 4096}
    bus.set_half_turn_homings.return_value = dict.fromkeys(JOINTS, 2048)
    bus.record_ranges_of_motion.return_value = (dict(SWEPT_MINS), dict(SWEPT_MAXES))
    return SimpleNamespace(
        bus=bus,
        calibration={},
        calibration_fpath="/dev/null",
        _save_calibration=MagicMock(),
        __class__=cls,
    )


@pytest.mark.parametrize("cls", [SOFollower, SOLeader])
def test_every_motor_is_swept_including_wrist_roll(cls):
    stub = _stub(cls)
    with patch("builtins.input", return_value=""):
        cls.calibrate(stub)

    # Called for all motors: either with no argument at all, or with a list containing wrist_roll.
    # What must never happen again is a reduced set that leaves the joint out.
    (args, kwargs) = stub.bus.record_ranges_of_motion.call_args
    swept = args[0] if args else kwargs.get("motors")
    assert swept is None or "wrist_roll" in swept


@pytest.mark.parametrize("cls", [SOFollower, SOLeader])
def test_the_swept_travel_is_what_gets_written(cls):
    stub = _stub(cls)
    with patch("builtins.input", return_value=""):
        cls.calibrate(stub)

    roll = stub.calibration["wrist_roll"]
    assert (roll.range_min, roll.range_max) == (113, 3981), (
        "wrist_roll's calibration must come from the sweep; (0, 4095) means the assignment is back"
    )
    # And the other joints are untouched by the change.
    assert (stub.calibration["elbow_flex"].range_min, stub.calibration["elbow_flex"].range_max) == (800, 3200)


@pytest.mark.parametrize("cls", [SOFollower, SOLeader])
def test_a_joint_that_really_spins_still_reports_full_travel(cls):
    """The assumption's *conclusion* was fine; only its unconditional nature was not.

    An arm whose `wrist_roll` does turn freely sweeps through the seam and records very nearly the
    full range on its own, which is what the old assignment was reaching for -- so nothing
    downstream of a genuinely free joint changes.
    """
    stub = _stub(cls)
    stub.bus.record_ranges_of_motion.return_value = (
        dict(SWEPT_MINS) | {"wrist_roll": 0},
        dict(SWEPT_MAXES) | {"wrist_roll": 4095},
    )
    with patch("builtins.input", return_value=""):
        cls.calibrate(stub)

    roll = stub.calibration["wrist_roll"]
    assert (roll.range_min, roll.range_max) == (0, 4095)


@pytest.mark.parametrize("cls", [SOFollower, SOLeader])
def test_wrist_roll_is_zeroed_on_the_middle_of_its_travel_not_on_the_enter_pose(cls):
    """ENTER held 90 deg off the travel's middle: the travel crosses the encoder seam, and without
    unwrapping it records 0-4095 and the ENTER pose becomes the zero. On 2026-09-25 that put the
    leader's and follower's wrist_roll zeros visibly apart."""
    stub = _stub(cls)
    # The same 3868-count travel as above, seen from a homing 613 counts off its middle.
    stub.bus.record_ranges_of_motion.return_value = (
        dict(SWEPT_MINS) | {"wrist_roll": -500},
        dict(SWEPT_MAXES) | {"wrist_roll": 3368},
    )
    with patch("builtins.input", return_value=""):
        cls.calibrate(stub)

    (_, kwargs) = stub.bus.record_ranges_of_motion.call_args
    assert "wrist_roll" in kwargs["unwrap"]
    roll = stub.calibration["wrist_roll"]
    assert (roll.homing_offset, roll.range_min, roll.range_max) == (2048 - 613, 113, 3981)
