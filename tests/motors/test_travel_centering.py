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

import pytest

from lerobot.motors.motors_bus import center_homing_on_travel, unwrap_step


@pytest.mark.parametrize(("prev", "curr", "step"), [(4090, 5, 11), (5, 4090, -11), (100, 300, 200)])
def test_unwrap_step_takes_the_short_way_across_the_seam(prev, curr, step):
    assert unwrap_step(prev, curr, 4096) == step


def test_a_travel_already_centred_keeps_its_homing():
    # The follower's measured travel (2026-09-03), homed with ENTER at its middle.
    assert center_homing_on_travel(2048, 113, 3981, 4096) == (2048, 113, 3981)


def test_rehoming_puts_the_middle_of_the_travel_at_half_a_turn():
    offset, lo, hi = center_homing_on_travel(0, -500, 3368, 4096)
    # Present = Actual - Homing: the old middle (1434) must now read 2047.
    assert 1434 - (offset - 0) == 2047
    assert (lo, hi) == (113, 3981)


def test_the_offset_stays_within_one_turn_of_zero():
    offset, _, _ = center_homing_on_travel(2000, 3000, 6000, 4096)
    assert -2047 <= offset <= 2048


def test_a_joint_that_spins_a_full_turn_has_no_middle():
    assert center_homing_on_travel(2048, -10, 4100, 4096) is None
