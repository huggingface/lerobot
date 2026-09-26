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
from __future__ import annotations

import pytest

libero = pytest.importorskip("lerobot.envs.libero")

strip_perturbation_from_instruction = libero.strip_perturbation_from_instruction

BASE_INSTRUCTION = "pick up the black bowl next to the ramekin and place it on the plate"


@pytest.mark.parametrize(
    ("task_name", "instruction"),
    [
        (
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_table_14",
            f"{BASE_INSTRUCTION} table 14",
        ),
        (
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_tb_3",
            f"{BASE_INSTRUCTION} tb 3",
        ),
        (
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_1",
            f"{BASE_INSTRUCTION} view 0 0 100 0 0 initstate 1",
        ),
        (
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_light_2",
            f"{BASE_INSTRUCTION} light 2",
        ),
    ],
)
def test_perturbation_suffix_is_removed_from_instruction(task_name: str, instruction: str) -> None:
    assert strip_perturbation_from_instruction(task_name, instruction) == BASE_INSTRUCTION


def test_language_variants_keep_their_rephrased_instruction() -> None:
    """`_language_` variants read a clean rephrasing from the BDDL file, so leave them alone."""
    task_name = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0"
    instruction = "grab the dark bowl beside the ramekin and set it on the plate"

    assert strip_perturbation_from_instruction(task_name, instruction) == instruction


def test_unperturbed_instruction_is_unchanged() -> None:
    task_name = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"

    assert strip_perturbation_from_instruction(task_name, BASE_INSTRUCTION) == BASE_INSTRUCTION
