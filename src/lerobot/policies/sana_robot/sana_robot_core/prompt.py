# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/data/format/prompt.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from lerobot.policies.sana_robot.sana_robot_core.robot80 import ACTION_MODE_EEF, ACTION_MODE_JOINT

ACTION_MODE_TEXT = {
    ACTION_MODE_EEF: (
        "robot_base_eef; end-effector targets are absolute future poses expressed in the robot base frame, "
        "joint targets are absolute future targets, and gripper targets are absolute future targets"
    ),
    ACTION_MODE_JOINT: (
        "joint_only; joint and gripper targets are absolute future targets and no end-effector action is supervised"
    ),
}


def prompt_line(name: str, value: str) -> str:
    """Return one prompt line ``name: value.`` with a single closing period."""
    return f"{name}: {str(value).strip().rstrip('.')}."


def build_prompts(embodiment: str, action_mode: str, view_descriptions, instruction: str | None) -> tuple:
    """Return the V + 1 prompts of a sample, one per view and one for the robot rows; without an instruction they are the unconditional prompts."""
    common = (
        prompt_line("Embodiment Type", embodiment),
        prompt_line("Action Mode", ACTION_MODE_TEXT[action_mode]),
    )
    task = (prompt_line("Instruction", instruction),) if instruction else ()
    views = tuple(
        "\n".join((*common, prompt_line("Observation View", description), *task))
        for description in view_descriptions
    )
    return (*views, "\n".join((*common, *task)))
