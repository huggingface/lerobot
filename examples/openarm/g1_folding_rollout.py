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

"""Roll out the OpenArm folding policy on a Unitree G1, policy in-process.

The G1 counterpart of ``rollout_retarget.py``: same entry point and same CLI, with the robot
wrapped in :class:`G1FoldingRobot` so the policy sees an OpenArm.

This runs the policy in the same process as the robot, which means the controller runs
wherever this script does. **In simulation that is fine and this is the convenient way to
test the retargeting end to end.** On real hardware prefer ``g1_folding_async.py``, which
keeps the balance loop on the robot and puts only the policy on the laptop's GPU.

Environment knobs: ``G1_MAX_STEP_DEG`` (per-tick joint clamp), ``G1_RETARGET_ITERS``,
``G1_RETARGET_WAIST``.
"""

from __future__ import annotations

import logging

from g1_folding_async import wrap_factory

logger = logging.getLogger("g1_folding_rollout")


def main() -> None:
    import lerobot.rollout.context as context

    context.make_robot_from_config = wrap_factory(context.make_robot_from_config)

    from rollout_retarget import _patch_rtc_realtime

    _patch_rtc_realtime()

    from lerobot.scripts.lerobot_rollout import main as rollout_main

    rollout_main()


if __name__ == "__main__":
    main()
