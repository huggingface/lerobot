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
wrapped in :class:`G1FoldingRobot` so the policy sees an OpenArm. Policy, IK and rollout loop
all run here; on the laptop a full tick of both solves costs ~8 ms, against ~50 ms of budget
at 20 Hz.

Running here does **not** put the balance loop here. What decides that is the robot config:
with ``--robot.is_simulation=false`` and no ``--robot.onboard``, ``UnitreeG1`` is the thin
client, and the controller runs on the robot under ``run_g1_server.py --onboard``. So on real
hardware, start that first::

    # robot -- SONIC holds the stand, cameras and hands served from here
    python src/lerobot/robots/unitree_g1/run_g1_server.py --onboard \\
        --controller SonicLowerBodyController --grippers \\
        --cameras '<name>=<by-path device>@WxH,...'

    # laptop -- this script
    python examples/openarm/g1_folding_rollout.py --robot.type=unitree_g1 ...

In simulation the controller does run in-process, because there is no robot to run it on.

Environment knobs: ``G1_MAX_STEP_DEG`` (per-tick joint clamp), ``G1_RETARGET_ITERS``,
``G1_RETARGET_REVERSE_ITERS``, ``G1_RETARGET_WAIST``.
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
