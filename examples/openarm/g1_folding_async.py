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

"""Run the OpenArm folding policy on a Unitree G1 over async inference.

Drop-in for ``lerobot.async_inference.robot_client``: same CLI, but the robot it builds is
wrapped in :class:`G1FoldingRobot` so the policy sees an OpenArm. Only the wrapping differs,
so everything the stock client does -- chunk aggregation, RTC, reconnects -- is unchanged.

Process placement is the point. The client runs **on the G1**, which puts the
``SonicLowerBodyController`` loop in-process on the robot rather than across the network,
and leaves only the policy (the part that wants a real GPU) on the laptop:

    # laptop (GPU)
    python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

    # robot, terminal 1 -- DDS bridge + cameras, loopback only
    python src/lerobot/robots/unitree_g1/run_g1_server.py --grippers \
        --cameras 'left_wrist=/dev/video8@1280x720,right_wrist=/dev/video4@1280x720,base=/dev/video6@640x480'

    # robot, terminal 2 -- this script
    python examples/openarm/g1_folding_async.py --server_address=<laptop>:8080 ...

The IK runs here too, on the Jetson, alongside SONIC's 50 Hz ONNX decode. If the gait starts
limping, turn ``G1_RETARGET_ITERS`` down before anything else: the solve is warm started from
the previous frame, so it converges well below the default 25 in steady state.

Environment knobs: ``G1_MAX_STEP_DEG`` (per-tick joint clamp), ``G1_RETARGET_ITERS``,
``G1_RETARGET_WAIST``.
"""

from __future__ import annotations

import logging
import os

from lerobot.robots.unitree_g1.g1_folding_robot import DEFAULT_MAX_STEP_DEG, G1FoldingRobot
from lerobot.robots.unitree_g1.g1_openarm_retarget import IK_ITERS, REVERSE_IK_ITERS

logger = logging.getLogger("g1_folding_async")


def wrap_factory(orig):
    """Wrap a ``make_robot_from_config`` so whatever it builds comes back OpenArm-shaped."""
    ik_iters = int(os.environ.get("G1_RETARGET_ITERS", str(IK_ITERS)))
    reverse_iters = int(os.environ.get("G1_RETARGET_REVERSE_ITERS", str(REVERSE_IK_ITERS)))
    use_waist = os.environ.get("G1_RETARGET_WAIST", "0") != "0"
    max_step_deg = float(os.environ.get("G1_MAX_STEP_DEG", str(DEFAULT_MAX_STEP_DEG)))

    def factory(cfg):
        real = orig(cfg)
        logger.info(
            f"Wrapping {type(real).__name__} with G1FoldingRobot (iters={ik_iters}/{reverse_iters}, "
            f"waist={use_waist}, max_step={max_step_deg} deg)"
        )
        return G1FoldingRobot(
            real,
            ik_iters=ik_iters,
            reverse_ik_iters=reverse_iters,
            use_waist=use_waist,
            max_step_deg=max_step_deg,
        )

    return factory


def main() -> None:
    from lerobot.async_inference import robot_client
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    robot_client.make_robot_from_config = wrap_factory(robot_client.make_robot_from_config)
    robot_client.async_client()


if __name__ == "__main__":
    main()
