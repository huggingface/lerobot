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

"""Policy deployment engine with pluggable rollout strategies.

``lerobot-rollout`` is the single CLI for running trained policies on
real robots.

Strategies
----------
    --strategy.type=base       Autonomous rollout, no recording
    --strategy.type=sentry     Continuous recording with auto-upload
    --strategy.type=highlight  Ring buffer + keystroke save
    --strategy.type=dagger     Human-in-the-loop (DAgger / RaC)

Inference backends
------------------
    --inference.type=sync      One policy call per control tick (default)
    --inference.type=rtc       Real-Time Chunking for slow VLA models

Usage examples
--------------
::

    # Base mode — quick evaluation with sync inference
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=30

    # Base mode — RTC inference for slow VLAs (Pi0, Pi0.5, SmolVLA)
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --inference.rtc.max_guidance_weight=10.0 \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="pick up cube" --duration=60

    # Sentry mode — continuous recording with periodic upload
    lerobot-rollout \\
        --strategy.type=sentry \\
        --strategy.upload_every_n_episodes=5 \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" --duration=3600

    # Highlight mode — ring buffer, press 's' to save, 'h' to push
    lerobot-rollout \\
        --strategy.type=highlight \\
        --strategy.ring_buffer_seconds=30 \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_highlight_data \\
        --dataset.single_task="pick up cube"

    # DAgger mode — human-in-the-loop corrections only
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.num_episodes=20 \\
        --policy.path=outputs/pretrain/checkpoints/last/pretrained_model \\
        --robot.type=bi_openarm_follower \\
        --teleop.type=openarm_mini \\
        --dataset.repo_id=user/rollout_hil_data \\
        --dataset.single_task="Fold the T-shirt"

    # DAgger mode — continuous recording with RTC inference
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.record_autonomous=true \\
        --strategy.num_episodes=50 \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --policy.path=user/my_pi0_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so101_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_dagger_rtc_data \\
        --dataset.single_task="Grasp the block"

    # With Rerun visualization and torch.compile
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --use_torch_compile=true

    # Resume a previous sentry recording session
    lerobot-rollout \\
        --strategy.type=sentry \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" \\
        --resume=true
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

logger = logging.getLogger(__name__)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Main entry point for policy deployment."""
    init_logging()

    if cfg.display_data:
        logger.info("Initializing Rerun visualization (ip=%s, port=%s)", cfg.display_ip, cfg.display_port)
        init_rerun(session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building rollout context...")
    ctx = build_rollout_context(cfg, shutdown_event)

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Rollout finished")


def main():
    """CLI entry point for ``lerobot-rollout``."""
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
