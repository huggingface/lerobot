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
    --strategy.type=episodic   Episode-oriented recording with reset phases
    --strategy.type=<name>     Any strategy from an installed ``lerobot_strategy_*``
                               package (see "Bring your own strategy" in the docs)

Inference backends
------------------
    --inference.type=sync      One policy call per control tick (default)
    --inference.type=rtc       Real-Time Chunking for slow VLA models
    --inference.type=remote    Policy served by lerobot-policy-server (see remote inference docs)

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

    # Interactive session: the robot stays idle until /start is typed, then /subtask,
    # /vqa, /autosteer, /reset and /stop drive the run from stdin. With
    # --strategy.type=sentry it also records, labeling frames with their task.
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" \\
        --interactive=true

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

    # Episodic mode — episode-oriented recording with reset phases
    lerobot-rollout \\
        --strategy.type=episodic \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_episodic_data \\
        --dataset.num_episodes=20 \\
        --dataset.single_task="Grab the cube"

    # Resume a previous sentry recording session
    lerobot-rollout \\
        --strategy.type=sentry \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" \\
        --resume=true

    # Rollout with custom video encoding parameters
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --dataset.rgb_encoder.vcodec=h264 \\
        --dataset.rgb_encoder.preset=fast \\
        --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2}

    # Stream to Foxglove instead of Rerun:
    # add --display_mode=foxglove, then connect the Foxglove app to ws://127.0.0.1:8765.
"""

import logging
import math
import threading
from collections.abc import Callable
from typing import cast

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.inference import RemoteInferenceConfig
from lerobot.remote_inference import AdmissionDeniedError, ErrorCode, ProtocolError
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    lekiwi,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import (
    InteractiveSession,
    LinkedEvent,
    RolloutConfig,
    build_rollout_context,
    create_strategy,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Main entry point for policy deployment."""
    init_logging(
        console_level=cfg.inference.log_level if isinstance(cfg.inference, RemoteInferenceConfig) else "INFO"
    )

    if cfg.display_data:
        logger.info(
            "Initializing %s visualization (ip=%s, port=%s)",
            cfg.display_mode,
            cfg.display_ip,
            cfg.display_port,
        )
        init_visualization(cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    if not isinstance(shutdown_event, threading.Event):
        raise RuntimeError("ProcessSignalHandler(use_threads=True) must hand out a threading.Event")
    if cfg.interactive:
        # /reset and /stop end the control loop via the local flag; process signals still
        # propagate through the parent event.
        shutdown_event = LinkedEvent(shutdown_event)

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
        if cfg.interactive:
            logger.info("Rollout setup complete — starting interactive session (robot idle until /start)")
            InteractiveSession(strategy, ctx).run()
        else:
            logger.info("Rollout setup complete, starting rollout...")
            strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    if ctx.policy.inference.failed:
        logger.error("Rollout ended by an inference fault: %s", ctx.policy.inference.failure_traceback)
        raise SystemExit(1)
    logger.info("Rollout finished")


def _admission_denial_message(error: AdmissionDeniedError) -> str:
    """Explain expected ownership contention without promising worker completion."""
    details = error.details or {}
    blocker = details.get("admission_blocker")
    if not isinstance(blocker, str):
        blocker = None
    remaining = details.get("absence_grace_remaining_s")
    if blocker in {"absence_grace", "awaiting_initial_presence"}:
        reason = (
            "the previous client is absent"
            if blocker == "absence_grace"
            else "the previous client is still establishing its presence"
        )
        if (
            isinstance(remaining, (float, int))
            and not isinstance(remaining, bool)
            and math.isfinite(remaining)
            and remaining >= 0
        ):
            reason += f"; about {remaining:.1f} s of cleanup grace remain (worker cleanup may take longer)"
        else:
            reason += "; waiting for its cleanup grace and worker cleanup"
    elif blocker == "active_session":
        reason = "another client owns the deployment; stop that client before retrying"
    elif blocker == "unfinished_inference":
        reason = "waiting for an unfinished model call and session cleanup; completion time is unknown"
    elif blocker in {"worker_cleanup_pending", "cleanup_queue_full"}:
        reason = "waiting for worker session cleanup; completion time is unknown"
    else:
        reason = "the deployment is busy; see server logs for the session owner or pending cleanup"
    return f"Remote admission denied for deployment {error.deployment!r}: {reason}."


def main() -> None:
    """CLI entry point for ``lerobot-rollout``."""
    register_third_party_plugins()
    try:
        cast(Callable[[], None], rollout)()
    except AdmissionDeniedError as exc:
        # Context construction releases connected hardware before propagating this.
        # Other protocol/runtime errors retain their traceback and failure status.
        logger.error("%s", _admission_denial_message(exc))
        logger.debug("Remote admission server diagnostic: %s", exc)
        raise SystemExit(1) from None
    except ProtocolError as exc:
        if exc.code in {ErrorCode.INCOMPATIBLE, ErrorCode.UNSUPPORTED, ErrorCode.PROTOCOL}:
            logger.error(
                "Remote compatibility check failed (%s): %s. Compare loaded client/server builds, "
                "protocol and requested modes/schemas; resolve the mismatch before another rollout. "
                "Use --inference.log_level=DEBUG and server --log_level=DEBUG for contract details.",
                exc.code,
                exc,
            )
        # Keep the original exception and traceback, including unexpected wire failures.
        raise


if __name__ == "__main__":
    main()
