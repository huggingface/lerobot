"""Interactive pilot review with native rollout and scratchpad state in the camera monitor."""

import os
import sys
import time
from dataclasses import asdict

import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

# Match the baseline comparison on this machine, whose optional Triton extensions
# cannot compile without a system C compiler. These are upstream PyTorch fallbacks.
qwen.FusedRMSNormGated = None
qwen.chunk_gated_delta_rule = None
qwen.fused_recurrent_gated_delta_rule = None
sys.path.insert(
    0,
    os.environ.get(
        "SO101_SIM_PLUGIN", "/home/maxime/github/lerobot-so101-fineart/examples/training/so101_fineart/sim"
    ),
)
import lerobot_robot_so101_sim  # noqa: E402, F401

from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402
from lerobot.scripts import lerobot_rollout  # noqa: E402

_build_context = lerobot_rollout.build_rollout_context


def build_review_context(*args, **kwargs):
    ctx = _build_context(*args, **kwargs)
    robot = ctx.hardware.robot_wrapper.inner
    policy = ctx.policy.policy
    timing = {"last_prediction_seconds": None}
    predict = policy.predict_action_chunk

    def timed_prediction(*args, **kwargs):
        started = time.monotonic()
        try:
            return predict(*args, **kwargs)
        finally:
            timing["last_prediction_seconds"] = round(time.monotonic() - started, 2)

    policy.predict_action_chunk = timed_prediction

    def status():
        state = policy._scratchpad
        return {
            "policy": "so101-g05-full-v1 (1000 optimizer updates)",
            "current_instruction": ctx.policy.inference.task,
            "actions_per_prediction": policy.config.n_action_steps,
            "scratchpad": asdict(state) if state is not None else None,
            "completed_task": policy.completed_task,
            **timing,
        }

    robot.status_provider = status

    def reset_environment():
        goal = robot.randomize_scene()
        # Old position-hold targets belong to the previous scene. A fresh wrapper
        # captures the new measured pose before any subsequent hold or action.
        previous = ctx.hardware.robot_wrapper
        wrapper = ThreadSafeRobot(robot)
        if previous.supports_hold:
            wrapper.configure_position_hold()
        ctx.hardware.robot_wrapper = wrapper
        ctx.hardware.initial_position = None
        observation = wrapper.get_observation()
        ctx.hardware.initial_position = {key: observation[key] for key in robot.action_features}
        timing["last_prediction_seconds"] = None
        return goal

    ctx.runtime.reset_environment = reset_environment
    return ctx


if __name__ == "__main__":
    # Only this review entry point decorates the normal context; strategy, CLI and
    # action dispatch remain the shared lerobot-rollout implementation.
    lerobot_rollout.build_rollout_context = build_review_context
    lerobot_rollout.main()
