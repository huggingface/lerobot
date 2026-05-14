#!/usr/bin/env python
"""Interactive demo for hot-prompt switching — no robot or GPU required.

Runs a fake control loop using a mock robot and a dummy policy.
Type a new task in the terminal and press Enter to switch the prompt live.

Usage
-----
    conda run -n lerobot_rollout python examples/hot_prompt_demo.py
    conda run -n lerobot_rollout python examples/hot_prompt_demo.py --fps=5
    conda run -n lerobot_rollout python examples/hot_prompt_demo.py --task="pick up cube" --fps=2

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from threading import Event
from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# Minimal logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hot_prompt_demo")


# ---------------------------------------------------------------------------
# Fake policy: returns random actions, prints the task it receives
# ---------------------------------------------------------------------------

class _EchoPolicy:
    """Dummy policy that logs the task it receives and returns a random action."""

    type = "echo"

    class config:
        use_amp = False
        action_feature_names = ["motor_1.pos", "motor_2.pos", "motor_3.pos"]

    def reset(self):
        pass

    def select_action(self, observation: dict) -> torch.Tensor:
        task = observation.get("task", ["<no task>"])
        task_str = task[0] if isinstance(task, list) else task
        logger.info("  [policy] received task: '%s'  →  sending random action", task_str)
        return torch.rand(1, 3) * 2 - 1   # shape [1, 3], range [-1, 1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="pick up cube", help="Initial task string")
    parser.add_argument("--fps", type=float, default=3.0, help="Control loop frequency (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="Run for N seconds (0 = infinite)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Broker + listener
    # ------------------------------------------------------------------
    from lerobot.rollout.prompt_broker import PromptBroker, StdinPromptListener

    shutdown_event = Event()
    broker = PromptBroker(initial_task=args.task)
    StdinPromptListener().start(broker, shutdown_event)

    # ------------------------------------------------------------------
    # 2. Inference engine (SyncInferenceEngine with a fake policy)
    # ------------------------------------------------------------------
    from lerobot.rollout import SyncInferenceConfig, create_inference_engine

    # dataset_features and ordered_action_keys must describe the action space
    action_keys = ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    dataset_features = {
        k: {"dtype": "float32", "shape": (1,), "names": None} for k in action_keys
    }

    fake_policy = _EchoPolicy()
    fake_robot_wrapper = MagicMock()
    fake_robot_wrapper.robot_type = "mock_robot"

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=fake_policy,
        preprocessor=MagicMock(side_effect=lambda x: x),   # identity
        postprocessor=MagicMock(side_effect=lambda x: x),  # identity
        robot_wrapper=fake_robot_wrapper,
        hw_features={},
        dataset_features=dataset_features,
        ordered_action_keys=action_keys,
        task=args.task,
        fps=args.fps,
        device="cpu",
        prompt_broker=broker,
    )

    # Patch make_robot_action so we don't need real dataset feature parsing
    import lerobot.rollout.inference.sync as _sync_mod
    _orig_make_robot_action = _sync_mod.make_robot_action

    def _fake_make_robot_action(tensor, features):
        vals = tensor.tolist()
        if not isinstance(vals, list):
            vals = [vals]
        return {k: v for k, v in zip(action_keys, vals)}

    _sync_mod.make_robot_action = _fake_make_robot_action

    # Patch prepare_observation_for_inference to inject the task into obs
    _orig_prepare = _sync_mod.prepare_observation_for_inference

    def _fake_prepare(obs, device, task, robot_type):
        obs["task"] = [task]
        return obs

    _sync_mod.prepare_observation_for_inference = _fake_prepare

    # ------------------------------------------------------------------
    # 3. Control loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Hot-prompt demo started")
    logger.info("  Initial task : '%s'", args.task)
    logger.info("  FPS          : %.0f", args.fps)
    logger.info("  Duration     : %s", f"{args.duration}s" if args.duration > 0 else "infinite (Ctrl-C to stop)")
    logger.info("=" * 60)
    logger.info("Type a new task below and press Enter to switch:")

    control_interval = 1.0 / args.fps
    start = time.perf_counter()
    tick = 0

    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()

            if args.duration > 0 and (loop_start - start) >= args.duration:
                logger.info("Duration limit reached (%.0fs)", args.duration)
                break

            # Fake observation
            obs = {"motor_1.pos": 0.0, "motor_2.pos": 0.0, "motor_3.pos": 0.0}
            action = engine.get_action(obs)
            tick += 1

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        shutdown_event.set()
        # Restore patched functions
        _sync_mod.make_robot_action = _orig_make_robot_action
        _sync_mod.prepare_observation_for_inference = _orig_prepare

    logger.info("Demo finished after %d ticks", tick)


if __name__ == "__main__":
    main()
