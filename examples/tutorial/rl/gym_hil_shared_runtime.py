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

"""Smoke-test gym-hil through LeRobot's shared scalar interaction runtime.

Install the optional dependencies once:

    uv sync --locked --extra hilserl

Run a short headless test:

    uv run python examples/tutorial/rl/gym_hil_shared_runtime.py --steps 25

Open the MuJoCo viewer and let random actions move the arm:

    uv run python examples/tutorial/rl/gym_hil_shared_runtime.py \
        --viewer --action random --steps 100

Test keyboard intervention while the automatic provider holds still:

    uv run python examples/tutorial/rl/gym_hil_shared_runtime.py \
        --control keyboard --action neutral --steps 1000

This intentionally uses a tiny action provider instead of a trained policy. It tests
the simulator boundary, canonical feature mapping, shared episode loop, outcomes and
requested-versus-applied action recording without needing a checkpoint.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.interaction import FeatureMapGymAdapter, GymEndpoint, ListRecorder, run_episode
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

GYM_HIL_ACTION_NAMES = (
    "ee.delta_x",
    "ee.delta_y",
    "ee.delta_z",
    "gripper.command",
)

GYM_HIL_TASKS = {
    "pick_cube": "gym_hil/PandaPickCube-v0",
    "arrange_boxes": "gym_hil/PandaArrangeBoxes-v0",
}


class GymHilOutcomeWrapper(gym.Wrapper):
    """Expose gym-hil's native and manually supplied success under one key."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        native_success = bool(info.get("succeed", False))
        manual_success = bool(info.get("next.success", False))
        info["is_success"] = native_success or manual_success
        return observation, reward, terminated, truncated, info


class RealtimeWrapper(gym.Wrapper):
    """Pace an interactive simulator so keyboard input can be observed."""

    def __init__(self, env: gym.Env, *, fps: float) -> None:
        super().__init__(env)
        self.period = 1.0 / fps

    def step(self, action):
        started_at = time.perf_counter()
        transition = self.env.step(action)
        time.sleep(max(0.0, self.period - (time.perf_counter() - started_at)))
        return transition


class SmokeActionProvider:
    """Produce deterministic neutral or random actions for boundary testing."""

    action_names = GYM_HIL_ACTION_NAMES

    def __init__(self, action_space: gym.spaces.Box, *, mode: str, seed: int) -> None:
        if action_space.shape != (len(self.action_names),):
            raise ValueError(
                "This example expects gym-hil's wrapped 4D end-effector action space, "
                f"but received shape {action_space.shape}."
            )
        self.action_space = action_space
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Keep the random stream deterministic across the full run."""

    def get_action(self, observation: dict[str, Any] | None) -> torch.Tensor:
        if observation is None or OBS_STATE not in observation:
            raise ValueError(f"Expected a canonical {OBS_STATE!r} observation")

        if self.mode == "neutral":
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            action[-1] = 1.0  # gym-hil uses 1.0 as the gripper hold command.
        else:
            action = self.rng.uniform(self.action_space.low, self.action_space.high).astype(np.float32)
        return torch.from_numpy(action)


def gym_hil_applied_action(requested_action: Any, info: dict[str, Any]) -> torch.Tensor:
    """Record a human override when gym-hil reports one."""
    applied_action = info.get("teleop_action", requested_action)
    return torch.as_tensor(applied_action, dtype=torch.float32)


def make_endpoint(args: argparse.Namespace) -> GymEndpoint:
    """Create one gym-hil task behind LeRobot's canonical endpoint contract."""
    require_package("gym-hil", extra="hilserl", import_name="gym_hil")
    import gym_hil  # noqa: F401

    uses_human_control = args.control != "none"
    env = gym.make(
        GYM_HIL_TASKS[args.task],
        render_mode="rgb_array",
        image_obs=args.image_obs,
        reward_type=args.reward,
        use_viewer=args.viewer or uses_human_control,
        use_inputs_control=uses_human_control,
        use_gamepad=args.control == "gamepad",
        auto_reset=False,
        reset_delay_seconds=0.0,
        disable_env_checker=True,
    )
    env = GymHilOutcomeWrapper(env)
    if args.viewer or uses_human_control:
        env = RealtimeWrapper(env, fps=args.fps)

    features_map = {"agent_pos": OBS_STATE}
    if args.image_obs:
        features_map.update(
            {
                "pixels/front": f"{OBS_IMAGES}.front",
                "pixels/wrist": f"{OBS_IMAGES}.wrist",
            }
        )
    else:
        features_map["environment_state"] = OBS_ENV_STATE

    adapter = FeatureMapGymAdapter(
        features_map,
        to_canonical_applied_action=gym_hil_applied_action,
    )
    return GymEndpoint(
        env,
        adapter=adapter,
        action_names=GYM_HIL_ACTION_NAMES,
        success_key="is_success",
    )


def describe_observation(observation: dict[str, Any]) -> str:
    """Format canonical observation keys and shapes for the smoke-test output."""
    parts = []
    for key, value in observation.items():
        shape = getattr(value, "shape", None)
        parts.append(f"{key}={tuple(shape) if shape is not None else type(value).__name__}")
    return ", ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=GYM_HIL_TASKS, default="pick_cube")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action", choices=("neutral", "random"), default="random")
    parser.add_argument("--reward", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--image-obs", action="store_true", help="Include front and wrist images.")
    parser.add_argument("--viewer", action="store_true", help="Open gym-hil's MuJoCo viewer.")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Wall-clock rate when the viewer or human control is enabled.",
    )
    parser.add_argument(
        "--control",
        choices=("none", "keyboard", "gamepad"),
        default="none",
        help="Allow human actions to replace provider actions; this also opens the viewer.",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    endpoint = make_endpoint(args)
    if not isinstance(endpoint.env.action_space, gym.spaces.Box):
        endpoint.close()
        raise TypeError(f"Expected a Box action space, got {type(endpoint.env.action_space).__name__}")

    action_provider = SmokeActionProvider(endpoint.env.action_space, mode=args.action, seed=args.seed)
    recorder = ListRecorder()

    print(f"Environment: {GYM_HIL_TASKS[args.task]}")
    print(f"Canonical action order: {GYM_HIL_ACTION_NAMES}")
    if args.control != "none":
        print(f"Human control: {args.control}; overrides will be stored as applied_action")
    if args.control == "keyboard":
        print(
            "Keyboard: Space toggles intervention; arrows move X/Y; "
            "Left/Right Shift move -Z/+Z; Left/Right Ctrl close/open the gripper; "
            "Enter ends with success; Escape ends with failure; Ctrl-C stops the script."
        )

    try:
        for episode_index in range(args.episodes):
            first_record_index = len(recorder.records)
            result = run_episode(
                endpoint=endpoint,
                action_provider=action_provider,
                recorder=recorder,
                max_steps=args.steps,
                seed=args.seed + episode_index,
                episode_id=f"gym-hil-{episode_index:04d}",
                actor_id="gym-hil-smoke-test",
            )
            episode_records = recorder.records[first_record_index:]
            interventions = sum(bool(record.info.get("is_intervention", False)) for record in episode_records)
            overrides = sum(
                not torch.equal(record.action, record.applied_action) for record in episode_records
            )
            print(
                f"Episode {episode_index + 1}: steps={result.num_steps}, "
                f"reward={result.total_reward}, success={result.success}, "
                f"terminated={result.terminated}, truncated={result.truncated}, "
                f"interventions={interventions}, overrides={overrides}"
            )
            if episode_records:
                print(f"Canonical observation: {describe_observation(episode_records[0].observation)}")
    finally:
        endpoint.close()


if __name__ == "__main__":
    main()
