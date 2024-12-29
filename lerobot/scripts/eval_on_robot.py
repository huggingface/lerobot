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
"""Evaluate a policy by running rollouts on the real robot and computing metrics.

Usage examples: evaluate a checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval_on_robot.py \
    -p outputs/train/model/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

**NOTE** (michel-aractingi): This script is incomplete and it is being prepared
for running training on the real robot.
"""

import argparse
import logging
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import trange

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless
from lerobot.common.robot_devices.robots.factory import Robot, make_robot
from lerobot.common.utils.utils import (
    init_hydra_config,
    init_logging,
    log_say,
)


def rollout(robot: Robot, policy: Policy, fps: int, control_time_s: float = 20, use_amp: bool = True) -> dict:
    """Run a batched policy rollout on the real robot.

    The return dictionary contains:
        "robot": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        robot: The robot class that defines the interface with the real robot.
        policy: The policy. Must be a PyTorch nn module.

    Returns:
        The dictionary described above.
    """
    # assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    # device = get_device_from_parameters(policy)

    # define keyboard listener
    listener, events = init_keyboard_listener()

    # Reset the policy. TODO (michel-aractingi) add real policy evaluation once the code is ready.
    # policy.reset()

    # Get observation from real robot
    observation = robot.capture_observation()

    # Calculate reward. TODO (michel-aractingi)
    # in HIL-SERL it will be with a reward classifier
    reward = calculate_reward(observation)
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []

    start_episode_t = time.perf_counter()
    timestamp = 0.0
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        all_observations.append(deepcopy(observation))
        # observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

        # Apply the next action.
        while events["pause_policy"] and not events["human_intervention_step"]:
            busy_wait(0.5)

        if events["human_intervention_step"]:
            # take over the robot's actions
            observation, action = robot.teleop_step(record_data=True)
            action = action["action"]  # teleop step returns torch tensors but in a dict
        else:
            # explore with policy
            with torch.inference_mode():
                action = robot.follower_arms["main"].read("Present_Position")
                action = torch.from_numpy(action)
                robot.send_action(action)
                # action = predict_action(observation, policy, device, use_amp)

        observation = robot.capture_observation()
        # Calculate reward
        # in HIL-SERL it will be with a reward classifier
        reward = calculate_reward(observation)

        all_actions.append(action)
        all_rewards.append(torch.from_numpy(reward))
        all_successes.append(torch.tensor([False]))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            events["human_intervention_step"] = False
            events["pause_policy"] = False
            break
    all_observations.append(deepcopy(observation))

    dones = torch.tensor([False] * len(all_actions))
    dones[-1] = True
    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "next.reward": torch.stack(all_rewards, dim=1),
        "next.success": torch.stack(all_successes, dim=1),
        "done": dones,
    }
    stacked_observations = {}
    for key in all_observations[0]:
        stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
    ret["observation"] = stacked_observations

    listener.stop()

    return ret


def eval_policy(
    robot: Robot,
    policy: torch.nn.Module,
    fps: float,
    n_episodes: int,
    control_time_s: int = 20,
    use_amp: bool = True,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    # TODO (michel-aractingi) comment this out for testing with a fixed policy
    # assert isinstance(policy, Policy)
    # policy.eval()

    sum_rewards = []
    max_rewards = []
    successes = []
    rollouts = []

    start_eval = time.perf_counter()
    progbar = trange(n_episodes, desc="Evaluating policy on real robot")
    for _batch_idx in progbar:
        rollout_data = rollout(robot, policy, fps, control_time_s, use_amp)

        rollouts.append(rollout_data)
        sum_rewards.append(sum(rollout_data["next.reward"]))
        max_rewards.append(max(rollout_data["next.reward"]))
        successes.append(rollout_data["next.success"][-1])

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "pc_success": success * 100,
            }
            for i, (sum_reward, max_reward, success) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    successes[:n_episodes],
                    strict=False,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(torch.cat(sum_rewards[:n_episodes]))),
            "avg_max_reward": float(np.nanmean(torch.cat(max_rewards[:n_episodes]))),
            "pc_success": float(np.nanmean(torch.cat(successes[:n_episodes])) * 100),
            "eval_s": time.time() - start_eval,
            "eval_ep_s": (time.time() - start_eval) / n_episodes,
        },
    }

    if robot.is_connected:
        robot.disconnect()

    return info


def calculate_reward(observation):
    """
    Method to calculate reward function in some way.
    In HIL-SERL this is done through defining a reward classifier
    """
    # reward = reward_classifier(observation)
    return np.array([0.0])


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["pause_policy"] = False
    events["human_intervention_step"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                # check if first space press then pause the policy for the user to get ready
                # if second space press then the user is ready to start intervention
                if not events["pause_policy"]:
                    print(
                        "Space key pressed. Human intervention required.\n"
                        "Place the leader in similar pose to the follower and press space again."
                    )
                    events["pause_policy"] = True
                    log_say("Human intervention stage. Get ready to take over.", play_sounds=True)
                else:
                    events["human_intervention_step"] = True
                    print("Space key pressed. Human intervention starting.")
                    log_say("Starting human intervention.", play_sounds=True)

        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    group.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )

    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)
    if not robot.is_connected:
        robot.connect()

    eval_policy(robot, None, fps=40, n_episodes=2, control_time_s=100)
