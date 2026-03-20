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
"""Goal State Generation via Oracle Policy.

Runs two environments side-by-side:
  - Left panel:  planning policy (e.g., act_simple_with_awm_head)
  - Right panel: oracle policy  (e.g., diffusion_pusht)

At each cycle:
  1. Capture current state from left (planning) env.
  2. Reset right (oracle) env to the same state.
  3. Roll out oracle for goal_state_interval steps (right panel animates).
  4. Roll out planning policy for n_action_steps steps (left panel animates).
  5. Repeat until the planning env terminates or max_steps is reached.

Outputs a side-by-side video.

Usage:
    python -m lerobot.scripts.lerobot_goal_state_generation \
        --oracle_policy_path lerobot/diffusion_pusht \
        --planning_policy_path lerobot/diffusion_pusht \
        --goal_state_interval 50 \
        --seed 42 \
        --output_dir outputs/goal_state_viz
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed


def _unwrap_to_base(vec_env):
    """Navigate through gymnasium wrappers to the base PushTEnv."""
    inner = vec_env.envs[0]
    base = inner
    chain = [type(base).__name__]
    while hasattr(base, "env"):
        base = base.env
        chain.append(type(base).__name__)
    return base, chain


def get_env_state(vec_env):
    """Extract the 5-element pushT state [agent_x, agent_y, block_x, block_y, block_angle]
    from a SyncVectorEnv with n_envs=1."""
    base, chain = _unwrap_to_base(vec_env)
    agent_pos = list(base.agent.position)
    block_pos = list(base.block.position)
    block_angle = float(base.block.angle)
    return np.array(agent_pos + block_pos + [block_angle], dtype=np.float64)


def render_frame(vec_env):
    """Render a single frame from a SyncVectorEnv with n_envs=1."""
    return vec_env.envs[0].render()


def load_policy_and_processors(policy_path, env_cfg, device):
    """Load a pretrained policy and its pre/post processors."""
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = Path(policy_path)
    cfg.device = device

    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": device},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
    )

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=cfg,
    )

    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def step_policy(policy, vec_env, observation, preprocessor, postprocessor,
                env_preprocessor, env_postprocessor):
    """Run one policy step: preprocess obs, select action, step env.
    Returns (new_observation, reward, terminated, truncated, info)."""
    obs = preprocess_observation(observation)
    obs = add_envs_task(vec_env, obs)
    obs = env_preprocessor(obs)
    obs = preprocessor(obs)

    with torch.inference_mode():
        action = policy.select_action(obs)
    action = postprocessor(action)

    action_transition = {"action": action}
    action_transition = env_postprocessor(action_transition)
    action = action_transition["action"]

    action_numpy = action.to("cpu").numpy()
    return vec_env.step(action_numpy)


def _reset_env_to_state(vec_env, state):
    """Reset a SyncVectorEnv(n_envs=1) and set the pushT state precisely.

    The built-in `reset(options={"reset_to_state": ...})` calls `space.step(dt)` after
    setting positions, which moves the dynamic block body due to collision resolution.
    Instead, we reset normally, then manually set positions and zero out velocities on
    the unwrapped env, and re-fetch the observation.
    """
    # Standard reset to initialize episode counters and physics space.
    vec_env.reset()

    base, _ = _unwrap_to_base(vec_env)

    # Set angle BEFORE position. In pymunk, setting angle on a body with a non-zero
    # center_of_gravity rotates around the CoG, which modifies body.position. By setting
    # angle first and position second, the position assignment is the final word.
    base.block.angle = float(state[4])
    base.block.position = list(state[2:4])
    base.agent.position = list(state[:2])

    # Zero out any velocity so the physics step below doesn't move the block.
    base.block.velocity = (0, 0)
    base.block.angular_velocity = 0

    # Run one physics step to update pymunk's internal shape spatial hash / world-space
    # vertices. Without this, render() may draw shapes at stale positions.
    # Because velocity is zeroed, this won't move the block.
    base.space.step(base.dt)

    # Re-fetch the observation from the env after our manual state override.
    observation = base.get_obs()

    # The VectorEnv expects batched observations; wrap in the same format as reset().
    # For SyncVectorEnv with n_envs=1, observations are dict of arrays with batch dim.
    batched_obs = {}
    for key, val in observation.items():
        batched_obs[key] = np.expand_dims(val, axis=0) if isinstance(val, np.ndarray) else val
    return batched_obs


def generate_goal_state(oracle_policy, oracle_env, current_state, goal_state_interval,
                        preprocessor, postprocessor, env_preprocessor, env_postprocessor):
    """Reset oracle env to current_state, roll out oracle for goal_state_interval steps.

    Returns:
        goal_observation: observation dict at the end of the oracle rollout.
        oracle_frames: list of rendered frames (H, W, C) uint8 arrays.
    """
    observation = _reset_env_to_state(oracle_env, current_state)
    oracle_policy.reset()

    frames = [render_frame(oracle_env)]
    steps_taken = 0
    while steps_taken < goal_state_interval:
        observation, reward, terminated, truncated, info = step_policy(
            oracle_policy, oracle_env, observation,
            preprocessor, postprocessor, env_preprocessor, env_postprocessor,
        )
        frames.append(render_frame(oracle_env))
        steps_taken += 1
        if terminated.any() or truncated.any():
            break

    return observation, frames


def run_side_by_side(
    oracle_policy_path: str,
    planning_policy_path: str,
    goal_state_interval: int,
    max_steps: int,
    seed: int,
    output_dir: str,
    device: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    env_cfg = PushtEnv()

    # ---- Create two separate environments ----
    planning_envs = make_env(env_cfg, n_envs=1)
    oracle_envs = make_env(env_cfg, n_envs=1)
    planning_env = planning_envs["pusht"][0]
    oracle_env = oracle_envs["pusht"][0]

    # ---- Load policies ----
    logging.info(f"Loading oracle policy from: {oracle_policy_path}")
    (oracle_policy, oracle_pre, oracle_post,
     oracle_env_pre, oracle_env_post) = load_policy_and_processors(
        oracle_policy_path, env_cfg, device,
    )

    logging.info(f"Loading planning policy from: {planning_policy_path}")
    (planning_policy, plan_pre, plan_post,
     plan_env_pre, plan_env_post) = load_policy_and_processors(
        planning_policy_path, env_cfg, device,
    )

    n_action_steps = planning_policy.config.n_action_steps

    # ---- Initial reset of planning env ----
    plan_obs, _info = planning_env.reset(seed=[seed])
    planning_policy.reset()

    # Collect all side-by-side frames.
    all_frames = []
    total_steps = 0
    cycle = 0
    done = False

    while total_steps < max_steps and not done:
        cycle += 1
        logging.info(f"--- Cycle {cycle} | total_steps={total_steps} ---")

        # 1. Capture current state from planning env.
        current_state = get_env_state(planning_env)
        logging.info(f"  Planning env state: {current_state}")

        # 2. Oracle rollout: reset oracle env to current state, roll out.
        goal_obs, oracle_frames = generate_goal_state(
            oracle_policy, oracle_env, current_state, goal_state_interval,
            oracle_pre, oracle_post, oracle_env_pre, oracle_env_post,
        )
        logging.info(f"  Oracle rolled out {len(oracle_frames)} frames")

        # Build side-by-side frames for the oracle phase.
        # Left panel frozen at current planning observation, right panel shows oracle.
        plan_frozen_frame = render_frame(planning_env)
        for oframe in oracle_frames:
            # Resize if needed to match heights.
            h = min(plan_frozen_frame.shape[0], oframe.shape[0])
            left = plan_frozen_frame[:h]
            right = oframe[:h]
            combined = np.concatenate([left, right], axis=1)
            all_frames.append(combined)

        # 3. Planning policy rollout for n_action_steps.
        oracle_final_frame = oracle_frames[-1]
        steps_this_cycle = 0
        while steps_this_cycle < n_action_steps and total_steps < max_steps:
            plan_obs, reward, terminated, truncated, info = step_policy(
                planning_policy, planning_env, plan_obs,
                plan_pre, plan_post, plan_env_pre, plan_env_post,
            )
            total_steps += 1
            steps_this_cycle += 1

            pframe = render_frame(planning_env)
            h = min(pframe.shape[0], oracle_final_frame.shape[0])
            left = pframe[:h]
            right = oracle_final_frame[:h]
            combined = np.concatenate([left, right], axis=1)
            all_frames.append(combined)

            if terminated.any() or truncated.any():
                done = True
                break

        logging.info(f"  Planning policy stepped {steps_this_cycle} steps")

    # ---- Write video ----
    stacked = np.stack(all_frames, axis=0)
    video_path = output_dir / "side_by_side.mp4"
    fps = env_cfg.fps
    write_video(str(video_path), stacked, fps)
    logging.info(f"Video saved to {video_path} ({len(all_frames)} frames, {fps} fps)")

    # ---- Cleanup ----
    planning_env.close()
    oracle_env.close()

    return str(video_path)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Goal state generation with oracle policy.")
    parser.add_argument("--oracle_policy_path", type=str, default="lerobot/diffusion_pusht",
                        help="HF Hub repo ID or local path for the oracle policy.")
    parser.add_argument("--planning_policy_path", type=str, default=None,
                        help="HF Hub repo ID or local path for the planning policy. "
                             "Defaults to oracle_policy_path if not set (useful for testing).")
    parser.add_argument("--goal_state_interval", type=int, default=50,
                        help="Number of oracle steps to roll out for each goal state.")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Maximum total steps for the planning env.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/goal_state_viz")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.planning_policy_path is None:
        args.planning_policy_path = args.oracle_policy_path

    run_side_by_side(
        oracle_policy_path=args.oracle_policy_path,
        planning_policy_path=args.planning_policy_path,
        goal_state_interval=args.goal_state_interval,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
