#!/usr/bin/env python

"""
Run a trained HIL-SERL policy on the real robot.

Mirrors the single-loop, reset-in-place pattern from `actor.py` /
`gym_manipulator.control_loop` so multi-episode runs reset cleanly.

Usage:
    python -m lerobot.rl.inference --config_path .../pretrained_model/train_config.json
"""

import logging
import sys
import time
from pathlib import Path

import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import so_follower  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401  # registers UR10RobotConfig
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)


def _resolve_pretrained_dir() -> str:
    """Extract the pretrained_model directory from --config_path CLI arg."""
    for i, arg in enumerate(sys.argv):
        if "config_path" not in arg:
            continue
        if "=" in arg:
            config_path = Path(arg.split("=", 1)[1])
        elif i + 1 < len(sys.argv):
            config_path = Path(sys.argv[i + 1])
        else:
            break
        pretrained_dir = config_path.parent
        if pretrained_dir.exists() and (pretrained_dir / "model.safetensors").exists():
            return str(pretrained_dir)
    raise ValueError(
        "Could not determine pretrained model directory. "
        "Pass --config_path .../pretrained_model/train_config.json"
    )


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    pretrained_dir = _resolve_pretrained_dir()
    logging.info(f"Loading pretrained model from: {pretrained_dir}")
    cfg.policy.pretrained_path = pretrained_dir

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, cfg.policy.device
    )

    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    logging.info(f"Policy loaded on {cfg.policy.device} ({type(policy).__name__})")

    fps = cfg.env.fps or 10
    dt = 1.0 / fps
    max_episodes = 100

    # Initial reset — same pattern as actor.act_with_policy and control_loop.
    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    episode_idx = 0
    episode_step = 0
    episode_reward = 0.0
    episode_start_time = time.perf_counter()

    logging.info(f"Inference at {fps} Hz. Triangle = success, Cross = fail, Ctrl+C = exit.")
    logging.info(f"--- Episode {episode_idx + 1} ---")

    try:
        while episode_idx < max_episodes:
            t0 = time.perf_counter()

            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            with torch.no_grad():
                action = policy.select_action(batch=observation)

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            reward = float(transition[TransitionKey.REWARD])
            done = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            episode_reward += reward
            episode_step += 1

            if episode_step % 10 == 0:
                logging.info(f"  step {episode_step}, reward={episode_reward:.2f}")

            if done or truncated:
                ep_time = time.perf_counter() - episode_start_time
                status = "SUCCESS" if episode_reward > 0 else "DONE"
                logging.info(
                    f"Episode {episode_idx + 1} {status}: "
                    f"reward={episode_reward:.2f} steps={episode_step} "
                    f"time={ep_time:.1f}s done={done} truncated={truncated}"
                )
                episode_idx += 1
                if episode_idx >= max_episodes:
                    break

                # Reset for next episode — drives the arm home, re-anchors tcp_xyz baseline,
                # zeroes TimeLimit step counter, etc. Same call sequence as actor.py:432-438.
                obs, info = env.reset()
                env_processor.reset()
                action_processor.reset()
                policy.reset()  # no-op for SAC; defensive for action-chunking policies.
                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)

                episode_step = 0
                episode_reward = 0.0
                episode_start_time = time.perf_counter()
                logging.info(f"--- Episode {episode_idx + 1} ---")

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        logging.info("Inference stopped by user.")
    except Exception:
        logging.exception("Inference failed")
    finally:
        logging.info(f"Completed {episode_idx} episodes")
        try:
            env.close()
        except Exception:
            logging.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logging.exception("teleop disconnect failed")


if __name__ == "__main__":
    main()
