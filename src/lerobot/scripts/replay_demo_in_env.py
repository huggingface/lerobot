"""Diagnostic: replay a demo episode's recorded actions in the eval env.

Confirms whether the eval env (gym_manipulator + SARM threshold) actually fires
success on a known-good action sequence. If replay fails to reach SARM=0.98,
the eval env differs from the recording env — policy will never succeed
regardless of architecture.

Usage:
    uv run python -m lerobot.scripts.replay_demo_in_env \\
        --config_path=src/lerobot/rl/sim_assembling_sarm_hilserl_rabc_v6_train.json \\
        --demo-repo=local/sim_assemble_actdp_combined_cont \\
        --episode-index=0
"""

import argparse
import logging
import sys

import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401  # registry
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401


def parse_aux_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--demo-repo", required=True)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=3, help="how many demo eps to replay")
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    aux = main._aux
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.rl.gym_manipulator import (
        create_transition,
        make_processors,
        make_robot_env,
        step_env_and_process_transition,
    )

    cfg.env.teleop = None
    env, _ = make_robot_env(cfg.env)
    env_proc, action_proc = make_processors(env, teleop_device=None, cfg=cfg.env, device="cuda")

    ds = LeRobotDataset(aux.demo_repo)
    n_eps = ds.meta.total_episodes
    eps_to_run = list(range(aux.episode_index, min(aux.episode_index + aux.n_episodes, n_eps)))
    logging.info("Replaying eps %s from %s (total %d eps)", eps_to_run, aux.demo_repo, n_eps)

    for ep_idx in eps_to_run:
        ep_meta = ds.meta.episodes[ep_idx]
        # iterate frames belonging to this episode
        from_idx = ep_meta["dataset_from_index"] if "dataset_from_index" in ep_meta else None
        to_idx = ep_meta["dataset_to_index"] if "dataset_to_index" in ep_meta else None
        if from_idx is None:
            ep_len = ep_meta.get("length", None)
            if ep_len is None:
                raise RuntimeError("cannot resolve ep length")
            from_idx = sum(ds.meta.episodes[i]["length"] for i in range(ep_idx))
            to_idx = from_idx + ep_len

        actions = []
        for fi in range(from_idx, to_idx):
            actions.append(ds[fi]["action"])

        obs, info = env.reset()
        env_proc.reset()
        action_proc.reset()
        complementary_data = {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_proc(data=transition)

        ep_reward = 0.0
        success = False
        steps_run = 0
        max_step_reward = 0.0
        for a in actions:
            a_t = a.float() if isinstance(a, torch.Tensor) else torch.as_tensor(a).float()
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=a_t,
                env_processor=env_proc,
                action_processor=action_proc,
            )
            r = float(transition[TransitionKey.REWARD])
            ep_reward += r
            max_step_reward = max(max_step_reward, r)
            steps_run += 1
            if transition[TransitionKey.DONE] and not transition[TransitionKey.TRUNCATED]:
                success = True
                break
            if transition[TransitionKey.TRUNCATED]:
                break
        logging.info(
            "ep_idx=%d demo_len=%d steps_run=%d success=%s ep_reward=%.3f max_step_r=%.3f",
            ep_idx, len(actions), steps_run, success, ep_reward, max_step_reward,
        )


if __name__ == "__main__":
    main._aux = parse_aux_args()
    main()
