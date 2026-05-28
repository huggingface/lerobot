"""Replay recorded v5 success-ep actions in eval env to verify env↔data alignment.

If recorded actions reproduce success (done=True), env config matches data
collection. If they don't, the env scene/IK params drifted vs recording.

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run --no-sync python \
        scripts_local/replay_v5_actions_in_eval_env.py \
        --config_path=src/lerobot/rl/sim_3stage_act_v5_eval_env.json \
        --dataset domrachev03/sim_3stage_v5_success --n-eps 3
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401


def parse_aux() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-eps", type=int, default=3)
    aux, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return aux


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    aux = main._aux
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg.env.teleop = None

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.rl.gym_manipulator import (
        create_transition,
        make_processors,
        make_robot_env,
        step_env_and_process_transition,
    )

    ds = LeRobotDataset(repo_id=aux.dataset)
    env, _ = make_robot_env(cfg.env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_proc, action_proc = make_processors(env, teleop_device=None, cfg=cfg.env, device=device)

    for ep_idx in range(aux.n_eps):
        ep_from = int(ds.meta.episodes["dataset_from_index"][ep_idx])
        ep_to = int(ds.meta.episodes["dataset_to_index"][ep_idx])
        ep_len = ep_to - ep_from
        actions_t = torch.stack([ds[ep_from + i]["action"] for i in range(ep_len)])  # (T, 5)

        obs, info = env.reset()
        env_proc.reset()
        action_proc.reset()
        comp = {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
        transition = create_transition(observation=obs, info=info, complementary_data=comp)
        transition = env_proc(data=transition)

        ep_r = 0.0
        max_r = 0.0
        succ = False
        for t in range(ep_len):
            a = actions_t[t].to(device).unsqueeze(0)
            transition = step_env_and_process_transition(
                env=env, transition=transition, action=a,
                env_processor=env_proc, action_processor=action_proc,
            )
            r = float(transition[TransitionKey.REWARD])
            ep_r += r
            max_r = max(max_r, r)
            if transition[TransitionKey.DONE]:
                succ = not transition[TransitionKey.TRUNCATED]
                break
        logging.info("ep %d: T=%d done@%s succ=%s sum_r=%.3f max_r=%.3f",
                     ep_idx, ep_len, t + 1, succ, ep_r, max_r)


if __name__ == "__main__":
    main._aux = parse_aux()
    main()
