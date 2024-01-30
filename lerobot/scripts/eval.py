from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from tensordict import TensorDict
from termcolor import colored

from lerobot.lib.envs.factory import make_env
from lerobot.lib.tdmpc import TDMPC
from lerobot.lib.utils import set_seed


def eval_agent(
    env, agent, num_episodes: int, save_video: bool = False, video_path: Path = None
):
    """Evaluate a trained agent and optionally save a video."""
    if save_video:
        assert video_path is not None
        assert video_path.suffix == ".mp4"
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    for i in range(num_episodes):
        td = env.reset()
        obs = {}
        obs["rgb"] = td["observation"]["camera"]
        obs["state"] = td["observation"]["robot_state"]

        done = False
        ep_reward = 0
        t = 0
        ep_success = False

        if save_video:
            frames = []
        while not done:
            action = agent.act(obs, t0=t == 0, eval_mode=True, step=100000)
            td = TensorDict({"action": action}, batch_size=[])

            td = env.step(td)

            reward = td["next", "reward"].item()
            success = td["next", "success"].item()
            done = td["next", "done"].item()

            obs = {}
            obs["rgb"] = td["next", "observation"]["camera"]
            obs["state"] = td["next", "observation"]["robot_state"]

            ep_reward += reward
            if success:
                ep_success = True
            if save_video:
                frame = env.render()
                frames.append(frame)
            t += 1
        episode_rewards.append(float(ep_reward))
        episode_successes.append(float(ep_success))
        episode_lengths.append(t)
        if save_video:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            frames = np.stack(frames)  # .transpose(0, 3, 1, 2)
            # TODO(rcadene): make fps configurable
            imageio.mimsave(video_path, frames, fps=15)
    return {
        "episode_reward": np.nanmean(episode_rewards),
        "episode_success": np.nanmean(episode_successes),
        "episode_length": np.nanmean(episode_lengths),
    }


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def eval(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Log dir:", "yellow", attrs=["bold"]), cfg.log_dir)

    env = make_env(cfg)
    agent = TDMPC(cfg)
    # ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/offline.pt"
    ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt"
    agent.load(ckpt_path)

    eval_metrics = eval_agent(
        env,
        agent,
        num_episodes=10,
        save_video=True,
        video_path=Path("tmp/2023_01_29_xarm_lift_final/eval.mp4"),
    )

    print(eval_metrics)


if __name__ == "__main__":
    eval()
