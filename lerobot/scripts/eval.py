from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from ..lib.envs import make_env
from ..lib.utils import set_seed


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
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        ep_success = False
        if save_video:
            frames = []
        while not done:
            action = agent.act(obs, t0=t == 0, eval_mode=True, step=step)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if "success" in info and info["success"]:
                ep_success = True
            if save_video:
                frame = env.render(
                    mode="rgb_array",
                    # TODO(rcadene): make height, width, camera_id configurable
                    height=384,
                    width=384,
                    camera_id=0,
                )
                frames.append(frame)
            t += 1
        episode_rewards.append(float(ep_reward))
        episode_successes.append(float(ep_success))
        episode_lengths.append(t)
        if save_video:
            frames = np.stack(frames).transpose(0, 3, 1, 2)
            video_path.parent.mkdir(parents=True, exist_ok=True)
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

    eval_metrics = eval_agent(env, agent, num_episodes=10, save_video=True)


if __name__ == "__main__":
    eval()
