from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from termcolor import colored
from torchrl.envs import EnvBase

from lerobot.common.envs.factory import make_env
from lerobot.common.tdmpc import TDMPC
from lerobot.common.utils import set_seed


def eval_policy(
    env: EnvBase,
    policy: TensorDictModule = None,
    num_episodes: int = 10,
    max_steps: int = 30,
    save_video: bool = False,
    video_dir: Path = None,
):
    rewards = []
    successes = []
    for i in range(num_episodes):
        ep_frames = []

        def rendering_callback(env, td=None):
            nonlocal ep_frames
            frame = env.render()
            ep_frames.append(frame)

        tensordict = env.reset()
        if save_video:
            # render first frame before rollout
            rendering_callback(env)

        rollout = env.rollout(
            max_steps=max_steps,
            policy=policy,
            callback=rendering_callback if save_video else None,
            auto_reset=False,
            tensordict=tensordict,
        )
        # print(", ".join([f"{x:.3f}" for x in rollout["next", "reward"][:,0].tolist()]))
        ep_reward = rollout["next", "reward"].sum()
        ep_success = rollout["next", "success"].any()
        rewards.append(ep_reward.item())
        successes.append(ep_success.item())

        if save_video:
            video_dir.mkdir(parents=True, exist_ok=True)
            # TODO(rcadene): make fps configurable
            video_path = video_dir / f"eval_episode_{i}.mp4"
            imageio.mimsave(video_path, np.stack(ep_frames), fps=15)

    metrics = {
        "avg_reward": np.nanmean(rewards),
        "pc_success": np.nanmean(successes) * 100,
    }
    return metrics


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def eval(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Log dir:", "yellow", attrs=["bold"]), cfg.log_dir)

    env = make_env(cfg)
    policy = TDMPC(cfg)
    # ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/offline.pt"
    ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt"
    policy.load(ckpt_path)

    policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    # policy can be None to rollout a random policy
    metrics = eval_policy(
        env,
        policy=policy,
        num_episodes=20,
        save_video=False,
        video_dir=Path("tmp/2023_01_29_xarm_lift_final"),
    )
    print(metrics)


if __name__ == "__main__":
    eval()
