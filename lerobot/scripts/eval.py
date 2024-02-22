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
    fps: int = 15,
    env_step: int = None,
    wandb=None,
):
    if wandb is not None:
        assert env_step is not None
    sum_rewards = []
    max_rewards = []
    successes = []
    for i in range(num_episodes):
        ep_frames = []

        def rendering_callback(env, td=None):
            ep_frames.append(env.render())

        tensordict = env.reset()
        if save_video or wandb:
            # render first frame before rollout
            rendering_callback(env)

        with torch.inference_mode():
            rollout = env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=rendering_callback if save_video or wandb else None,
                auto_reset=False,
                tensordict=tensordict,
                auto_cast_to_device=True,
            )
        # print(", ".join([f"{x:.3f}" for x in rollout["next", "reward"][:,0].tolist()]))
        ep_sum_reward = rollout["next", "reward"].sum()
        ep_max_reward = rollout["next", "reward"].max()
        ep_success = rollout["next", "success"].any()
        sum_rewards.append(ep_sum_reward.item())
        max_rewards.append(ep_max_reward.item())
        successes.append(ep_success.item())

        if save_video or wandb:
            stacked_frames = np.stack(ep_frames)

            if save_video:
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"eval_episode_{i}.mp4"
                imageio.mimsave(video_path, stacked_frames, fps=fps)

            first_episode = i == 0
            if wandb and first_episode:
                eval_video = wandb.Video(
                    stacked_frames.transpose(0, 3, 1, 2), fps=fps, format="mp4"
                )
                wandb.log({"eval_video": eval_video}, step=env_step)

    metrics = {
        "avg_sum_reward": np.nanmean(sum_rewards),
        "avg_max_reward": np.nanmean(max_rewards),
        "pc_success": np.nanmean(successes) * 100,
    }
    return metrics


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def eval_cli(cfg: dict):
    eval(cfg, out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def eval(cfg: dict, out_dir=None):
    if out_dir is None:
        raise NotImplementedError()

    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Log dir:", "yellow", attrs=["bold"]), out_dir)

    env = make_env(cfg)

    if cfg.pretrained_model_path:
        policy = TDMPC(cfg)
        if "offline" in cfg.pretrained_model_path:
            policy.step = 25000
        elif "final" in cfg.pretrained_model_path:
            policy.step = 100000
        else:
            raise NotImplementedError()
        policy.load(cfg.pretrained_model_path)

        policy = TensorDictModule(
            policy,
            in_keys=["observation", "step_count"],
            out_keys=["action"],
        )
    else:
        # when policy is None, rollout a random policy
        policy = None

    metrics = eval_policy(
        env,
        policy=policy,
        save_video=True,
        video_dir=Path(out_dir) / "eval",
        fps=cfg.fps,
        max_steps=cfg.episode_length,
        num_episodes=cfg.eval_episodes,
    )
    print(metrics)


if __name__ == "__main__":
    eval_cli()
