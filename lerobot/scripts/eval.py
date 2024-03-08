import logging
import threading
import time
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import init_logging, set_seed


def write_video(video_path, stacked_frames, fps):
    imageio.mimsave(video_path, stacked_frames, fps=fps)


def eval_policy(
    env: EnvBase,
    policy: TensorDictModule = None,
    num_episodes: int = 10,
    max_steps: int = 30,
    save_video: bool = False,
    video_dir: Path = None,
    fps: int = 15,
    return_first_video: bool = False,
):
    start = time.time()
    sum_rewards = []
    max_rewards = []
    successes = []
    threads = []
    for i in tqdm.tqdm(range(num_episodes)):
        ep_frames = []
        if save_video or (return_first_video and i == 0):

            def render_frame(env):
                ep_frames.append(env.render())  # noqa: B023

            env.register_rendering_hook(render_frame)

        with torch.inference_mode():
            rollout = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_cast_to_device=True,
            )
        # print(", ".join([f"{x:.3f}" for x in rollout["next", "reward"][:,0].tolist()]))
        ep_sum_reward = rollout["next", "reward"].sum()
        ep_max_reward = rollout["next", "reward"].max()
        ep_success = rollout["next", "success"].any()
        sum_rewards.append(ep_sum_reward.item())
        max_rewards.append(ep_max_reward.item())
        successes.append(ep_success.item())

        if save_video or (return_first_video and i == 0):
            stacked_frames = np.stack(ep_frames)

            if save_video:
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"eval_episode_{i}.mp4"
                thread = threading.Thread(
                    target=write_video,
                    args=(str(video_path), stacked_frames, fps),
                )
                thread.start()
                threads.append(thread)

            if return_first_video and i == 0:
                first_video = stacked_frames.transpose(0, 3, 1, 2)

    env.reset_rendering_hooks()

    for thread in threads:
        thread.join()

    info = {
        "avg_sum_reward": np.nanmean(sum_rewards),
        "avg_max_reward": np.nanmean(max_rewards),
        "pc_success": np.nanmean(successes) * 100,
        "eval_s": time.time() - start,
        "eval_ep_s": (time.time() - start) / num_episodes,
    }
    if return_first_video:
        return info, first_video
    return info


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def eval_cli(cfg: dict):
    eval(cfg, out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def eval(cfg: dict, out_dir=None):
    if out_dir is None:
        raise NotImplementedError()

    init_logging()

    if cfg.device == "cuda":
        assert torch.cuda.is_available()
    else:
        logging.warning("Using CPU, this will be slow.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    log_output_dir(out_dir)

    logging.info("make_offline_buffer")
    offline_buffer = make_offline_buffer(cfg)

    logging.info("make_env")
    env = make_env(cfg, transform=offline_buffer.transform)

    if cfg.policy.pretrained_model_path:
        policy = make_policy(cfg)
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
        fps=cfg.env.fps,
        max_steps=cfg.env.episode_length // cfg.n_action_steps,
        num_episodes=cfg.eval_episodes,
    )
    print(metrics)

    logging.info("End of eval")


if __name__ == "__main__":
    eval_cli()
