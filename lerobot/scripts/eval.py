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
from torchrl.envs import EnvBase, SerialEnv
from torchrl.envs.batched_envs import BatchedEnvBase

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import init_logging, set_seed


def write_video(video_path, stacked_frames, fps):
    imageio.mimsave(video_path, stacked_frames, fps=fps)


def eval_policy(
    env: BatchedEnvBase,
    policy: TensorDictModule = None,
    num_episodes: int = 10,
    max_steps: int = 30,
    save_video: bool = False,
    video_dir: Path = None,
    fps: int = 15,
    return_first_video: bool = False,
):
    policy.eval()
    start = time.time()
    sum_rewards = []
    max_rewards = []
    successes = []
    threads = []  # for video saving threads
    episode_counter = 0  # for saving the correct number of videos

    # TODO(alexander-soare): if num_episodes is not evenly divisible by the batch size, this will do more work than
    # needed as I'm currently taking a ceil.
    for i in tqdm.tqdm(range(-(-num_episodes // env.batch_size[0]))):
        ep_frames = []

        def maybe_render_frame(env: EnvBase, _):
            if save_video or (return_first_video and i == 0):  # noqa: B023
                ep_frames.append(env.render())  # noqa: B023

        with torch.inference_mode():
            rollout = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_cast_to_device=True,
                callback=maybe_render_frame,
            )
        # print(", ".join([f"{x:.3f}" for x in rollout["next", "reward"][:,0].tolist()]))
        batch_sum_reward = rollout["next", "reward"].flatten(start_dim=1).sum(dim=-1)
        batch_max_reward = rollout["next", "reward"].flatten(start_dim=1).max(dim=-1)[0]
        batch_success = rollout["next", "success"].flatten(start_dim=1).any(dim=-1)
        sum_rewards.extend(batch_sum_reward.tolist())
        max_rewards.extend(batch_max_reward.tolist())
        successes.extend(batch_success.tolist())

        if save_video or (return_first_video and i == 0):
            batch_stacked_frames = np.stack(ep_frames)  # (t, b, *)
            batch_stacked_frames = batch_stacked_frames.transpose(
                1, 0, *range(2, batch_stacked_frames.ndim)
            )  # (b, t, *)

            if save_video:
                for stacked_frames in batch_stacked_frames:
                    if episode_counter >= num_episodes:
                        continue
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"eval_episode_{episode_counter}.mp4"
                    thread = threading.Thread(
                        target=write_video,
                        args=(str(video_path), stacked_frames, fps),
                    )
                    thread.start()
                    threads.append(thread)
                    episode_counter += 1

            if return_first_video and i == 0:
                first_video = batch_stacked_frames[0].transpose(0, 3, 1, 2)

    env.reset_rendering_hooks()

    for thread in threads:
        thread.join()

    info = {
        "avg_sum_reward": np.nanmean(sum_rewards[:num_episodes]),
        "avg_max_reward": np.nanmean(max_rewards[:num_episodes]),
        "pc_success": np.nanmean(successes[:num_episodes]) * 100,
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
    env = SerialEnv(
        cfg.rollout_batch_size,
        create_env_fn=make_env,
        create_env_kwargs=[
            {"cfg": cfg, "seed": s, "transform": offline_buffer.transform}
            for s in range(cfg.seed, cfg.seed + cfg.rollout_batch_size)
        ],
    )

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
        max_steps=cfg.env.episode_length,
        num_episodes=cfg.eval_episodes,
    )
    print(metrics)

    logging.info("End of eval")


if __name__ == "__main__":
    eval_cli()
