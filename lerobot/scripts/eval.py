"""Evaluate a policy on an environment by running rollouts and computing metrics.

The script may be run in one of two ways:

1. By providing the path to a config file with the --config argument.
2. By providing a HuggingFace Hub ID with the --hub-id argument. You may also provide a revision number with the
    --revision argument.

In either case, it is possible to override config arguments by adding a list of config.key=value arguments.

Examples:

You have a specific config file to go with trained model weights, and want to run 10 episodes.

```
python lerobot/scripts/eval.py --config PATH/TO/FOLDER/config.yaml \
    policy.pretrained_model_path=PATH/TO/FOLDER/weights.pth` eval_episodes=10
```

You have a HuggingFace Hub ID, you know which revision you want, and want to run 10 episodes (note that in this case,
you don't need to specify which weights to use):

```
python lerobot/scripts/eval.py --hub-id HUB/ID --revision v1.0 eval_episodes=10
```
"""

import argparse
import json
import logging
import os.path as osp
import threading
import time
from datetime import datetime as dt
from pathlib import Path

import einops
import hydra
import imageio
import numpy as np
import torch
import tqdm
from huggingface_hub import snapshot_download
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from torchrl.envs.batched_envs import BatchedEnvBase

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.abstract import AbstractPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import get_safe_torch_device, init_logging, set_global_seed


def write_video(video_path, stacked_frames, fps):
    imageio.mimsave(video_path, stacked_frames, fps=fps)


def eval_policy(
    env: BatchedEnvBase,
    policy: AbstractPolicy,
    num_episodes: int = 10,
    max_steps: int = 30,
    save_video: bool = False,
    video_dir: Path = None,
    fps: int = 15,
    return_first_video: bool = False,
):
    if policy is not None:
        policy.eval()
    start = time.time()
    sum_rewards = []
    max_rewards = []
    successes = []
    seeds = []
    threads = []  # for video saving threads
    episode_counter = 0  # for saving the correct number of videos

    # TODO(alexander-soare): if num_episodes is not evenly divisible by the batch size, this will do more work than
    # needed as I'm currently taking a ceil.
    for i in tqdm.tqdm(range(-(-num_episodes // env.batch_size[0]))):
        ep_frames = []

        def maybe_render_frame(env: EnvBase, _):
            if save_video or (return_first_video and i == 0):  # noqa: B023
                ep_frames.append(env.render())  # noqa: B023

        # Clear the policy's action queue before the start of a new rollout.
        if policy is not None:
            policy.clear_action_queue()

        if env.is_closed:
            env.start()  # needed to be able to get the seeds the first time as BatchedEnvs are lazy
        seeds.extend(env._next_seed)
        with torch.inference_mode():
            # TODO(alexander-soare): When `break_when_any_done == False` this rolls out for max_steps even when all
            # envs are done the first time. But we only use the first rollout. This is a waste of compute.
            rollout = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_cast_to_device=True,
                callback=maybe_render_frame,
                break_when_any_done=env.batch_size[0] == 1,
            )
        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        # Note: this assumes that the shape of the done key is (batch_size, max_steps, 1).
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        rollout_steps = rollout["next", "done"].shape[1]
        done_indices = torch.argmax(rollout["next", "done"].to(int), axis=1)  # (batch_size, rollout_steps)
        mask = (torch.arange(rollout_steps) <= done_indices).unsqueeze(-1)  # (batch_size, rollout_steps, 1)
        batch_sum_reward = einops.reduce((rollout["next", "reward"] * mask), "b n 1 -> b", "sum")
        batch_max_reward = einops.reduce((rollout["next", "reward"] * mask), "b n 1 -> b", "max")
        batch_success = einops.reduce((rollout["next", "success"] * mask), "b n 1 -> b", "any")
        sum_rewards.extend(batch_sum_reward.tolist())
        max_rewards.extend(batch_max_reward.tolist())
        successes.extend(batch_success.tolist())

        if save_video or (return_first_video and i == 0):
            batch_stacked_frames = np.stack(ep_frames)  # (t, b, *)
            batch_stacked_frames = batch_stacked_frames.transpose(
                1, 0, *range(2, batch_stacked_frames.ndim)
            )  # (b, t, *)

            if save_video:
                for stacked_frames, done_index in zip(
                    batch_stacked_frames, done_indices.flatten().tolist(), strict=False
                ):
                    if episode_counter >= num_episodes:
                        continue
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"eval_episode_{episode_counter}.mp4"
                    thread = threading.Thread(
                        target=write_video,
                        args=(str(video_path), stacked_frames[:done_index], fps),
                    )
                    thread.start()
                    threads.append(thread)
                    episode_counter += 1

            if return_first_video and i == 0:
                first_video = batch_stacked_frames[0].transpose(0, 3, 1, 2)

    for thread in threads:
        thread.join()

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:num_episodes],
                    max_rewards[:num_episodes],
                    successes[:num_episodes],
                    seeds[:num_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": np.nanmean(sum_rewards[:num_episodes]),
            "avg_max_reward": np.nanmean(max_rewards[:num_episodes]),
            "pc_success": np.nanmean(successes[:num_episodes]) * 100,
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / num_episodes,
        },
    }
    if return_first_video:
        return info, first_video
    return info


def eval(cfg: dict, out_dir=None, stats_path=None):
    if out_dir is None:
        raise NotImplementedError()

    init_logging()

    # Check device is available
    get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making transforms.")
    offline_buffer = make_offline_buffer(cfg, stats_path=stats_path)

    logging.info("Making environment.")
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

    info = eval_policy(
        env,
        policy=policy,
        save_video=True,
        video_dir=Path(out_dir) / "eval",
        fps=cfg.env.fps,
        max_steps=cfg.env.episode_length,
        num_episodes=cfg.eval_episodes,
    )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


def _relative_path_between(path1: Path, path2: Path) -> Path:
    """Returns path1 relative to path2."""
    path1 = path1.absolute()
    path2 = path2.absolute()
    try:
        return path1.relative_to(path2)
    except ValueError:  # most likely because path1 is not a subpath of path2
        common_parts = Path(osp.commonpath([path1, path2])).parts
        return Path(
            "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a specific yaml config you want to use.")
    group.add_argument("--hub-id", help="HuggingFace Hub ID for a pretrained model.")
    parser.add_argument("--revision", help="Optionally provide the HuggingFace Hub revision ID.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.config is not None:
        # Note: For the config_path, Hydra wants a path relative to this script file.
        hydra.initialize(
            config_path=str(
                _relative_path_between(Path(args.config).absolute().parent, Path(__file__).parent)
            )
        )
        cfg = hydra.compose(Path(args.config).stem, args.overrides)
        # TODO(alexander-soare): Save and load stats in trained model directory.
        stats_path = None
    elif args.hub_id is not None:
        folder = Path(snapshot_download(args.hub_id, revision="v1.0"))
        cfg = hydra.initialize(config_path=str(_relative_path_between(folder, Path(__file__).parent)))
        cfg = hydra.compose("config", args.overrides)
        cfg.policy.pretrained_model_path = folder / "model.pt"
        stats_path = folder / "stats.pth"

    eval(
        cfg,
        out_dir=f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{cfg.env.name}_{cfg.policy.name}",
        stats_path=stats_path,
    )
