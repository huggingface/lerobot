"""Evaluate a policy on an environment by running rollouts and computing metrics.

The script may be run in one of two ways:

1. By providing the path to a config file with the --config argument.
2. By providing a HuggingFace Hub ID with the --hub-id argument. You may also provide a revision number with the
    --revision argument.

In either case, it is possible to override config arguments by adding a list of config.key=value arguments.

Examples:

You have a specific config file to go with trained model weights, and want to run 10 episodes.

```
python lerobot/scripts/eval.py \
--config PATH/TO/FOLDER/config.yaml \
policy.pretrained_model_path=PATH/TO/FOLDER/weights.pth \
eval_episodes=10
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
import threading
import time
from datetime import datetime as dt
from pathlib import Path

import einops
import gymnasium as gym
import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import postprocess_action, preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed


def write_video(video_path, stacked_frames, fps):
    imageio.mimsave(video_path, stacked_frames, fps=fps)


def eval_policy(
    env: gym.vector.VectorEnv,
    policy,
    save_video: bool = False,
    video_dir: Path = None,
    # TODO(rcadene): make it possible to overwrite fps? we should use env.fps
    fps: int = 15,
    return_first_video: bool = False,
    transform: callable = None,
    seed=None,
):
    if policy is not None:
        policy.eval()
    device = "cpu" if policy is None else next(policy.parameters()).device

    start = time.time()
    sum_rewards = []
    max_rewards = []
    all_successes = []
    seeds = []
    threads = []  # for video saving threads
    episode_counter = 0  # for saving the correct number of videos

    num_episodes = len(env.envs)

    # TODO(alexander-soare): if num_episodes is not evenly divisible by the batch size, this will do more work than
    # needed as I'm currently taking a ceil.
    ep_frames = []

    def maybe_render_frame(env):
        if save_video:  # noqa: B023
            if return_first_video:
                visu = env.envs[0].render(mode="visualization")
                visu = visu[None, ...]  # add batch dim
            else:
                # TODO(now): Put mode back in.
                visu = np.stack([env.render() for env in env.envs])
                # visu = np.stack([env.render(mode="visualization") for env in env.envs])
            ep_frames.append(visu)  # noqa: B023

    for _ in range(num_episodes):
        seeds.append("TODO")

    if hasattr(policy, "reset"):
        policy.reset()
    else:
        logging.warning(
            f"Policy {policy} doesnt have a `reset` method. It is required if the policy relies on an internal state during rollout."
        )

    # reset the environment
    observation, info = env.reset(seed=seed)
    maybe_render_frame(env)

    rewards = []
    successes = []
    dones = []

    done = torch.tensor([False for _ in env.envs])
    step = 0
    while not done.all():
        # apply transform to normalize the observations
        observation = preprocess_observation(observation, transform)

        # send observation to device/gpu
        observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

        # get the next action for the environment
        with torch.inference_mode():
            action = policy.select_action(observation, step)

        # apply inverse transform to unnormalize the action
        action = postprocess_action(action, transform)

        # apply the next
        observation, reward, terminated, truncated, info = env.step(action)
        maybe_render_frame(env)

        # TODO(rcadene): implement a wrapper over env to return torch tensors in float32 (and cuda?)
        reward = torch.from_numpy(reward)
        terminated = torch.from_numpy(terminated)
        truncated = torch.from_numpy(truncated)
        # environment is considered done (no more steps), when success state is reached (terminated is True),
        # or time limit is reached (truncated is True), or it was previsouly done.
        done = terminated | truncated | done

        if "final_info" in info:
            # VectorEnv stores is_success into `info["final_info"][env_id]["is_success"]` instead of `info["is_success"]`
            success = [
                env_info["is_success"] if env_info is not None else False for env_info in info["final_info"]
            ]
        else:
            success = [False for _ in env.envs]
        success = torch.tensor(success)

        rewards.append(reward)
        dones.append(done)
        successes.append(success)

        step += 1

    rewards = torch.stack(rewards, dim=1)
    successes = torch.stack(successes, dim=1)
    dones = torch.stack(dones, dim=1)

    # Figure out where in each rollout sequence the first done condition was encountered (results after
    # this won't be included).
    # Note: this assumes that the shape of the done key is (batch_size, max_steps).
    # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
    done_indices = torch.argmax(dones.to(int), axis=1)  # (batch_size, rollout_steps)
    expand_done_indices = done_indices[:, None].expand(-1, step)
    expand_step_indices = torch.arange(step)[None, :].expand(num_episodes, -1)
    mask = (expand_step_indices <= expand_done_indices).int()  # (batch_size, rollout_steps)
    batch_sum_reward = einops.reduce((rewards * mask), "b n -> b", "sum")
    batch_max_reward = einops.reduce((rewards * mask), "b n -> b", "max")
    batch_success = einops.reduce((successes * mask), "b n -> b", "any")
    sum_rewards.extend(batch_sum_reward.tolist())
    max_rewards.extend(batch_max_reward.tolist())
    all_successes.extend(batch_success.tolist())

    env.close()

    if save_video or return_first_video:
        batch_stacked_frames = np.stack(ep_frames, 1)  # (b, t, *)

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

        if return_first_video:
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
                    all_successes[:num_episodes],
                    seeds[:num_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:num_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:num_episodes])),
            "pc_success": float(np.nanmean(all_successes[:num_episodes]) * 100),
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
    # TODO(alexander-soare): Completely decouple datasets from evaluation.
    transform = make_dataset(cfg, stats_path=stats_path).transform

    logging.info("Making environment.")
    env = make_env(cfg, num_parallel_envs=cfg.rollout_batch_size)

    # when policy is None, rollout a random policy
    policy = make_policy(cfg) if cfg.policy.pretrained_model_path else None

    info = eval_policy(
        env,
        policy=policy,
        save_video=True,
        video_dir=Path(out_dir) / "eval",
        fps=cfg.env.fps,
        # TODO(rcadene): what should we do with the transform?
        transform=transform,
        seed=cfg.seed,
    )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


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
        cfg = init_hydra_config(args.config, args.overrides)
        # TODO(alexander-soare): Save and load stats in trained model directory.
        stats_path = None
    elif args.hub_id is not None:
        folder = Path(snapshot_download(args.hub_id, revision=args.revision))
        cfg = init_hydra_config(
            folder / "config.yaml", [f"policy.pretrained_model_path={folder / 'model.pt'}", *args.overrides]
        )
        stats_path = folder / "stats.pth"

    eval(
        cfg,
        out_dir=f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{cfg.env.name}_{cfg.policy.name}",
        stats_path=stats_path,
    )
