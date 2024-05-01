"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_policy_pusht_image)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_policy_pusht_image eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_policy_pusht_image/checkpoints/005000 \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_policy_pusht_image/tree/main.
"""

import argparse
import json
import logging
import threading
import time
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import einops
import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from PIL import Image as PILImage
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import postprocess_action, preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: torch.nn.Module,
    max_episodes_rendered: int = 0,
    video_dir: Path = None,
    return_episode_data: bool = False,
    seed=None,
):
    """
    set `return_episode_data` to return a Hugging Face dataset object in an "episodes" key of the return dict.
    """
    fps = env.unwrapped.metadata["render_fps"]

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

    def render_frame(env):
        # noqa: B023
        eps_rendered = min(max_episodes_rendered, len(env.envs))
        visu = np.stack([env.envs[i].render() for i in range(eps_rendered)])
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
    if max_episodes_rendered > 0:
        render_frame(env)

    observations = []
    actions = []
    # episode
    # frame_id
    # timestamp
    rewards = []
    successes = []
    dones = []

    done = torch.tensor([False for _ in env.envs])
    step = 0
    max_steps = env.envs[0]._max_episode_steps
    progbar = trange(max_steps, desc=f"Running eval with {max_steps} steps (maximum) per rollout.")
    while not done.all():
        # format from env keys to lerobot keys
        observation = preprocess_observation(observation)
        if return_episode_data:
            observations.append(deepcopy(observation))

        # send observation to device/gpu
        observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

        # get the next action for the environment
        with torch.inference_mode():
            action = policy.select_action(observation, step=step)

        # convert to cpu numpy
        action = postprocess_action(action)

        # apply the next action
        observation, reward, terminated, truncated, info = env.step(action)
        if max_episodes_rendered > 0:
            render_frame(env)

        # TODO(rcadene): implement a wrapper over env to return torch tensors in float32 (and cuda?)
        action = torch.from_numpy(action)
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

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        successes.append(success)

        step += 1
        progbar.update()

    env.close()

    # add the last observation when the env is done
    if return_episode_data:
        observation = preprocess_observation(observation)
        observations.append(deepcopy(observation))

    if return_episode_data:
        new_obses = {}
        for key in observations[0].keys():  # noqa: SIM118
            new_obses[key] = torch.stack([obs[key] for obs in observations], dim=1)
        observations = new_obses
    actions = torch.stack(actions, dim=1)
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

    # similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`)
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    num_episodes = dones.shape[0]
    total_frames = 0
    id_from = 0
    for ep_id in range(num_episodes):
        num_frames = done_indices[ep_id].item() + 1
        total_frames += num_frames

        # TODO(rcadene): We need to add a missing last frame which is the observation
        # of a done state. it is critical to have this frame for tdmpc to predict a "done observation/state"
        if return_episode_data:
            ep_dict = {
                "action": actions[ep_id, :num_frames],
                "episode_index": torch.tensor([ep_id] * num_frames),
                "frame_index": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / fps,
                "next.done": dones[ep_id, :num_frames],
                "next.reward": rewards[ep_id, :num_frames].type(torch.float32),
            }
            for key in observations:
                ep_dict[key] = observations[key][ep_id][:num_frames]
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

    # similar logic is implemented in dataset preprocessing
    if return_episode_data:
        data_dict = {}
        keys = ep_dicts[0].keys()
        for key in keys:
            if "image" not in key:
                data_dict[key] = torch.cat([x[key] for x in ep_dicts])
            else:
                if key not in data_dict:
                    data_dict[key] = []
                for ep_dict in ep_dicts:
                    for img in ep_dict[key]:
                        # sanity check that images are channel first
                        c, h, w = img.shape
                        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

                        # sanity check that images are float32 in range [0,1]
                        assert img.dtype == torch.float32, f"expect torch.float32, but instead {img.dtype=}"
                        assert img.max() <= 1, f"expect pixels lower than 1, but instead {img.max()=}"
                        assert img.min() >= 0, f"expect pixels greater than 1, but instead {img.min()=}"

                        # from float32 in range [0,1] to uint8 in range [0,255]
                        img *= 255
                        img = img.type(torch.uint8)

                        # convert to channel last and numpy as expected by PIL
                        img = PILImage.fromarray(img.permute(1, 2, 0).numpy())

                        data_dict[key].append(img)

        data_dict["index"] = torch.arange(0, total_frames, 1)
        episode_data_index["from"] = torch.tensor(episode_data_index["from"])
        episode_data_index["to"] = torch.tensor(episode_data_index["to"])

        # TODO(rcadene): clean this
        features = {}
        for key in observations:
            if "image" in key:
                features[key] = Image()
            else:
                features[key] = Sequence(
                    length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None)
                )
        features.update(
            {
                "action": Sequence(
                    length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
                ),
                "episode_index": Value(dtype="int64", id=None),
                "frame_index": Value(dtype="int64", id=None),
                "timestamp": Value(dtype="float32", id=None),
                "next.reward": Value(dtype="float32", id=None),
                "next.done": Value(dtype="bool", id=None),
                #'next.success': Value(dtype='bool', id=None),
                "index": Value(dtype="int64", id=None),
            }
        )
        features = Features(features)
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)

    if max_episodes_rendered > 0:
        batch_stacked_frames = np.stack(ep_frames, 1)  # (b, t, *)

        for stacked_frames, done_index in zip(
            batch_stacked_frames, done_indices.flatten().tolist(), strict=False
        ):
            if episode_counter >= max_episodes_rendered:
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

        videos = einops.rearrange(batch_stacked_frames, "b t h w c -> b t c h w")

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
    if return_episode_data:
        info["episodes"] = {
            "hf_dataset": hf_dataset,
            "episode_data_index": episode_data_index,
        }
    if max_episodes_rendered > 0:
        info["videos"] = videos
    return info


def eval(
    pretrained_policy_path: str | None = None,
    hydra_cfg_path: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if hydra_cfg_path is None:
        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)
    out_dir = (
        f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"
    )

    if out_dir is None:
        raise NotImplementedError()

    # Check device is available
    get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg, num_parallel_envs=hydra_cfg.eval.n_episodes)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    info = eval_policy(
        env,
        policy,
        max_episodes_rendered=10,
        video_dir=Path(out_dir) / "eval",
        return_episode_data=False,
        seed=hydra_cfg.seed,
    )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        # remove pytorch tensors which are not serializable to save the evaluation results only
        del info["videos"]
        json.dump(info, f, indent=2)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        eval(hydra_cfg_path=args.config, config_overrides=args.overrides)
    else:
        try:
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except HFValidationError:
            logging.warning(
                "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID. "
                "Treating it as a local directory."
            )
        except RepositoryNotFoundError:
            logging.warning(
                "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub. Treating "
                "it as a local directory."
            )
        pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        eval(pretrained_policy_path=pretrained_policy_path, config_overrides=args.overrides)
