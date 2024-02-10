import pickle
import time
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from termcolor import colored
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.openx import OpenXExperienceReplay
from torchrl.data.replay_buffers import PrioritizedSliceSampler

from lerobot.common.datasets.simxarm import SimxarmExperienceReplay
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger
from lerobot.common.tdmpc import TDMPC
from lerobot.common.utils import set_seed
from lerobot.scripts.eval import eval_policy


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def train(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.log_dir)

    env = make_env(cfg)
    policy = TDMPC(cfg)
    # ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/offline.pt"
    ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt"
    policy.load(ckpt_path)

    td_policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    # initialize offline dataset

    dataset_id = f"xarm_{cfg.task}_medium"

    num_traj_per_batch = cfg.batch_size  # // cfg.horizon
    # TODO(rcadene): Sampler outputs a batch_size <= cfg.batch_size.
    # We would need to add a transform to pad the tensordict to ensure batch_size == cfg.batch_size.
    sampler = PrioritizedSliceSampler(
        max_capacity=100_000,
        alpha=0.7,
        beta=0.9,
        num_slices=num_traj_per_batch,
        strict_length=False,
    )

    # TODO(rcadene): use PrioritizedReplayBuffer
    offline_buffer = SimxarmExperienceReplay(
        dataset_id,
        # download="force",
        download=True,
        streaming=False,
        root="data",
        sampler=sampler,
    )

    num_steps = len(offline_buffer)
    index = torch.arange(0, num_steps, 1)
    sampler.extend(index)

    # offline_buffer._storage.device = torch.device("cuda")
    # offline_buffer._storage._storage.to(torch.device("cuda"))
    # TODO(rcadene): add online_buffer

    # Observation encoder
    # Dynamics predictor
    # Reward predictor
    # Policy
    # Qs state-action value predictor
    # V state value predictor

    L = Logger(cfg.log_dir, cfg)

    episode_idx = 0
    start_time = time.time()
    step = 0
    last_log_step = 0
    last_save_step = 0

    while step < cfg.train_steps:
        is_offline = True
        num_updates = cfg.episode_length
        _step = step + num_updates
        rollout_metrics = {}

        # if step >= cfg.offline_steps:
        #     is_offline = False

        #     # Collect trajectory
        #     obs = env.reset()
        #     episode = Episode(cfg, obs)
        #     success = False
        #     while not episode.done:
        #         action = policy.act(obs, step=step, t0=episode.first)
        #         obs, reward, done, info = env.step(action.cpu().numpy())
        #         reward = reward_normalizer(reward)
        #         mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0
        #         success = info.get('success', False)
        #         episode += (obs, action, reward, done, mask, success)
        #     assert len(episode) <= cfg.episode_length
        #     buffer += episode
        #     episode_idx += 1
        #     rollout_metrics = {
        #         'episode_reward': episode.cumulative_reward,
        #         'episode_success': float(success),
        #         'episode_length': len(episode)
        #     }
        #     num_updates = len(episode) * cfg.utd
        #     _step = min(step + len(episode), cfg.train_steps)

        # Update model
        train_metrics = {}
        if is_offline:
            for i in range(num_updates):
                train_metrics.update(policy.update(offline_buffer, step + i))
        # else:
        #     for i in range(num_updates):
        #         train_metrics.update(
        #             policy.update(buffer, step + i // cfg.utd,
        #                          demo_buffer=offline_buffer if cfg.balanced_sampling else None)
        #         )

        # Log training metrics
        env_step = int(_step * cfg.action_repeat)
        common_metrics = {
            "episode": episode_idx,
            "step": _step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "is_offline": float(is_offline),
        }
        train_metrics.update(common_metrics)
        train_metrics.update(rollout_metrics)
        L.log(train_metrics, category="train")

        # Evaluate policy periodically
        if step == 0 or env_step - last_log_step >= cfg.eval_freq:

            eval_metrics = eval_policy(
                env,
                td_policy,
                num_episodes=cfg.eval_episodes,
                # TODO(rcadene): add step, env_step, L.video
            )

            # TODO(rcadene):
            # if hasattr(env, "get_normalized_score"):
            #     eval_metrics['normalized_score'] = env.get_normalized_score(eval_metrics["episode_reward"]) * 100.0

            common_metrics.update(eval_metrics)

            L.log(common_metrics, category="eval")
            last_log_step = env_step - env_step % cfg.eval_freq

        # Save model periodically
        # if cfg.save_model and env_step - last_save_step >= cfg.save_freq:
        #     L.save_model(policy, identifier=env_step)
        #     print(f"Model has been checkpointed at step {env_step}")
        #     last_save_step = env_step - env_step % cfg.save_freq

        # if cfg.save_model and is_offline and _step >= cfg.offline_steps:
        #     # save the model after offline training
        #     L.save_model(policy, identifier="offline")

        step = _step

    # dataset_d4rl = D4RLExperienceReplay(
    #     dataset_id="maze2d-umaze-v1",
    #     split_trajs=False,
    #     batch_size=1,
    #     sampler=SamplerWithoutReplacement(drop_last=False),
    #     prefetch=4,
    #     direct_download=True,
    # )

    # dataset_openx = OpenXExperienceReplay(
    #     "cmu_stretch",
    #     batch_size=1,
    #     num_slices=1,
    #     #download="force",
    #     streaming=False,
    #     root="data",
    # )


if __name__ == "__main__":
    train()
