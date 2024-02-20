import time

import hydra
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from termcolor import colored
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.openx import OpenXExperienceReplay
from torchrl.data.replay_buffers import PrioritizedSliceSampler

from lerobot.common.datasets.factory import make_offline_buffer
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
    if cfg.pretrained_model_path:
        ckpt_path = (
            "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/offline.pt"
        )
        if "offline" in cfg.pretrained_model_path:
            policy.step = 25000
        elif "final" in cfg.pretrained_model_path:
            policy.step = 100000
        else:
            raise NotImplementedError()
        policy.load(ckpt_path)

    td_policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    # initialize offline dataset

    offline_buffer = make_offline_buffer(cfg)

    if cfg.balanced_sampling:
        num_traj_per_batch = cfg.batch_size

        online_sampler = PrioritizedSliceSampler(
            max_capacity=100_000,
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            num_slices=num_traj_per_batch,
            strict_length=False,
        )

        online_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100_000),
            sampler=online_sampler,
        )

    L = Logger(cfg.log_dir, cfg)

    online_episode_idx = 0
    start_time = time.time()
    step = 0
    last_log_step = 0
    last_save_step = 0

    while step < cfg.train_steps:
        is_offline = True
        num_updates = cfg.episode_length
        _step = step + num_updates
        rollout_metrics = {}

        if step >= cfg.offline_steps:
            is_offline = False

            # TODO: use SyncDataCollector for that?
            with torch.no_grad():
                rollout = env.rollout(
                    max_steps=cfg.episode_length,
                    policy=td_policy,
                    auto_cast_to_device=True,
                )
            assert len(rollout) <= cfg.episode_length
            rollout["episode"] = torch.tensor(
                [online_episode_idx] * len(rollout), dtype=torch.int
            )
            online_buffer.extend(rollout)

            ep_reward = rollout["next", "reward"].sum()
            ep_success = rollout["next", "success"].any()

            online_episode_idx += 1
            rollout_metrics = {
                "avg_reward": np.nanmean(ep_reward),
                "pc_success": np.nanmean(ep_success) * 100,
            }
            num_updates = len(rollout) * cfg.utd
            _step = min(step + len(rollout), cfg.train_steps)

        # Update model
        for i in range(num_updates):
            if is_offline:
                train_metrics = policy.update(offline_buffer, step + i)
            else:
                train_metrics = policy.update(
                    online_buffer,
                    step + i // cfg.utd,
                    demo_buffer=offline_buffer if cfg.balanced_sampling else None,
                )

        # Log training metrics
        env_step = int(_step * cfg.action_repeat)
        common_metrics = {
            "episode": online_episode_idx,
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
