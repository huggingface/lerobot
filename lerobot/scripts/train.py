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
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import set_seed
from lerobot.scripts.eval import eval_policy


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(
    out_dir=None, job_name=None, config_name="default", config_path="../configs"
):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


def log_training_metrics(L, metrics, step, online_episode_idx, start_time, is_offline):
    common_metrics = {
        "episode": online_episode_idx,
        "step": step,
        "total_time": time.time() - start_time,
        "is_offline": float(is_offline),
    }
    metrics.update(common_metrics)
    L.log(metrics, category="train")


def eval_policy_and_log(
    env, td_policy, step, online_episode_idx, start_time, is_offline, cfg, L
):
    common_metrics = {
        "episode": online_episode_idx,
        "step": step,
        "total_time": time.time() - start_time,
        "is_offline": float(is_offline),
    }
    metrics, first_video = eval_policy(
        env,
        td_policy,
        num_episodes=cfg.eval_episodes,
        return_first_video=True,
    )
    metrics.update(common_metrics)
    L.log(metrics, category="eval")

    if cfg.wandb.enable:
        eval_video = L._wandb.Video(first_video, fps=cfg.fps, format="mp4")
        L._wandb.log({"eval_video": eval_video}, step=step)


def train(cfg: dict, out_dir=None, job_name=None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), out_dir)

    env = make_env(cfg)
    policy = make_policy(cfg)

    td_policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    # initialize offline dataset

    offline_buffer = make_offline_buffer(cfg)

    # TODO(rcadene): move balanced_sampling, per_alpha, per_beta outside policy
    if cfg.policy.balanced_sampling:
        num_traj_per_batch = cfg.policy.batch_size

        online_sampler = PrioritizedSliceSampler(
            max_capacity=100_000,
            alpha=cfg.policy.per_alpha,
            beta=cfg.policy.per_beta,
            num_slices=num_traj_per_batch,
            strict_length=True,
        )

        online_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100_000),
            sampler=online_sampler,
        )

    L = Logger(out_dir, job_name, cfg)

    online_episode_idx = 0
    start_time = time.time()
    step = 0

    # First eval with a random model or pretrained
    eval_policy_and_log(
        env, td_policy, step, online_episode_idx, start_time, is_offline, cfg, L
    )

    # Train offline
    for _ in range(cfg.offline_steps):
        # TODO(rcadene): is it ok if step_t=0 = 0 and not 1 as previously done?
        metrics = policy.update(offline_buffer, step)

        if step % cfg.log_freq == 0:
            log_training_metrics(
                L, metrics, step, online_episode_idx, start_time, is_offline=False
            )

        if step > 0 and step % cfg.eval_freq == 0:
            eval_policy_and_log(
                env, td_policy, step, online_episode_idx, start_time, is_offline, cfg, L
            )

        if step > 0 and cfg.save_model and step % cfg.save_freq == 0:
            print(f"Checkpoint model at step {step}")
            L.save_model(policy, identifier=step)

        step += 1

    # Train online
    demo_buffer = offline_buffer if cfg.policy.balanced_sampling else None
    for _ in range(cfg.online_steps):
        # TODO: use SyncDataCollector for that?
        with torch.no_grad():
            rollout = env.rollout(
                max_steps=cfg.env.episode_length,
                policy=td_policy,
                auto_cast_to_device=True,
            )
        assert len(rollout) <= cfg.env.episode_length
        rollout["episode"] = torch.tensor(
            [online_episode_idx] * len(rollout), dtype=torch.int
        )
        online_buffer.extend(rollout)

        ep_sum_reward = rollout["next", "reward"].sum()
        ep_max_reward = rollout["next", "reward"].max()
        ep_success = rollout["next", "success"].any()
        metrics = {
            "avg_sum_reward": np.nanmean(ep_sum_reward),
            "avg_max_reward": np.nanmean(ep_max_reward),
            "pc_success": np.nanmean(ep_success) * 100,
        }

        online_episode_idx += 1

        for _ in range(cfg.policy.utd):
            train_metrics = policy.update(
                online_buffer,
                step,
                demo_buffer=demo_buffer,
            )
            metrics.update(train_metrics)
            if step % cfg.log_freq == 0:
                log_training_metrics(
                    L, metrics, step, online_episode_idx, start_time, is_offline=False
                )

            if step > 0 and step & cfg.eval_freq == 0:
                eval_policy_and_log(
                    env,
                    td_policy,
                    step,
                    online_episode_idx,
                    start_time,
                    is_offline,
                    cfg,
                    L,
                )

            if step > 0 and cfg.save_model and step % cfg.save_freq == 0:
                print(f"Checkpoint model at step {step}")
                L.save_model(policy, identifier=step)

            step += 1


if __name__ == "__main__":
    train_cli()
