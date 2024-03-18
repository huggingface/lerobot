import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import PrioritizedSliceSampler

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import format_big_number, init_logging, set_seed
from lerobot.scripts.eval import eval_policy


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(out_dir=None, job_name=None, config_name="default", config_path="../configs"):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


def log_train_info(logger, info, step, cfg, offline_buffer, is_offline):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    data_s = info["data_s"]
    update_s = info["update_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.policy.batch_size
    avg_samples_per_ep = offline_buffer.num_samples / offline_buffer.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / offline_buffer.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # in seconds
        f"data_s:{data_s:.3f}",
        f"updt_s:{update_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_offline"] = is_offline

    logger.log_dict(info, step, mode="train")


def log_eval_info(logger, info, step, cfg, offline_buffer, is_offline):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.policy.batch_size
    avg_samples_per_ep = offline_buffer.num_samples / offline_buffer.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / offline_buffer.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_offline"] = is_offline

    logger.log_dict(info, step, mode="eval")


def train(cfg: dict, out_dir=None, job_name=None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()

    if cfg.device == "cuda":
        assert torch.cuda.is_available()
    else:
        logging.warning("Using CPU, this will be slow.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info("make_offline_buffer")
    offline_buffer = make_offline_buffer(cfg)

    # TODO(rcadene): move balanced_sampling, per_alpha, per_beta outside policy
    if cfg.policy.balanced_sampling:
        logging.info("make online_buffer")
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
            transform=offline_buffer.transform,
        )

    logging.info("make_env")
    env = make_env(cfg, transform=offline_buffer.transform)

    logging.info("make_policy")
    policy = make_policy(cfg)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    td_policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )

    # log metrics to terminal and wandb
    logger = Logger(out_dir, job_name, cfg)

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.offline_steps=} ({format_big_number(cfg.offline_steps)})")
    logging.info(f"{cfg.online_steps=}")
    logging.info(f"{cfg.env.action_repeat=}")
    logging.info(f"{offline_buffer.num_samples=} ({format_big_number(offline_buffer.num_samples)})")
    logging.info(f"{offline_buffer.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    step = 0  # number of policy update (forward + backward + optim)

    is_offline = True
    for offline_step in range(cfg.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")
        # TODO(rcadene): is it ok if step_t=0 = 0 and not 1 as previously done?
        policy.train()
        train_info = policy.update(offline_buffer, step)
        if step % cfg.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_buffer, is_offline)

        if step > 0 and step % cfg.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            eval_info, first_video = eval_policy(
                env,
                td_policy,
                num_episodes=cfg.eval_episodes,
                max_steps=cfg.env.episode_length // cfg.n_action_steps,
                return_first_video=True,
                video_dir=Path(out_dir) / "eval",
                save_video=True,
            )
            log_eval_info(logger, eval_info, step, cfg, offline_buffer, is_offline)
            if cfg.wandb.enable:
                logger.log_video(first_video, step, mode="eval")
            logging.info("Resume training")

        if step > 0 and cfg.save_model and step % cfg.save_freq == 0:
            logging.info(f"Checkpoint policy at step {step}")
            logger.save_model(policy, identifier=step)
            logging.info("Resume training")

        step += 1

    demo_buffer = offline_buffer if cfg.policy.balanced_sampling else None
    online_step = 0
    is_offline = False
    for env_step in range(cfg.online_steps):
        if env_step == 0:
            logging.info("Start online training by interacting with environment")
        # TODO: add configurable number of rollout? (default=1)
        with torch.no_grad():
            rollout = env.rollout(
                max_steps=cfg.env.episode_length // cfg.n_action_steps,
                policy=td_policy,
                auto_cast_to_device=True,
            )
        assert len(rollout) <= cfg.env.episode_length // cfg.n_action_steps
        # set same episode index for all time steps contained in this rollout
        rollout["episode"] = torch.tensor([env_step] * len(rollout), dtype=torch.int)
        online_buffer.extend(rollout)

        ep_sum_reward = rollout["next", "reward"].sum()
        ep_max_reward = rollout["next", "reward"].max()
        ep_success = rollout["next", "success"].any()
        rollout_info = {
            "avg_sum_reward": np.nanmean(ep_sum_reward),
            "avg_max_reward": np.nanmean(ep_max_reward),
            "pc_success": np.nanmean(ep_success) * 100,
            "env_step": env_step,
            "ep_length": len(rollout),
        }

        for _ in range(cfg.policy.utd):
            train_info = policy.update(
                online_buffer,
                step,
                demo_buffer=demo_buffer,
            )
            if step % cfg.log_freq == 0:
                train_info.update(rollout_info)
                log_train_info(logger, train_info, step, cfg, offline_buffer, is_offline)

            if step > 0 and step % cfg.eval_freq == 0:
                logging.info(f"Eval policy at step {step}")
                eval_info, first_video = eval_policy(
                    env,
                    td_policy,
                    num_episodes=cfg.eval_episodes,
                    max_steps=cfg.env.episode_length // cfg.n_action_steps,
                    return_first_video=True,
                )
                log_eval_info(logger, eval_info, step, cfg, offline_buffer, is_offline)
                if cfg.wandb.enable:
                    logger.log_video(first_video, step, mode="eval")
                logging.info("Resume training")

            if step > 0 and cfg.save_model and step % cfg.save_freq == 0:
                logging.info(f"Checkpoint policy at step {step}")
                logger.save_model(policy, identifier=step)
                logging.info("Resume training")

            step += 1
            online_step += 1

    logging.info("End of training")


if __name__ == "__main__":
    train_cli()
