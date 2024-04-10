import logging
from copy import deepcopy
from pathlib import Path

import hydra
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
)
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


def log_train_info(logger, info, step, cfg, dataset, is_offline):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    data_s = info["data_s"]
    update_s = info["update_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.policy.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
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


def log_eval_info(logger, info, step, cfg, dataset, is_offline):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.policy.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
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


def calculate_online_sample_weight(n_off: int, n_on: int, pc_on: float):
    """
    Calculate the sampling weight to be assigned to samples so that a specified percentage of the batch comes from online dataset (on average).

    Parameters:
    - n_off (int): Number of offline samples, each with a sampling weight of 1.
    - n_on (int): Number of online samples.
    - pc_on (float): Desired percentage of online samples in decimal form (e.g., 50% as 0.5).

    The total weight of offline samples is n_off * 1.0.
    The total weight of offline samples is n_on * w.
    The total combined weight of all samples is n_off + n_on * w.
    The fraction of the weight that is online is n_on * w / (n_off + n_on * w).
    We want this fraction to equal pc_on, so we set up the equation n_on * w / (n_off + n_on * w) = pc_on.
    The solution is w = - (n_off * pc_on) / (n_on * (pc_on - 1))
    """
    assert 0.0 <= pc_on <= 1.0
    return -(n_off * pc_on) / (n_on * (pc_on - 1))


def add_episodes_inplace(episodes, online_dataset, concat_dataset, sampler, pc_online_samples):
    data_dict = episodes["data_dict"]
    data_ids_per_episode = episodes["data_ids_per_episode"]

    if len(online_dataset) == 0:
        # initialize online dataset
        online_dataset.data_dict = data_dict
        online_dataset.data_ids_per_episode = data_ids_per_episode
    else:
        # find episode index and data frame indices according to previous episode in online_dataset
        start_episode = max(online_dataset.data_ids_per_episode.keys()) + 1
        start_index = online_dataset.data_dict["index"][-1].item() + 1
        data_dict["episode"] += start_episode
        data_dict["index"] += start_index

        # extend online dataset
        for key in data_dict:
            # TODO(rcadene): avoid reallocating memory at every step by preallocating memory or changing our data structure
            online_dataset.data_dict[key] = torch.cat([online_dataset.data_dict[key], data_dict[key]])
        for ep_id in data_ids_per_episode:
            online_dataset.data_ids_per_episode[ep_id + start_episode] = (
                data_ids_per_episode[ep_id] + start_index
            )

    # update the concatenated dataset length used during sampling
    concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

    # update the sampling weights for each frame so that online frames get sampled a certain percentage of times
    len_online = len(online_dataset)
    len_offline = len(concat_dataset) - len_online
    weight_offline = 1.0
    weight_online = calculate_online_sample_weight(len_offline, len_online, pc_online_samples)
    sampler.weights = torch.tensor([weight_offline] * len_offline + [weight_online] * len(online_dataset))

    # update the total number of samples used during sampling
    sampler.num_samples = len(concat_dataset)


def train(cfg: dict, out_dir=None, job_name=None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()
    if cfg.online_steps > 0:
        assert cfg.rollout_batch_size == 1, "rollout_batch_size > 1 not supported for online training steps"

    init_logging()

    # Check device is available
    get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(cfg.seed)

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)

    logging.info("make_env")
    env = make_env(cfg, num_parallel_envs=cfg.eval_episodes)

    logging.info("make_policy")
    policy = make_policy(cfg)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # log metrics to terminal and wandb
    logger = Logger(out_dir, job_name, cfg)

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.offline_steps=} ({format_big_number(cfg.offline_steps)})")
    logging.info(f"{cfg.online_steps=}")
    logging.info(f"{offline_dataset.num_samples=} ({format_big_number(offline_dataset.num_samples)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def _maybe_eval_and_maybe_save(step):
        if step % cfg.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            eval_info = eval_policy(
                env,
                policy,
                video_dir=Path(out_dir) / "eval",
                max_episodes_rendered=4,
                transform=offline_dataset.transform,
                seed=cfg.seed,
            )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_offline)
            if cfg.wandb.enable:
                logger.log_video(eval_info["videos"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.save_model and step % cfg.save_freq == 0:
            logging.info(f"Checkpoint policy after step {step}")
            logger.save_model(policy, identifier=step)
            logging.info("Resume training")

    # create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=4,
        batch_size=cfg.policy.batch_size,
        shuffle=True,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    step = 0  # number of policy update (forward + backward + optim)
    is_offline = True
    for offline_step in range(cfg.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")
        policy.train()
        batch = next(dl_iter)

        for key in batch:
            batch[key] = batch[key].to(cfg.device, non_blocking=True)

        train_info = policy(batch, step)

        # TODO(rcadene): is it ok if step_t=0 = 0 and not 1 as previously done?
        if step % cfg.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_offline)

        # Note: _maybe_eval_and_maybe_save happens **after** the `step`th training update has completed, so we pass in
        # step + 1.
        _maybe_eval_and_maybe_save(step + 1)

        step += 1

    # create an env dedicated to online episodes collection from policy rollout
    rollout_env = make_env(cfg, num_parallel_envs=1)

    # create an empty online dataset similar to offline dataset
    online_dataset = deepcopy(offline_dataset)
    online_dataset.data_dict = {}
    online_dataset.data_ids_per_episode = {}

    # create dataloader for online training
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    weights = [1.0] * len(concat_dataset)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(concat_dataset), replacement=True
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        num_workers=4,
        batch_size=cfg.policy.batch_size,
        sampler=sampler,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    online_step = 0
    is_offline = False
    for env_step in range(cfg.online_steps):
        if env_step == 0:
            logging.info("Start online training by interacting with environment")

        with torch.no_grad():
            eval_info = eval_policy(
                rollout_env,
                policy,
                transform=offline_dataset.transform,
                seed=cfg.seed,
            )

            online_pc_sampling = cfg.get("demo_schedule", 0.5)
            add_episodes_inplace(
                eval_info["episodes"], online_dataset, concat_dataset, sampler, online_pc_sampling
            )

        for _ in range(cfg.policy.utd):
            policy.train()
            batch = next(dl_iter)

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            train_info = policy(batch, step)

            if step % cfg.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_offline)

            # Note: _maybe_eval_and_maybe_save happens **after** the `step`th training update has completed, so we pass
            # in step + 1.
            _maybe_eval_and_maybe_save(step + 1)

            step += 1
            online_step += 1

    logging.info("End of training")


if __name__ == "__main__":
    train_cli()
