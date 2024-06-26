#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock

import datasets
import hydra
import torch
from datasets import concatenate_datasets
from datasets.utils import disable_progress_bars, enable_progress_bars
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import calculate_episode_data_index, cycle, hf_transform_to_torch
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy


def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        lr_scheduler = None
    elif cfg.policy.name == "diffusion":
        optimizer = torch.optim.Adam(
            policy.diffusion.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif policy.name == "tdmpc":
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler


def update_policy(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    step: int = 0,
    lock=None,
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward(batch, step)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        loss = output_dict["loss"]
    grad_scaler.scale(loss).backward()

    # Unscale the graident of the optimzer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if isinstance(policy, PolicyWithUpdate):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
        **{k: v for k, v in output_dict.items() if k != "loss"},
    }
    info.update({k: v for k, v in output_dict.items() if k not in info})

    return info


def log_train_info(logger: Logger, info, step, cfg, dataset, is_offline):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
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
        f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # if not ~0, you are bottlenecked by cpu or io
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
    num_samples = (step + 1) * cfg.training.batch_size
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


def update_online_buffer(
    online_dataset: LeRobotDataset,
    concat_dataset: torch.utils.data.ConcatDataset,
    sampler: torch.utils.data.WeightedRandomSampler,
    new_hf_dataset: datasets.Dataset,
    new_episode_data_index: dict[str, torch.Tensor],
    online_sampling_ratio: float,
    buffer_capacity: float | None = None,
):
    """
    Modifies the online_dataset, concat_dataset, and sampler in place by integrating
    new episodes from new_hf_dataset into the online_dataset, updating the concatenated
    dataset's structure and adjusting the sampling strategy based on the specified
    percentage of online samples.

    Args:
        online_dataset: The existing online dataset to be updated.
        concat_dataset: The concatenated dataset that combines offline and online datasets (in that order),
            used for sampling purposes.
        sampler: A sampler that will be updated to reflect changes in the dataset sizes and specified sampling
            weights.
        new_hf_dataset: A Hugging Face dataset containing the new episodes to be added.
        new_episode_data_index: A dictionary containing two keys ("from" and "to") associated to dataset
            indices. They indicate the start index and end index of each episode in the dataset.
        online_sampling_ratio: The target percentage of samples that should come from the online dataset
            during sampling operations.
        buffer_capacity: A maximum capacity (in units of frames) for the online dataset. The dataset is
            treated like a queue where the first frames in are removed, if necessary, to make space for new
            frames.
    """
    # Sanity check to make sure that new_hf_dataset starts from 0.
    assert new_hf_dataset["episode_index"][0].item() == 0
    assert new_hf_dataset["index"][0].item() == 0
    # Sanity check to make sure that new_episode_data_index is aligned with new_hf_dataset.
    assert new_episode_data_index["from"][0].item() == 0
    assert new_episode_data_index["to"][-1].item() - 1 == new_hf_dataset["index"][-1].item()

    if len(online_dataset) == 0:
        # Initialize online dataset.
        online_dataset.hf_dataset = new_hf_dataset
        online_dataset.episode_data_index = new_episode_data_index
    else:
        # Get the indices required to continue where the data in online_dataset finishes.
        start_new_episode_indices = online_dataset.hf_dataset["episode_index"][-1].item() + 1
        start_new_indices = online_dataset.hf_dataset["index"][-1].item() + 1

        # Shift the indices of new_hf_dataset.
        disable_progress_bars()  # Dataset.map has a tqdm progress bar
        # note: we dont shift "frame_index" since it represents the index of the frame in the episode it
        # belongs to
        new_hf_dataset = new_hf_dataset.map(
            lambda episode_index, data_index: {
                "episode_index": episode_index + start_new_episode_indices,
                "index": data_index + start_new_indices,
            },
            input_columns=["episode_index", "index"],
        )
        enable_progress_bars()

        n_surplus = 0
        if (
            buffer_capacity is not None
            and (n_surplus := len(online_dataset) + len(new_hf_dataset) - buffer_capacity) > 0
        ):
            # Remove as many frames from the dataset as need to keep within the desired capacity.
            n_surplus = len(online_dataset.hf_dataset) - buffer_capacity
            hf_dataset = online_dataset.hf_dataset.select(range(n_surplus, len(online_dataset.hf_dataset)))
            # Remap the indices
            start_episode_index = hf_dataset["episode_index"][0]
            start_index = hf_dataset["index"][0]
            disable_progress_bars()
            online_dataset.hf_dataset = hf_dataset.map(
                lambda episode_index, data_index: {
                    "episode_index": episode_index - start_episode_index,
                    "index": data_index - start_index,
                },
                input_columns=["episode_index", "index"],
            )
            enable_progress_bars()

        # Extend the online dataset with the new data.
        online_dataset.hf_dataset = concatenate_datasets([online_dataset.hf_dataset, new_hf_dataset])

        # Minor optimization: if we didn't have to remove surplus frames we can calculate the episode data
        # index with a shortcut.
        if n_surplus == 0:
            online_dataset.episode_data_index = {
                k: torch.cat(
                    [
                        online_dataset.episode_data_index[k],
                        new_episode_data_index[k] + start_new_episode_indices,
                    ]
                )
                for k in ["from", "to"]
            }
        else:
            # Calculate the episode_data_index
            online_dataset.episode_data_index = calculate_episode_data_index(online_dataset.hf_dataset)
            # Shift cache indices
            if online_dataset.cache is not None:
                shifted_cache = {k - n_surplus: v for k, v in online_dataset.cache.items() if k >= n_surplus}
                online_dataset.cache = shifted_cache

    # update the concatenated dataset length used during sampling
    concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

    # update the sampling weights for each frame so that online frames get sampled a certain percentage of
    # times
    len_online = len(online_dataset)
    len_offline = len(concat_dataset) - len_online
    sampler.weights = torch.tensor(
        [(1 - online_sampling_ratio) / len_offline] * len_offline
        + [online_sampling_ratio / len_online] * len_online
    )

    # update the total number of samples used during sampling
    sampler.num_samples = len(concat_dataset)

    # Note: This is needed as for some reason, some HF dataset operations seem to remove the transform?
    online_dataset.hf_dataset.set_transform(hf_transform_to_torch)


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()

    # Check if settings are compatible with online training.
    if isinstance(cfg.dataset_repo_id, ListConfig):
        raise NotImplementedError("Online training with LeRobotMultiDataset is not implemented.")

    # If we are resuming a run, we need to check that a checkpoint exists in the log directory, and we need
    # to check for any differences between the provided config and the checkpoint's config.
    if cfg.resume:
        if not Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                "You have set resume=True, but there is no model checkpoint in "
                f"{Logger.get_last_checkpoint_dir(out_dir)}"
            )
        checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
        logging.info(
            colored(
                "You have set resume=True, indicating that you wish to resume a run",
                color="yellow",
                attrs=["bold"],
            )
        )
        # Get the configuration file from the last checkpoint.
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
        # Check for differences between the checkpoint configuration and provided configuration.
        # Hack to resolve the delta_timestamps ahead of time in order to properly diff.
        resolve_delta_timestamps(cfg)
        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # Ignore the `resume` and parameters.
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        # Log a warning about differences between the checkpoint configuration and the provided
        # configuration.
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
        cfg = checkpoint_cfg
        cfg.resume = True
    elif Logger.get_last_checkpoint_dir(out_dir).exists():
        raise RuntimeError(
            f"The configured output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists."
        )

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.training.eval_freq > 0:
        logging.info("make_env")
        eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    )
    assert isinstance(policy, nn.Module)
    # Create optimizer and scheduler
    # Temporary hack to move optimizer out of policy
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)
    online_dataset = None
    if cfg.resume:
        step, online_dataset = logger.load_last_training_state(optimizer, lr_scheduler)
        if online_dataset is not None:
            resolve_delta_timestamps(cfg)
            online_dataset.delta_timestamps = cfg.training.delta_timestamps

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{offline_dataset.num_samples=} ({format_big_number(offline_dataset.num_samples)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                assert eval_env is not None
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_offline=True)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpont(
                step,
                policy,
                optimizer,
                lr_scheduler,
                identifier=step_identifier,
                online_buffer=online_dataset,
            )
            logging.info("Resume training")

    # create dataloader for offline training
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.dataloader_persistent_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    is_offline = True
    offline_step = 0
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp,
            step=step,
        )

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_offline=True)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1)

        step += 1
        offline_step += 1  # noqa: SIM113

    # create an env dedicated to online episodes collection from policy rollout
    online_training_env = make_env(cfg, n_envs=cfg.training.online_rollout_batch_size)
    if not cfg.resume or online_dataset is None:
        # create an empty online dataset similar to offline dataset
        online_dataset = deepcopy(offline_dataset)
        # TODO(now): Consolidate the reset into one method.
        online_dataset.hf_dataset = {}
        online_dataset.episode_data_index = {}
        online_dataset.cache = {}

    online_rollout_policy = deepcopy(policy)
    online_rollout_policy.eval()

    # create dataloader for online training
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    if len(offline_dataset) > 0 and len(online_dataset) > 0:
        weights = torch.tensor(
            [(1 - cfg.training.online_sampling_ratio) / len(offline_dataset)] * len(offline_dataset)
            + [cfg.training.online_sampling_ratio / len(online_dataset)] * len(online_dataset)
        )
    elif len(offline_dataset) > 0:
        weights = torch.ones(len(offline_dataset))
    elif len(online_dataset) > 0:
        weights = torch.ones(len(online_dataset))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(concat_dataset), replacement=True
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        num_workers=0,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    lock = Lock()

    executor = ThreadPoolExecutor(max_workers=1)

    online_step = 0
    online_rollout_s = 0
    update_online_buffer_s = 0
    await_rollout_s = 0
    rollout_start_seed = 0
    is_offline = False
    while True:
        if online_step == cfg.training.online_steps:
            break

        if online_step == 0:
            logging.info("Start online training by interacting with environment")

        def sample_trajectory_and_update_buffer():
            nonlocal rollout_start_seed
            with lock:
                online_rollout_policy.load_state_dict(policy.state_dict())
            online_rollout_policy.eval()
            start_rollout_time = time.perf_counter()
            with torch.no_grad():
                eval_info = eval_policy(
                    online_training_env,
                    online_rollout_policy,
                    n_episodes=cfg.training.online_rollout_n_episodes,
                    max_episodes_rendered=min(10, cfg.training.online_rollout_n_episodes),
                    videos_dir=Path("test"),
                    return_episode_data=True,
                    start_seed=(
                        rollout_start_seed := (rollout_start_seed + cfg.training.batch_size) % 1000000
                    ),
                    enable_progbar=False,
                )
            online_rollout_s = time.perf_counter() - start_rollout_time

            with lock:
                start_update_buffer_time = time.perf_counter()
                update_online_buffer(
                    online_dataset,
                    concat_dataset,
                    sampler,
                    new_hf_dataset=eval_info["episodes"]["hf_dataset"],
                    new_episode_data_index=eval_info["episodes"]["episode_data_index"],
                    online_sampling_ratio=cfg.training.online_sampling_ratio,
                    buffer_capacity=cfg.training.online_buffer_capacity,
                )
                update_online_buffer_s = time.perf_counter() - start_update_buffer_time

            return online_rollout_s, update_online_buffer_s

        future = executor.submit(sample_trajectory_and_update_buffer)
        if len(online_dataset) == 0 or not cfg.training.do_online_rollout_async:
            start_time = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()

        policy.train()
        for _ in range(cfg.training.online_steps_between_rollouts):
            with lock:
                start_time = time.perf_counter()
                batch = next(dl_iter)
                dataloading_s = time.perf_counter() - start_time

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            train_info = update_policy(
                policy,
                batch,
                optimizer,
                cfg.training.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
                step=step,
                lock=lock,
            )

            train_info["dataloading_s"] = dataloading_s
            train_info["await_rollout_s"] = await_rollout_s
            train_info["online_rollout_s"] = online_rollout_s
            train_info["update_online_buffer_s"] = update_online_buffer_s

            if step % cfg.training.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_offline)

            # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
            # so we pass in step + 1.
            evaluate_and_checkpoint_if_needed(step + 1)

            step += 1
            online_step += 1

        if future.running():
            start = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()
            await_rollout_s = time.perf_counter() - start

    if eval_env:
        eval_env.close()
    logging.info("End of training")


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
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


if __name__ == "__main__":
    train_cli()
