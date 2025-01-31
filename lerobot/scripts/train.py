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
from dataclasses import asdict
from pprint import pformat
from threading import Lock

import numpy as np
import torch
from torch.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.optim.factory import load_training_state, make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
    set_global_seed,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward(batch)
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

    if has_method(policy, "update"):
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


def log_train_info(
    logger: Logger, info: dict, step: int, cfg: TrainPipelineConfig, dataset: LeRobotDataset, is_online: bool
):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
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
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="train")


def log_eval_info(logger, info, step, cfg, dataset, is_online):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
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
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="eval")


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    logging.info(pformat(asdict(cfg)))

    # log metrics to terminal and wandb
    logger = Logger(cfg)

    if cfg.seed is not None:
        set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    offline_dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        ds_meta=offline_dataset.meta,
    )
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device, enabled=cfg.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(cfg.output_dir)
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.offline.steps=} ({format_big_number(cfg.offline.steps)})")
    logging.info(f"{cfg.online.steps=}")
    logging.info(f"{offline_dataset.num_frames=} ({format_big_number(offline_dataset.num_frames)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step: int, is_online: bool):
        _num_digits = max(6, len(str(cfg.offline.steps + cfg.online.steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.env is not None and cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.save_checkpoint and (
            step % cfg.save_freq == 0 or step == cfg.offline.steps + cfg.online.steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpoint(
                step,
                step_identifier,
                policy,
                optimizer,
                lr_scheduler,
            )
            logging.info("Resume training")

    # create dataloader for offline training
    if getattr(cfg.policy, "drop_n_last_frames", None):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    offline_step = 0
    for _ in range(step, cfg.offline.steps):
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
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp,
        )

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1  # noqa: SIM113

    if cfg.online.steps == 0:
        if eval_env:
            eval_env.close()
        logging.info("End of training")
        return

    # Online training.

    # Create an env dedicated to online episodes collection from policy rollout.
    online_env = make_env(cfg.env, n_envs=cfg.online.rollout_batch_size)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, offline_dataset.meta)
    online_buffer_path = logger.log_dir / "online_buffer"
    if cfg.resume and not online_buffer_path.exists():
        # If we are resuming a run, we default to the data shapes and buffer capacity from the saved online
        # buffer.
        logging.warning(
            "When online training is resumed, we load the latest online buffer from the prior run, "
            "and this might not coincide with the state of the buffer as it was at the moment the checkpoint "
            "was made. This is because the online buffer is updated on disk during training, independently "
            "of our explicit checkpointing mechanisms."
        )
    online_dataset = OnlineBuffer(
        online_buffer_path,
        data_spec={
            **{
                key: {"shape": ft.shape, "dtype": np.dtype("float32")}
                for key, ft in policy.config.input_features.items()
            },
            **{
                key: {"shape": ft.shape, "dtype": np.dtype("float32")}
                for key, ft in policy.config.output_features.items()
            },
            "next.reward": {"shape": (), "dtype": np.dtype("float32")},
            "next.done": {"shape": (), "dtype": np.dtype("?")},
            "task_index": {"shape": (), "dtype": np.dtype("int64")},
            # FIXME: 'next.success' is expected by pusht env but not xarm
            "next.success": {"shape": (), "dtype": np.dtype("?")},
        },
        buffer_capacity=cfg.online.buffer_capacity,
        fps=online_env.unwrapped.metadata["render_fps"],
        delta_timestamps=delta_timestamps,
    )

    # If we are doing online rollouts asynchronously, deepcopy the policy to use for online rollouts (this
    # makes it possible to do online rollouts in parallel with training updates).
    online_rollout_policy = deepcopy(policy) if cfg.online.do_rollout_async else policy

    # Create dataloader for online training.
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    sampler_weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
        online_dataset=online_dataset,
        # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
        # this final observation in the offline datasets, but we might add them in future.
        online_drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0) + 1,
        online_sampling_ratio=cfg.online.sampling_ratio,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sampler_weights,
        num_samples=len(concat_dataset),
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    if cfg.online.do_rollout_async:
        # Lock and thread pool executor for asynchronous online rollouts.
        lock = Lock()
        # Note: 1 worker because we only ever want to run one set of online rollouts at a time. Batch
        # parallelization of rollouts is handled within the job.
        executor = ThreadPoolExecutor(max_workers=1)
    else:
        lock = None

    online_step = 0
    online_rollout_s = 0  # time take to do online rollout
    update_online_buffer_s = 0  # time taken to update the online buffer with the online rollout data
    # Time taken waiting for the online buffer to finish being updated. This is relevant when using the async
    # online rollout option.
    await_update_online_buffer_s = 0
    rollout_start_seed = cfg.online.env_seed

    while True:
        if online_step == cfg.online.steps:
            break

        if online_step == 0:
            logging.info("Start online training by interacting with environment")

        def sample_trajectory_and_update_buffer():
            nonlocal rollout_start_seed

            with lock if lock is not None else nullcontext():
                online_rollout_policy.load_state_dict(policy.state_dict())

            online_rollout_policy.eval()
            start_rollout_time = time.perf_counter()

            with torch.no_grad():
                eval_info = eval_policy(
                    online_env,
                    online_rollout_policy,
                    n_episodes=cfg.online.rollout_n_episodes,
                    max_episodes_rendered=min(10, cfg.online.rollout_n_episodes),
                    videos_dir=logger.log_dir / "online_rollout_videos",
                    return_episode_data=True,
                    start_seed=(rollout_start_seed := (rollout_start_seed + cfg.batch_size) % 1000000),
                )
            online_rollout_s = time.perf_counter() - start_rollout_time

            if len(offline_dataset.meta.tasks) > 1:
                raise NotImplementedError("Add support for multi task.")

            # Hack to add a task to the online_dataset (0 is the first task of the offline_dataset)
            total_num_frames = eval_info["episodes"]["index"].shape[0]
            eval_info["episodes"]["task_index"] = torch.tensor([0] * total_num_frames, dtype=torch.int64)

            with lock if lock is not None else nullcontext():
                start_update_buffer_time = time.perf_counter()
                online_dataset.add_data(eval_info["episodes"])

                # Update the concatenated dataset length used during sampling.
                concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

                # Update the sampling weights.
                sampler.weights = compute_sampler_weights(
                    offline_dataset,
                    offline_drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                    online_dataset=online_dataset,
                    # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
                    # this final observation in the offline datasets, but we might add them in future.
                    online_drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0) + 1,
                    online_sampling_ratio=cfg.online.sampling_ratio,
                )
                sampler.num_frames = len(concat_dataset)

                update_online_buffer_s = time.perf_counter() - start_update_buffer_time

            return online_rollout_s, update_online_buffer_s

        if lock is None:
            online_rollout_s, update_online_buffer_s = sample_trajectory_and_update_buffer()
        else:
            future = executor.submit(sample_trajectory_and_update_buffer)
            # If we aren't doing async rollouts, or if we haven't yet gotten enough examples in our buffer, wait
            # here until the rollout and buffer update is done, before proceeding to the policy update steps.
            if len(online_dataset) <= cfg.online.buffer_seed_size:
                online_rollout_s, update_online_buffer_s = future.result()

        if len(online_dataset) <= cfg.online.buffer_seed_size:
            logging.info(f"Seeding online buffer: {len(online_dataset)}/{cfg.online.buffer_seed_size}")
            continue

        policy.train()
        for _ in range(cfg.online.steps_between_rollouts):
            with lock if lock is not None else nullcontext():
                start_time = time.perf_counter()
                batch = next(dl_iter)
                dataloading_s = time.perf_counter() - start_time

            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

            train_info = update_policy(
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
                lock=lock,
            )

            train_info["dataloading_s"] = dataloading_s
            train_info["online_rollout_s"] = online_rollout_s
            train_info["update_online_buffer_s"] = update_online_buffer_s
            train_info["await_update_online_buffer_s"] = await_update_online_buffer_s
            with lock if lock is not None else nullcontext():
                train_info["online_buffer_size"] = len(online_dataset)

            if step % cfg.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_online=True)

            # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
            # so we pass in step + 1.
            evaluate_and_checkpoint_if_needed(step + 1, is_online=True)

            step += 1
            online_step += 1

        # If we're doing async rollouts, we should now wait until we've completed them before proceeding
        # to do the next batch of rollouts.
        if cfg.online.do_rollout_async and future.running():
            start = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()
            await_update_online_buffer_s = time.perf_counter() - start

        if online_step >= cfg.online.steps:
            break

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
