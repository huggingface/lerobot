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
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.tensor(0.0, device=accelerator.device)

    # Use accelerator's optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    # Initialize accelerator with DDP configuration for unused parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        mixed_precision=getattr(cfg, "mixed_precision", "no"),
        log_with="wandb" if cfg.wandb.enable and cfg.wandb.project else None,
        project_dir=str(cfg.output_dir) if cfg.wandb.enable and cfg.wandb.project else None,
        kwargs_handlers=[ddp_kwargs],
    )

    # Only log on main process
    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Use accelerator's device instead of manual device selection
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if accelerator.is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if accelerator.is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    if accelerator.is_main_process:
        logging.info("Creating optimizer and scheduler")

    # Scale learning rate and scheduler parameters for distributed training
    original_lr = cfg.optimizer.lr
    scaled_lr = original_lr * accelerator.num_processes
    cfg.optimizer.lr = scaled_lr

    # Scale scheduler parameters to account for faster data consumption in distributed training
    original_scheduler_params = {}
    if cfg.scheduler is not None:
        # Store original values
        if hasattr(cfg.scheduler, "num_warmup_steps"):
            original_scheduler_params["num_warmup_steps"] = cfg.scheduler.num_warmup_steps
            cfg.scheduler.num_warmup_steps = int(cfg.scheduler.num_warmup_steps * accelerator.num_processes)

        if hasattr(cfg.scheduler, "num_decay_steps"):
            original_scheduler_params["num_decay_steps"] = cfg.scheduler.num_decay_steps
            cfg.scheduler.num_decay_steps = int(cfg.scheduler.num_decay_steps * accelerator.num_processes)

    if accelerator.is_main_process:
        logging.info(
            f"Scaling learning rate from {original_lr} to {scaled_lr} for {accelerator.num_processes} processes"
        )
        if cfg.scheduler is not None and original_scheduler_params:
            for param_name, original_value in original_scheduler_params.items():
                new_value = getattr(cfg.scheduler, param_name)
                logging.info(f"Scaling scheduler {param_name} from {original_value} to {new_value}")

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # No need for GradScaler when using accelerator's mixed precision
    _grad_scaler = GradScaler(device.type, enabled=False)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if accelerator.is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} = {cfg.batch_size * accelerator.num_processes}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Prepare model, optimizer, scheduler, and dataloader with accelerator
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # No need for manual device transfers - accelerator handles this
        # The dataloader is already prepared and tensors are on the correct device

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

                # Use accelerator to unwrap the model before saving
                unwrapped_policy = accelerator.unwrap_model(policy)
                save_checkpoint(checkpoint_dir, step, cfg, unwrapped_policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Wait for all processes after checkpoint is saved
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            # Only evaluate on main process
            if accelerator.is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")

                # Use unwrapped model for evaluation and accelerator's autocast
                unwrapped_policy = accelerator.unwrap_model(policy)
                with (
                    torch.no_grad(),
                    accelerator.autocast(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        unwrapped_policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

            # Wait for evaluation to complete before continuing training
            accelerator.wait_for_everyone()

    if eval_env:
        eval_env.close()

    if accelerator.is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            # Use unwrapped model for pushing to hub
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
