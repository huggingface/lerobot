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
from accelerate.utils import set_seed as accelerate_set_seed
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    
    # Use accelerator's autocast context if mixed precision is enabled
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    
    # Use accelerator for backward pass
    accelerator.backward(loss)
    
    # Gradient clipping - accelerator handles unscaling automatically
    if accelerator.sync_gradients and grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.tensor(0.0)
    
    optimizer.step()
    lr_scheduler.step() if lr_scheduler is not None else None
    optimizer.zero_grad()
    
    # Update policy-specific buffers if needed
    if has_method(policy, "update"):
        policy.update()
    
    # Gather metrics across all processes
    loss_value = accelerator.gather(loss.detach()).mean().item()
    grad_norm_value = accelerator.gather(grad_norm).mean().item()
    
    train_metrics.loss = loss_value
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    
    # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin
    from lerobot.common.utils.wandb_utils import WandBLogger, cfg_to_group, get_wandb_run_id_from_filesystem

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.policy.use_amp else "no",
        gradient_accumulation_steps=cfg.policy.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.enable else None,
        kwargs_handlers=[ddp_kwargs],
        project_dir=cfg.output_dir,
    )
    

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": cfg.job_name,
                "notes": cfg.wandb.notes,
                "tags": cfg_to_group(cfg, return_list=True),
                "dir": cfg.output_dir,
                "config": cfg.to_dict(),
                "save_code": False,
                "job_type": "train_eval",
                "mode": cfg.wandb.mode if cfg.wandb.mode in ["online", "offline", "disabled"] else "online",
                "resume": "must" if cfg.resume else None,
                "id": cfg.wandb.run_id if cfg.wandb.run_id else (
                    get_wandb_run_id_from_filesystem(cfg.output_dir) if cfg.resume else None
                ),
            }
        }
    )
    
    # Set seed for reproducibility
    if cfg.seed is not None:
        accelerate_set_seed(cfg.seed)
    
    # Setup device - accelerator handles device placement
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create dataset
    if accelerator.is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    
    # Create evaluation environment (only on main process)
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and accelerator.is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    # Create policy
    if accelerator.is_main_process:
        logging.info("Creating policy")
    
    # Use accelerator's device instead of cfg.policy.device
    with accelerator.main_process_first():
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
    
    # Create optimizer and scheduler
    if accelerator.is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    
    step = 0  # number of policy updates
    
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    # Prepare dataloader
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
        pin_memory=True,
        drop_last=True,  # Important for distributed training
    )
    
    # Prepare for distributed training
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    
    # Log training info (only on main process)
    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info(f"Number of processes: {accelerator.num_processes}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Create metrics trackers
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes,  # Account for all processes
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step
    )
    
    # Training loop
    policy.train()
    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")
    
    # Create iterator from dataloader
    dl_iter = iter(dataloader)
    
    for current_step in range(step, cfg.steps):
        start_time = time.perf_counter()
        
        # Get next batch, cycling through dataloader if needed
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        
        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        # Update policy
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
        )
        
        # Increment step counter
        step += 1
        train_tracker.step()
        
        # Determine if we should log, save, or evaluate
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        
        # Logging (only on main process)
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            wandb_log_dict = train_tracker.to_dict()
            if output_dict:
                wandb_log_dict.update(output_dict)
            for k, v in wandb_log_dict.items():
                accelerator.log({f"{'train'}/{k}": v}, step=step)
            train_tracker.reset_averages()
        
        # Checkpointing (only on main process)
        if cfg.save_checkpoint and is_saving_step and accelerator.is_main_process:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            
            # Wait for all processes before saving
            accelerator.wait_for_everyone()
            
            # Unwrap model for saving
            unwrapped_policy = accelerator.unwrap_model(policy)
            save_checkpoint(checkpoint_dir, step, cfg, unwrapped_policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # if wandb_logger:
            #     wandb_logger.log_policy(checkpoint_dir)

        # Evaluation (only on main process)
        if cfg.env and is_eval_step and accelerator.is_main_process:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            
            # Unwrap model for evaluation
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.eval()
            
            with torch.no_grad():
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
                cfg.batch_size * accelerator.num_processes,
                dataset.num_frames,
                dataset.num_episodes,
                eval_metrics,
                initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            

            wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
            for k, v in wandb_log_dict.items():
                accelerator.log({f"{'eval'}/{k}": v}, step=step)
            
            # Set back to training mode
            policy.train()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Cleanup
    if eval_env and accelerator.is_main_process:
        eval_env.close()
    
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
