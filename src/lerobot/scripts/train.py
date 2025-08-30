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
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch

# Fix tokenizer parallelism conflicts with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from lerobot.policies.factory import make_policy, make_processor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
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
    get_safe_torch_device,
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
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    
    # Forward pass timing
    forward_start = time.perf_counter()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    forward_time = time.perf_counter() - forward_start
    
    # Backward pass timing
    backward_start = time.perf_counter()
    grad_scaler.scale(loss).backward()
    backward_time = time.perf_counter() - backward_start

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer step timing
    optim_start = time.perf_counter()
    
    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()
        
    optim_time = time.perf_counter() - optim_start
    total_time = time.perf_counter() - start_time

    # Collect timing statistics for RLearN policy (averaged reporting every minute)
    if getattr(policy, "name", None) == "rlearn":
        # Initialize timing accumulator if not exists
        if not hasattr(policy, '_train_timing_stats'):
            policy._train_timing_stats = {
                'forward_times': [],
                'backward_times': [],
                'optim_times': [],
                'total_times': [],
                'last_print_time': time.perf_counter()
            }
        
        # Accumulate current step's timings
        stats = policy._train_timing_stats
        stats['forward_times'].append(forward_time * 1000)
        stats['backward_times'].append(backward_time * 1000)
        stats['optim_times'].append(optim_time * 1000)
        stats['total_times'].append(total_time * 1000)
        
        # Print averaged stats every minute (60 seconds)
        current_time = time.perf_counter()
        if current_time - stats['last_print_time'] >= 60.0:
            n_samples = len(stats['forward_times'])
            if n_samples > 0:
                print(f"\nTraining Step Average Timing (last {n_samples} steps):")
                print(f"  Forward pass:       {sum(stats['forward_times'])/n_samples:.2f} ms")
                print(f"  Backward pass:      {sum(stats['backward_times'])/n_samples:.2f} ms")
                print(f"  Optimizer step:     {sum(stats['optim_times'])/n_samples:.2f} ms")
                print(f"  Total update:       {sum(stats['total_times'])/n_samples:.2f} ms")
                print(f"  Avg steps/sec:      {1000.0/(sum(stats['total_times'])/n_samples):.2f}")
                print("-" * 50)
            
            # Reset stats for next minute
            for key in stats:
                if key != 'last_print_time':
                    stats[key] = []
            stats['last_print_time'] = current_time

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    # Pass episode_data_index for RLearN to calculate proper progress
    episode_data_index = dataset.episode_data_index if hasattr(dataset, "episode_data_index") else None
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        episode_data_index=episode_data_index,
    )
    preprocessor, postprocessor = make_processor(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path, dataset_stats=dataset.meta.stats
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

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
        persistent_workers=cfg.num_workers > 0,  # Keep workers alive
        prefetch_factor=2 if cfg.num_workers > 0 else None,  # Prefetch batches
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    # RLearN-only: pixels per second throughput
    try:
        if getattr(policy, "name", None) == "rlearn":
            train_metrics["pix_s"] = AverageMeter("pix/s", ":.1f")
    except Exception:
        pass

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        # Data loading timing
        data_start = time.perf_counter()
        batch = next(dl_iter)
        data_loading_time = time.perf_counter() - data_start
        
        # Preprocessing timing  
        preprocess_start = time.perf_counter()
        batch = preprocessor(batch)
        preprocess_time = time.perf_counter() - preprocess_start
        
        train_tracker.dataloading_s = data_loading_time + preprocess_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # RLearN-only: compute pixel throughput (pixels per second)
        if getattr(policy, "name", None) == "rlearn":
            def _count_pixels(x: torch.Tensor) -> int:
                # Expect shapes: (B,T,C,H,W) or (B,C,H,W)
                if x.dim() == 5:
                    b, t, _, h, w = x.shape
                    return int(b * t * h * w)
                if x.dim() == 4:
                    b, _, h, w = x.shape
                    return int(b * h * w)
                return 0

            total_pixels = 0
            for k, v in batch.items():
                if "image" not in k.lower():
                    continue
                if isinstance(v, torch.Tensor):
                    total_pixels += _count_pixels(v)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    # list of T tensors shaped (B,C,H,W)
                    total_pixels += sum(_count_pixels(t) for t in v)

            # Avoid div-by-zero
            meter = train_tracker.update_s
            upd_s = meter.val if isinstance(meter, AverageMeter) else float(meter)
            upd_s = max(upd_s, 1e-8)
            pix_per_s = float(total_pixels) / upd_s
            try:
                train_tracker.pix_s = pix_per_s
            except Exception:
                pass

        # Collect data pipeline timing for RLearN (averaged reporting every minute)
        if getattr(policy, "name", None) == "rlearn":
            # Initialize data timing accumulator if not exists
            if not hasattr(policy, '_data_timing_stats'):
                policy._data_timing_stats = {
                    'data_loading_times': [],
                    'preprocess_times': [],
                    'last_print_time': time.perf_counter()
                }
            
            # Accumulate current step's data timings
            data_stats = policy._data_timing_stats
            data_stats['data_loading_times'].append(data_loading_time * 1000)
            data_stats['preprocess_times'].append(preprocess_time * 1000)
            
            # Print averaged stats every minute (60 seconds)  
            current_time = time.perf_counter()
            if current_time - data_stats['last_print_time'] >= 60.0:
                n_samples = len(data_stats['data_loading_times'])
                if n_samples > 0:
                    avg_data_loading = sum(data_stats['data_loading_times']) / n_samples
                    avg_preprocessing = sum(data_stats['preprocess_times']) / n_samples
                    
                    print(f"\nData Pipeline Average Timing (last {n_samples} steps):")
                    print(f"  Data loading:       {avg_data_loading:.2f} ms")
                    print(f"  Preprocessing:      {avg_preprocessing:.2f} ms") 
                    print(f"  Total data pipeline: {avg_data_loading + avg_preprocessing:.2f} ms")
                    print("-" * 50)
                
                # Reset stats for next minute
                for key in data_stats:
                    if key != 'last_print_time':
                        data_stats[key] = []
                data_stats['last_print_time'] = current_time
            
        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
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

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)
        if preprocessor:
            preprocessor.push_to_hub(cfg.policy.repo_id)
        if postprocessor:
            postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
