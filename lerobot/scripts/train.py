#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2024‑2025 …

import logging, time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, List

import torch
import torch.nn.functional as F
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.constants import ACTION
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
from lerobot.scripts.plot_trajectory import plot_epoch_trajectories


# ─────────────────────────────────────────
# 辅助函数：batch 级 action‑MSE（仅 xyz）
# ─────────────────────────────────────────
def compute_action_mse(policy: PreTrainedPolicy, batch: dict, device: torch.device) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=policy.config.use_amp):
        gt = batch[ACTION].to(device)[:, : policy.config.n_action_steps, :3]
        pred = policy.predict_actions_batch(batch)[:, : policy.config.n_action_steps, :3]
        return F.mse_loss(pred, gt, reduction="mean")


# ─────────────────────────────────────────
# update_policy（与官方实现一致）
# ─────────────────────────────────────────
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
):
    start = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, _ = policy.forward(batch)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start
    return train_metrics


# ─────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # ---------- WandB ----------
    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
    if wandb_logger is None:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # ---------- 随机数 & 设备 ----------
    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---------- 数据 ----------
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # ---------- Policy ----------
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    # ---------- Optim ----------
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    # ---------- Eval env (可选) ----------
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # ---------- 断点恢复 ----------
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # ---------- Dataloader ----------
    shuffle = not hasattr(cfg.policy, "drop_n_last_frames")
    sampler = None
    if not shuffle:
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    epoch_size = len(dataloader)

    # ---------- Metrics ----------
    meters = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "action_mse": AverageMeter("mse", ":.4f"),
    }
    tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, meters, initial_step=step)

    # ---------- 轨迹缓存 ----------
    traj_pool: List[tuple[torch.Tensor, torch.Tensor]] = []
    epoch_idx = step // epoch_size

    logging.info("Start offline training")
    while step < cfg.steps:
        # ================= batch =================
        t0 = time.perf_counter()
        batch = next(dl_iter)
        tracker.dataloading_s = time.perf_counter() - t0
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        tracker = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler,
            lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # ---------- action‑MSE ----------
        mse_val = compute_action_mse(policy, batch, device)
        tracker.metrics["action_mse"].update(mse_val.item(), n=1)
        tracker.action_mse = mse_val.item()

        # ---------- 轨迹样本 ----------
        if len(traj_pool) < 3:
            gt_xyz   = batch[ACTION][:, : policy.config.n_action_steps, :3].detach().cpu()
            pred_xyz = policy.predict_actions_batch(batch)[:, : policy.config.n_action_steps, :3].detach().cpu()
            traj_pool.append((gt_xyz[0], pred_xyz[0]))

        # ========== logging ==========
        step += 1
        tracker.step()
        if cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info(tracker)
            if wandb_logger:
                wandb_logger.log_dict(tracker.to_dict(), step)
            tracker.reset_averages()

        # ========== epoch finished ==========
        # if step % epoch_size == 0:
        TRJ_FREQ = 100   # 想多快画一次就写多小
        if step % TRJ_FREQ == 0:
            paths = plot_epoch_trajectories(
                gt_xyz_all=torch.stack([p[0] for p in traj_pool]),
                pred_xyz_all=torch.stack([p[1] for p in traj_pool]),
                save_dir=cfg.output_dir / "trajectory_plots",
                epoch_idx=epoch_idx,
            )
            if wandb_logger:
                wandb_logger.log_images(paths, step, caption=f"epoch_{epoch_idx}_trajectories")
            traj_pool.clear()
            epoch_idx += 1

        # ========== checkpoint ==========
        if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
            ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(ckpt_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(ckpt_dir)
            if wandb_logger:
                wandb_logger.log_policy(ckpt_dir)

        # ========== evaluation ==========
        if eval_env and cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=cfg.policy.use_amp):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_meters = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("succ", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(cfg.batch_size, 0, 0, eval_meters, initial_step=step)
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success     = eval_info["aggregated"].pop("pc_success")
            eval_tracker.eval_s         = eval_info["aggregated"].pop("eval_s")

            logging.info(eval_tracker)
            if wandb_logger:
                wandb_logger.log_dict({**eval_tracker.to_dict(), **eval_info}, step, mode="eval")
                if eval_info["video_paths"]:
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # ---------- 结束 ----------
    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
