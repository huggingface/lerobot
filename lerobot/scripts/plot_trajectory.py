#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数：可画单条或一批 xyz 轨迹对比图，并返回图片路径。
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def _plot_one(gt_xyz: np.ndarray, pred_xyz: np.ndarray, title: str, save_path: Path):
    """内部函数：真正画一张 3D 轨迹图."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GT", linestyle="-")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Pred", linestyle="--")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title(title), ax.legend()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_trajectories(
    gt_xyz_all: torch.Tensor,         # (B, T, 3)
    pred_xyz_all: torch.Tensor,       # (B, T, 3)
    save_dir: Path,
    epoch_idx: int,
    n_samples: int = 3,
) -> list[Path]:
    """
    取前 n_samples 条序列，分别画 GT 与 Pred 对比图，返回图片路径列表。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(min(n_samples, gt_xyz_all.shape[0])):
        gt_np   = gt_xyz_all[i].cpu().numpy()
        pred_np = pred_xyz_all[i].cpu().numpy()
        title   = f"epoch_{epoch_idx}_sample_{i}"
        out_p   = save_dir / f"{title}.png"
        _plot_one(gt_np, pred_np, title, out_p)
        paths.append(out_p)

    return paths
