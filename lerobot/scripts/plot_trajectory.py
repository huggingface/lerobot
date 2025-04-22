#!/usr/bin/env python
"""
plot_trajectory.py
==================
• plot_trajectory_comparison() —— 画单条轨迹 (xyz) 的 GT vs Pred
• plot_epoch_trajectories()    —— 随机抽 num_samples 条轨迹绘制并返回图片路径
• get_predicted_action()       —— 专为 π0 设计的动作提取器
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (必需请勿删除)

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters


# ============================== 绘图工具 ================================= #
def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def plot_trajectory_comparison(
    gt: torch.Tensor | np.ndarray,
    pred: torch.Tensor | np.ndarray,
    title: str = "Trajectory Comparison (xyz only)",
    save_path: str | Path | None = None,
):
    """画单条 xyz 轨迹对比"""
    gt, pred = _to_numpy(gt), _to_numpy(pred)
    if gt.ndim == 3:  # (B,T,3)
        gt = gt[0]
    if pred.ndim == 3:
        pred = pred[0]

    gt, pred = gt[:, :3], pred[:, :3]  # 只留 xyz

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "b-", label="Ground Truth", linewidth=2)
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "r--", label="Predicted", linewidth=2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title); ax.legend(); ax.view_init(elev=20, azim=45)

    all_pts = np.vstack([gt, pred])
    pad = 0.1 * (all_pts.max(0) - all_pts.min(0))
    ax.set_xlim(all_pts[:, 0].min() - pad[0], all_pts[:, 0].max() + pad[0])
    ax.set_ylim(all_pts[:, 1].min() - pad[1], all_pts[:, 1].max() + pad[1])
    ax.set_zlim(all_pts[:, 2].min() - pad[2], all_pts[:, 2].max() + pad[2])

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_epoch_trajectories(
    gt_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    save_dir: str | Path,
    epoch: int,
    num_samples: int = 3,
) -> List[Path]:
    """
    随机抽 num_samples 条样本绘图，返回图片文件列表。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gt_np, pred_np = _to_numpy(gt_batch), _to_numpy(pred_batch)
    idxs = random.sample(range(gt_np.shape[0]), min(num_samples, gt_np.shape[0]))

    paths: List[Path] = []
    for i, idx in enumerate(idxs):
        path = save_dir / f"trajectory_epoch_{epoch}_sample_{i}.png"
        plot_trajectory_comparison(gt_np[idx], pred_np[idx],
                                   title=f"Epoch {epoch} – Sample {i}",
                                   save_path=path)
        paths.append(path)
    return paths


# ========================== π0 动作提取工具 ============================ #
def _ensure_bt_d(a: torch.Tensor) -> torch.Tensor:
    """把 (B,D) → (B,1,D)，保持 (B,T,D) 不变。"""
    return a.unsqueeze(1) if a.dim() == 2 else a


def get_predicted_action(
    policy: PreTrainedPolicy,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """
    返回 π0 的预测动作 (B,T,D)。优先用 predict_action()，
    否则 fallback 到 sample_actions()；失败则返回 None。
    """
    device = get_device_from_parameters(policy)

    # 1) predict_action ----------------------------------------------------
    if hasattr(policy, "predict_action"):
        try:
            out = policy.predict_action({"obs": batch["obs"]})
            if isinstance(out, tuple):
                out = out[1] if len(out) > 1 else out[0]
            if isinstance(out, dict):
                act = out.get("action_pred") or out.get("action")
                if isinstance(act, torch.Tensor):
                    return _ensure_bt_d(act).to(device)
        except Exception as e:
            print(f"[DEBUG] predict_action failed: {e}")

    # 2) sample_actions ----------------------------------------------------
    try:
        imgs, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks = policy.prepare_language(batch)

        acts = policy.model.sample_actions(imgs, img_masks, lang_tokens, lang_masks, state)
        act_dim = policy.config.action_feature.shape[0]
        acts = acts[:, :, :act_dim]                                # 去 padding
        acts = policy.unnormalize_outputs({"action": acts})["action"]
        return acts.to(device)
    except Exception as e:
        print(f"[DEBUG] sample_actions failed: {e}")

    # 3) 未获取到动作 ------------------------------------------------------
    return None
