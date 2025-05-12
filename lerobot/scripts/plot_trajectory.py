from __future__ import annotations

"""Utility functions for visualising action trajectories.

This module supersedes the previous `plot_trajectory.py` by automatically
handling two common action encodings:

* **Pose‐based**: `(x, y, z, rot..., gripper)` per hand (≥8 dim typically)
* **Joint‐based**: `n_joints` joint positions (6 or 7 dim usually)

It aims to remain *plug‑and‑play* – you can drop it into other workspaces
without touching the rest of your pipeline.  Simply import the top‑level
:func:`plot_epoch_trajectories` in place of the old implementation.

Example
-------
>>> from pathlib import Path
>>> import torch
>>> gt, pred = torch.randn(4, 10, 14), torch.randn(4, 10, 14)  # (B,T,D)
>>> out_paths = plot_epoch_trajectories(gt, pred, Path('tmp/plots'), index=1)
"""

from pathlib import Path
from typing import List, Literal, Tuple
from __future__ import annotations

"""Utility functions for visualising action trajectories.

This module supersedes the previous `plot_trajectory.py` by automatically
handling two common action encodings:

* **Pose‐based**: `(x, y, z, rot..., gripper)` per hand (≥8 dim typically)
* **Joint‐based**: `n_joints` joint positions (6 or 7 dim usually)

It aims to remain *plug‑and‑play* – you can drop it into other workspaces
without touching the rest of your pipeline.  Simply import the top‑level
:func:`plot_epoch_trajectories` in place of the old implementation.

Example
-------
>>> from pathlib import Path
>>> import torch
>>> gt, pred = torch.randn(4, 10, 14), torch.randn(4, 10, 14)  # (B,T,D)
>>> out_paths = plot_epoch_trajectories(gt, pred, Path('tmp/plots'), index=1)
"""

from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch
import matplotlib.pyplot as plt

__all__ = [
    "plot_epoch_trajectories",
]

###############################################################################
# Low‑level helpers
###############################################################################


def _plot_xyz(
    gt_xyz: np.ndarray, pred_xyz: np.ndarray, title: str, save_path: Path
) -> None:
    """3‑D XYZ trajectory plot."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GroundTruth", linestyle="-")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_rotations(
    gt_rot: np.ndarray,
    pred_rot: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot each rotation channel as a separate line on a common axis."""

    fig, ax = plt.subplots()
    rot_dim = gt_rot.shape[1]
    for c in range(rot_dim):
        ax.plot(gt_rot[:, c], label=f"GT_R{c}")
        ax.plot(pred_rot[:, c], linestyle="--", label=f"Pred_R{c}")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.legend(ncol=2, fontsize="small")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gripper(gt: np.ndarray, pred: np.ndarray, title: str, save_path: Path) -> None:
    """Plot gripper open value over time."""

    fig, ax = plt.subplots()
    ax.plot(gt, label="GT_grip")
    ax.plot(pred, linestyle="--", label="Pred_grip")

__all__ = [
    "plot_epoch_trajectories",
]

###############################################################################
# Low‑level helpers
###############################################################################


def _plot_xyz(
    gt_xyz: np.ndarray, pred_xyz: np.ndarray, title: str, save_path: Path
) -> None:
    """3‑D XYZ trajectory plot."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GroundTruth", linestyle="-")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_rotations(
    gt_rot: np.ndarray,
    pred_rot: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot each rotation channel as a separate line on a common axis."""

    fig, ax = plt.subplots()
    rot_dim = gt_rot.shape[1]
    for c in range(rot_dim):
        ax.plot(gt_rot[:, c], label=f"GT_R{c}")
        ax.plot(pred_rot[:, c], linestyle="--", label=f"Pred_R{c}")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.legend(ncol=2, fontsize="small")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gripper(gt: np.ndarray, pred: np.ndarray, title: str, save_path: Path) -> None:
    """Plot gripper open value over time."""

    fig, ax = plt.subplots()
    ax.plot(gt, label="GT_grip")
    ax.plot(pred, linestyle="--", label="Pred_grip")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_xlabel("t")
    ax.legend()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_joints(
    gt_joint: np.ndarray, pred_joint: np.ndarray, title: str, save_path: Path
) -> None:
    """Plot *n_joint* joint trajectories in a multi‑row figure.

    Parameters
    ----------
    gt_joint, pred_joint
        Arrays of shape ``(T, n_joints)``.
    title, save_path
        Obvious.
    """

    n_joints = gt_joint.shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(6, 1.8 * n_joints), sharex=True)
    if n_joints == 1:  # when n_joints == 1 matplotlib returns a single Axes
        axes = [axes]
    time = np.arange(gt_joint.shape[0])
    for j, ax in enumerate(axes):
        ax.plot(time, gt_joint[:, j], label="GT", linewidth=1.3)
        ax.plot(time, pred_joint[:, j], label="Pred", linewidth=1.1, linestyle="--")
        ax.set_ylabel(f"J{j}")
        ax.grid(True, linestyle=":", linewidth=0.5)
        if j == 0:
            ax.set_title(title)
        if j == n_joints - 1:
            ax.set_xlabel("t")
    axes[0].legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

###############################################################################
# Top‑level API
###############################################################################


def _decide_mode(per_hand: int, rotation_flag: bool | None) -> Literal["pose", "joints"]:
    """Heuristic for deciding whether we are dealing with pose or joints.

    * If the user explicitly set *rotation* (not None) we respect it.  When
      *rotation* is True we assume pose data.
    * Otherwise we assume **joints** if ``per_hand`` is 6 or 7 (common for most
      arms).  Everything else defaults to **pose**.
    """

    if rotation_flag is not None:
        return "pose" if rotation_flag else "joints"
    return "joints" if per_hand in {6, 7} else "pose"


@torch.no_grad()
def plot_epoch_trajectories(
    gt_all: torch.Tensor,  # (B, T, D)
    pred_all: torch.Tensor,  # (B, T, D)
    save_dir: Path,
    index: int,
    *,
    naming: Literal["epoch", "step"] = "epoch",
    one_hand: bool = False,
    rotation: bool | None = None,
    n_samples: int = 3,
) -> List[Path]:
    """Generate trajectory plots for a subset of samples.

    The function now seamlessly supports *pose* **and** *joint* encodings – it
    will *auto‑detect* the proper mode unless you explicitly specify the
    ``rotation`` flag.

    Returns
    -------
    list[Path]
        Filepaths of the saved figures.
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    B, T, D = gt_all.shape
    paths: List[Path] = []

    # Decide how many hands we have and per‑hand dimensionality
    n_hands = 1 if one_hand else 2
    if D % n_hands != 0:
        raise ValueError(f"Cannot split D={D} into {n_hands} hand(s)")
    per_hand = D // n_hands

    mode = _decide_mode(per_hand, rotation)

    prefix = "epoch" if naming == "epoch" else "step"

    for hand_idx in range(n_hands):
        start, end = hand_idx * per_hand, (hand_idx + 1) * per_hand
        for i in range(min(n_samples, B)):
            gt_np = gt_all[i, :, start:end].cpu().numpy()
            pred_np = pred_all[i, :, start:end].cpu().numpy()

            if mode == "pose":
                # ── XYZ ────────────────────────────────────────────────
                title_xyz = f"{prefix}_{index}_sample_{i}_hand{hand_idx}_xyz"
                out_xyz = save_dir / f"{title_xyz}.png"
                _plot_xyz(gt_np[:, :3], pred_np[:, :3], title_xyz, out_xyz)
                paths.append(out_xyz)

                # ── Rotation (optional) ───────────────────────────────
                rot_dim = per_hand - 3 - 1  # minus xyz and gripper
                if rot_dim > 0:
                    title_rot = f"{prefix}_{index}_sample_{i}_hand{hand_idx}_rot"
                    out_rot = save_dir / f"{title_rot}.png"
                    _plot_rotations(gt_np[:, 3 : 3 + rot_dim], pred_np[:, 3 : 3 + rot_dim], title_rot, out_rot)
                    paths.append(out_rot)

                # ── Gripper ───────────────────────────────────────────
                title_grip = f"{prefix}_{index}_sample_{i}_hand{hand_idx}_gripper"
                out_grip = save_dir / f"{title_grip}.png"
                _plot_gripper(gt_np[:, -1], pred_np[:, -1], title_grip, out_grip)
                paths.append(out_grip)

            else:  # mode == "joints"
                title_joint = f"{prefix}_{index}_sample_{i}_hand{hand_idx}_joint"
                out_joint = save_dir / f"{title_joint}.png"
                _plot_joints(gt_np, pred_np, title_joint, out_joint)
                paths.append(out_joint)

    return paths