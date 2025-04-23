from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def _plot_one(gt_xyz: np.ndarray, pred_xyz: np.ndarray, title: str, save_path: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GroundTruth", linestyle="-")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", linestyle="--")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title); ax.legend()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_epoch_trajectories(
    gt_all: torch.Tensor,             # (B, T, D)
    pred_all: torch.Tensor,           # (B, T, D)
    save_dir: Path,
    index: int,
    naming: str = 'epoch',            # 'epoch' or 'step'
    one_hand: bool = False,
    rotation: bool = False,
    n_samples: int = 3,
) -> list[Path]:
    """
    Plot for each sample & each hand:
      - 3D XYZ trajectory
      - (optionally) rotation channels vs time
      - 2D gripper opening vs time

    Args:
      gt_all:   (B,T,D) ground truth action sequences
      pred_all: (B,T,D) predicted action sequences
      naming:   prefix mode for filenames
      one_hand: True = treat D as one hand; False = split into two hands
      rotation: True = also plot rotation channels
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    B, T, D = gt_all.shape
    paths: list[Path] = []

    # decide number of hands and per-hand dims
    n_hands = 1 if one_hand else 2
    if D % n_hands != 0:
        raise ValueError(f"Cannot split D={D} into {n_hands} hand(s)")
    per_hand = D // n_hands

    prefix = 'epoch' if naming == 'epoch' else 'step'

    for hand_idx in range(n_hands):
        start = hand_idx * per_hand
        end = start + per_hand

        for i in range(min(n_samples, B)):
            gt_np   = gt_all[i, :, start:end].cpu().numpy()
            pred_np = pred_all[i, :, start:end].cpu().numpy()

            # --- XYZ plot ---
            title_xyz = f"{prefix}_{index}_sample_{i}_hand{hand_idx}_xyz"
            out_xyz   = save_dir / f"{title_xyz}.png"
            _plot_one(gt_np[:, :3], pred_np[:, :3], title_xyz, out_xyz)
            paths.append(out_xyz)

            # --- Rotation plot ---
            if rotation:
                rot_dim = per_hand - 3 - 1  # remainder after x/y/z and gripper
                if rot_dim <= 0:
                    raise ValueError(f"No rotation channels to plot (per_hand={per_hand})")
                # extract rotation channels
                gt_rot   = gt_np[:, 3:3+rot_dim]
                pred_rot = pred_np[:, 3:3+rot_dim]
                # 2D line plot of each rotation channel vs time
                fig, ax = plt.subplots()
                for c in range(rot_dim):
                    ax.plot(gt_rot[:, c],   label=f"GroundTruth_Right{c}")
                    ax.plot(pred_rot[:, c], linestyle='--', label=f"Prediction_Right{c}")
                ax.set_title(f"{prefix}_{index}_sample_{i}_hand{hand_idx}_rot")
                ax.legend()
                out_rot = save_dir / f"{prefix}_{index}_sample_{i}_hand{hand_idx}_rot.png"
                fig.savefig(out_rot, dpi=200, bbox_inches="tight")
                plt.close(fig)
                paths.append(out_rot)

            # --- Gripper plot (last channel) ---
            gt_grip   = gt_np[:, -1]
            pred_grip = pred_np[:, -1]
            fig, ax = plt.subplots()
            ax.plot(gt_grip,   label="GroundTruth_gripper")
            ax.plot(pred_grip, linestyle='--', label="Prediction_gripper")
            ax.set_title(f"{prefix}_{index}_sample_{i}_hand{hand_idx}_gripper")
            ax.legend()
            out_grip = save_dir / f"{prefix}_{index}_sample_{i}_hand{hand_idx}_gripper.png"
            fig.savefig(out_grip, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(out_grip)

    return paths
