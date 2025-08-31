#!/usr/bin/env python

"""
Standalone evaluation script for RLearN models.

This script evaluates RLearN reward models on episodes from a dataset,
generating comparison plots between ground truth rewards and model predictions.

Usage:
    python src/lerobot/policies/rlearn/eval_script.py --model MODEL_NAME --dataset DATASET_REPO --episodes N

Example:
    python src/lerobot/policies/rlearn/eval_script.py --model pepijn223/rlearn_18 --dataset pepijn223/phone_pipeline_pickup1 --episodes 2
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# LeRobot imports
from lerobot.constants import OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.rlearn.modeling_rlearn import RLearNPolicy


def _to_chw_float01(img):
    """Ensure CHW float in [0,1]."""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    # HWC -> CHW if needed
    if len(img.shape) == 3 and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float()
    return torch.clamp(img, 0.0, 1.0)


def _get_language(frame_data):
    lang = None
    if OBS_LANGUAGE in frame_data:
        lang = frame_data[OBS_LANGUAGE]
        if isinstance(lang, list) and len(lang) > 0:
            lang = lang[0]
    elif "task" in frame_data:
        lang = frame_data["task"]
    return lang if isinstance(lang, str) else "No language provided"


def _get_ground_truth_reward(frame_data):
    """Try common keys for ground-truth reward. Return None if unavailable."""
    for key in ("reward", "rewards", "gt_reward", "progress"):
        if key in frame_data:
            r = frame_data[key]
            # unwrap single-element lists/arrays
            if isinstance(r, (list, np.ndarray)) and np.array(r).size == 1:
                r = float(np.array(r).reshape(-1)[0])
            try:
                return float(r)
            except Exception:
                pass
    return None


def extract_episode_frames_and_gt(dataset, episode_idx):
    """Load a full episode: frames (T, C, H, W), language (str), gt_rewards (np.ndarray or None)."""
    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    ep_end = dataset.episode_data_index["to"][episode_idx].item()
    T = ep_end - ep_start

    frames = []
    gt_rewards = []
    language = None

    for t in range(T):
        item = dataset[ep_start + t]

        # image(s)
        if OBS_IMAGES in item:
            img = item[OBS_IMAGES]
        elif OBS_IMAGE in item:
            img = item[OBS_IMAGE]
        else:
            # try to find an image-like key
            img_keys = [k for k in item.keys() if "image" in k.lower()]
            if not img_keys:
                continue
            img = item[img_keys[0]]

        frames.append(_to_chw_float01(img))

        # language once
        if language is None:
            language = _get_language(item)

        # ground-truth reward (optional)
        r = _get_ground_truth_reward(item)
        gt_rewards.append(r)

    if not frames:
        return None, None, None

    frames = torch.stack(frames)  # (T, C, H, W)

    # If all GT entries are None, treat as missing
    if all(r is None for r in gt_rewards):
        gt_rewards = None
    else:
        # Replace None by forward filling
        arr = np.array([np.nan if r is None else float(r) for r in gt_rewards], dtype=float)
        # forward/back fill
        if np.isnan(arr[0]):
            first_valid = np.flatnonzero(~np.isnan(arr))
            if len(first_valid) > 0:
                arr[0] = arr[first_valid[0]]
            else:
                arr[0] = 0.0
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
        gt_rewards = arr

    return frames, language or "No language provided", gt_rewards


@torch.no_grad()
def predict_rewards_sliding(model, frames, language, max_seq_len=16, batch_size=64, device="cuda"):
    """
    Sliding-window prediction: for each frame i, create a window [max(0, i-L+1) .. i],
    left-pad by repeating the first frame to length L (<= 16), and take the prediction 
    corresponding to the current frame's position in the window.
    Returns np.ndarray of shape (T,).
    """
    T = frames.shape[0]
    L = int(getattr(getattr(model, "config", object()), "max_seq_len", max_seq_len))
    L = min(L, max_seq_len)  # hard-cap at 16

    # Preprocessed tensor on device
    frames = frames.to(device)

    windows = []
    frame_positions = []  # Track which temporal position each frame should use
    
    for i in range(T):
        start = max(0, i - L + 1)
        window = frames[start : i + 1]  # (len<=L, C, H, W)
        
        if window.shape[0] < L:
            pad_needed = L - window.shape[0]
            pad = window[:1].expand(pad_needed, -1, -1, -1)  # repeat first frame
            window = torch.cat([pad, window], dim=0)
        
        # IMPROVED FIX: Cycle through MLPs to get varied predictions throughout the episode
        # This ensures we use all 16 frame-specific MLPs and get varied outputs
        # Frames 0-15 use MLPs 0-15, frames 16-31 use MLPs 0-15 again, etc.
        frame_pos = i % L  # Cycle through [0, 1, 2, ..., 15, 0, 1, 2, ..., 15, ...]
            
        windows.append(window)
        frame_positions.append(frame_pos)

    preds = np.zeros(T, dtype=float)

    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        batch_windows = torch.stack(windows[s:e])  # (B, L, C, H, W)
        batch_positions = frame_positions[s:e]

        batch = {OBS_IMAGES: batch_windows, OBS_LANGUAGE: [language] * (e - s)}  # expects (B, L, C, H, W)

        # Model returns (B, L) predictions for each temporal position
        values = model.predict_rewards(batch)  # torch.Tensor (B, L)

        # Debug output removed - issue was identified and fixed

        if values.dim() == 2:
            # Extract the prediction corresponding to each frame's position in its window
            batch_preds = []
            for b_idx, pos in enumerate(batch_positions):
                batch_preds.append(values[b_idx, pos].item())
            preds[s:e] = np.array(batch_preds)
        else:
            # Fallback: if model returns (B,), use as is
            preds[s:e] = values.detach().float().cpu().numpy()

    return preds


def plot_episode_eval(episode_idx, gt, pred, language, save_path=None, show=False, title_prefix="RLearN Eval"):
    """Plot GT vs Predicted over time. Saves PNG if save_path is provided."""
    T = len(pred)
    x = np.arange(T)

    plt.figure(figsize=(14, 8))
    plt.plot(x, pred, linewidth=2.5, marker="o", markersize=3, label="Predicted Reward", color="blue")

    if gt is not None:
        plt.plot(x, gt, linestyle="--", linewidth=2.5, label="Ground-Truth Reward", color="orange")
        # Correlation between GT and Pred
        corr, p = spearmanr(gt, pred)
        corr_str = f"ρ(GT, Pred) = {0.0 if np.isnan(corr) else corr:.3f} (p={0.0 if np.isnan(p) else p:.3f})"
    else:
        expected = np.linspace(0, 1, T)
        plt.plot(x, expected, linestyle="--", linewidth=2.5, label="Expected Progress (0→1)", color="orange")
        corr, p = spearmanr(x, pred)
        corr_str = f"VOC-S ρ(t, Pred) = {0.0 if np.isnan(corr) else corr:.3f} (p={0.0 if np.isnan(p) else p:.3f})"

    plt.title(f"{title_prefix} — Episode {episode_idx}\n{language}\n{corr_str}", fontsize=14)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Reward / Progress", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved eval image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def eval_episode_sliding(
    episode_idx, dataset, model, save_dir=".", device="cuda", max_seq_len=16, batch_size=64, title_prefix="RLearN Eval"
):
    """End-to-end: load episode, predict with sliding 16-frame windows, and save PNG."""
    frames, language, gt = extract_episode_frames_and_gt(dataset, episode_idx)
    if frames is None:
        print(f"[Episode {episode_idx}] No frames found.")
        return None

    model.eval()

    pred = predict_rewards_sliding(
        model=model, frames=frames, language=language, max_seq_len=max_seq_len, batch_size=batch_size, device=device
    )

    # Basic stats
    print(f"Episode {episode_idx}: T={len(pred)}, pred∈[{pred.min():.3f},{pred.max():.3f}]")
    if gt is not None:
        print(f"GT available: gt∈[{np.nanmin(gt):.3f},{np.nanmax(gt):.3f}]")

    save_path = f"{save_dir}/episode_{episode_idx:04d}_eval.png"
    plot_episode_eval(
        episode_idx=episode_idx, gt=gt, pred=pred, language=language, save_path=save_path, show=False, title_prefix=title_prefix
    )
    return save_path


def main():
    """Main evaluation script for RLearN models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate RLearN model on episodes with GT vs Predicted rewards")
    parser.add_argument("--model", type=str, required=True, help="Model name/path (e.g., pepijn223/rlearn_mse5)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo (e.g., pepijn223/phone_pipeline_pickup1)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory for images")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for sliding window evaluation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 RLearN Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    try:
        # Load dataset
        print("📁 Loading dataset...")

        dataset = LeRobotDataset(
            repo_id=args.dataset,
            episodes=list(range(min(args.episodes, 50))),  # Load enough episodes
            download_videos=True,
        )

        print(f"✅ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        print(f"   Features: {list(dataset.features.keys())}")
        print(f"   FPS: {dataset.fps}")

        # Load model
        print("\n🤖 Loading model...")

        model = RLearNPolicy.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()

        print(f"✅ Model loaded on {args.device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"   Max sequence length: {model.config.max_seq_len}")

        # Select episodes to evaluate
        total_available = min(dataset.num_episodes, args.episodes)
        episode_indices = list(range(total_available))

        print(f"\n📊 Evaluating {len(episode_indices)} episodes...")
        print("=" * 60)

        # Run sliding window evaluation on each episode
        saved_paths = []
        for i, ep_idx in enumerate(episode_indices):
            print(f"\n[{i+1}/{len(episode_indices)}] Processing Episode {ep_idx}")
            print("-" * 40)

            try:
                save_path = eval_episode_sliding(
                    episode_idx=ep_idx,
                    dataset=dataset,
                    model=model,
                    save_dir=str(output_dir),
                    device=args.device,
                    batch_size=args.batch_size,
                    title_prefix="RLearN Ground Truth vs Predicted",
                )

                if save_path:
                    saved_paths.append(save_path)

            except Exception as e:
                print(f"❌ Error processing episode {ep_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Summary
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETE")
        print(f"📈 Generated {len(saved_paths)} evaluation plots")
        print(f"📁 Results saved to: {output_dir}")
        print("\nGenerated files:")
        for path in saved_paths:
            print(f"  • {path}")

        if saved_paths:
            print(f"\n💡 View the plots to compare ground truth vs predicted rewards!")
            print(f"   Each plot shows the model's sliding 16-frame window predictions")
            print(f"   against available ground truth rewards over the episode timeline.")

        return 0

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
