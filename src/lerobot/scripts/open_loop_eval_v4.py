"""Open-loop evaluation script using standard LeRobot inference pipeline.

This script evaluates policies on LeRobot datasets by following the SAME code path
as lerobot_eval.py and lerobot_record.py (inference mode). This ensures:
- State normalization is handled correctly by the preprocessor
- Action unnormalization is handled correctly by the postprocessor
- All processing steps match training exactly

Key difference from open_loop_eval_v3.py:
- Uses `make_pre_post_processors()` to create the same pipelines used in training/eval
- Uses `preprocessor(batch)` before policy inference
- Uses `postprocessor(action)` after policy inference
- Does NOT manually construct batches or call decode_action directly

Usage:
------
python src/lerobot/scripts/open_loop_eval_v4.py \
    --dataset-repo-id=izuluaga/finish_sandwich \
    --policy-repo-id=aravindhs-NV/finish_sandwich_fix_processor_0114_bs32 \
    --episode-ids 0 1 2 \
    --save-dir=./open_loop_eval_v4_outputs \
    --steps=200 \
    --inference-interval=10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

# Reuse plotting function from open_loop_eval_v3
from lerobot.scripts.open_loop_eval_v3 import plot_trajectory_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_episode_frames(dataset: LeRobotDataset, episode_id: int) -> list[dict[str, Any]]:
    """Get all frames for a specific episode from LeRobotDataset."""
    dataset._ensure_hf_dataset_loaded()

    episode_indices = []
    for idx in range(len(dataset.hf_dataset)):
        if dataset.hf_dataset[idx]["episode_index"] == episode_id:
            episode_indices.append(idx)

    episode_frames = []
    for idx in episode_indices:
        try:
            frame = dataset[idx]
            episode_frames.append(frame)
        except Exception as e:
            logger.warning(f"Error loading frame {idx} from episode {episode_id}: {e}")
            continue

    logger.info(f"Found {len(episode_frames)} frames for episode {episode_id}")
    return episode_frames


def evaluate_trajectory_with_pipeline(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: LeRobotDataset,
    episode_id: int,
    steps: int = 200,
    inference_interval: int = 10,
    save_plot_path: str | None = None,
) -> tuple[float, float]:
    """
    Evaluate a trajectory using the standard LeRobot inference pipeline.
    
    This follows the same code path as lerobot_eval.py:
    1. preprocessor(observation) - handles state normalization
    2. policy.select_action(processed_obs) - returns normalized action
    3. postprocessor(action) - handles unnormalization
    """
    # Get all frames for this episode
    episode_frames = get_episode_frames(dataset, episode_id)
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    traj_length = len(episode_frames)
    actual_steps = min(steps, traj_length)
    logger.info(f"Evaluating {actual_steps} steps (trajectory length: {traj_length})")

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Get action dimension from dataset
    action_key = "action"
    action_dim = (
        dataset.meta.features.get(action_key, {}).get("shape", [0])[0]
        if action_key in dataset.meta.features
        else 0
    )
    state_key = "observation.state"
    state_dim = (
        dataset.meta.features.get(state_key, {}).get("shape", [0])[0]
        if state_key in dataset.meta.features
        else 0
    )

    logger.info(f"Action dim: {action_dim}, State dim: {state_dim}")
    logger.info(f"Inference interval: {inference_interval} steps")

    # Collect predictions and ground truth
    pred_actions = []
    gt_actions = []
    state_joints = []

    device = policy.device

    for step_count in range(0, actual_steps, inference_interval):
        if step_count >= len(episode_frames):
            break

        logger.info(f"Inferencing at step: {step_count}")

        # Get frame data
        frame = episode_frames[step_count]

        # Add task to observation (same pattern as lerobot_eval.py add_envs_task)
        observation = dict(frame)  # Copy frame
        
        # Ensure task is present
        if "task" not in observation:
            observation["task"] = "complete the task"

        # Apply preprocessor (handles state normalization, VLM processing, etc.)
        # This is the key step that was MISSING in open_loop_eval_v3!
        try:
            processed_obs = preprocessor(observation)
        except Exception as e:
            logger.error(f"Preprocessor failed at step {step_count}: {e}", exc_info=True)
            raise

        # Run policy inference using select_action (same as lerobot_eval.py)
        with torch.inference_mode():
            # select_action returns normalized action and handles action queue internally
            action = policy.select_action(processed_obs)

        # Apply postprocessor (handles unnormalization)
        try:
            unnormalized_action = postprocessor(action)
        except Exception as e:
            logger.error(f"Postprocessor failed at step {step_count}: {e}", exc_info=True)
            raise

        # Convert to numpy
        if isinstance(unnormalized_action, torch.Tensor):
            action_np = unnormalized_action.cpu().numpy()
        else:
            action_np = np.array(unnormalized_action)

        # Handle batch dimension - action shape could be (B, D) or (D,)
        if action_np.ndim == 2:
            action_np = action_np[0]  # Take first batch element

        # Log first step for debugging
        if step_count == 0:
            logger.info(f"Unnormalized action sample: {action_np[:min(6, len(action_np))]}")
            
            # Also log GT action for comparison
            gt_action = frame[action_key]
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.cpu().numpy()
            logger.info(f"GT action sample: {gt_action[:min(6, len(gt_action))]}")

        # Truncate to dataset action dim if needed
        if len(action_np) > action_dim and action_dim > 0:
            action_np = action_np[:action_dim]

        # Collect predictions for inference_interval steps
        for j in range(min(inference_interval, actual_steps - step_count)):
            pred_actions.append(action_np.copy())

    # Extract ground truth actions for all steps
    for step_count in range(actual_steps):
        frame = episode_frames[step_count]
        
        # Ground truth action
        gt_action = frame[action_key]
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
        gt_actions.append(gt_action.flatten())

        # State
        if state_key in frame:
            state = frame[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            state_joints.append(state.flatten())
        else:
            state_joints.append(np.zeros(state_dim))

    # Convert to arrays
    pred_actions = np.array(pred_actions)[:actual_steps]
    gt_actions = np.array(gt_actions)
    state_joints = np.array(state_joints)

    # Handle shape mismatch
    if pred_actions.shape != gt_actions.shape:
        logger.warning(
            f"Shape mismatch: pred {pred_actions.shape}, gt {gt_actions.shape}"
        )
        min_len = min(len(pred_actions), len(gt_actions))
        min_dim = min(pred_actions.shape[1], gt_actions.shape[1])
        pred_actions = pred_actions[:min_len, :min_dim]
        gt_actions = gt_actions[:min_len, :min_dim]

    # Compute metrics
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))

    print(f"\n=== Episode {episode_id} Evaluation Results ===")
    print(f"Action MSE: {mse:.6f}")
    print(f"Action MAE: {mae:.6f}")
    print(f"GT action shape: {gt_actions.shape}")
    print(f"Pred action shape: {pred_actions.shape}")
    print("=" * 40)

    logger.info(f"Action MSE: {mse:.6f}, MAE: {mae:.6f}")

    # Plot trajectory
    if save_plot_path:
        plot_trajectory_results(
            state_joints_across_time=state_joints,
            gt_action_across_time=gt_actions,
            pred_action_across_time=pred_actions,
            traj_id=episode_id,
            state_keys=["state"],
            action_keys=["action"],
            action_horizon=inference_interval,
            save_plot_path=save_plot_path,
        )
        logger.info(f"Plot saved to: {save_plot_path}")

    return mse, mae


@dataclass
class EvalConfig:
    """Configuration for open-loop evaluation using standard pipeline."""

    dataset_repo_id: str = "izuluaga/finish_sandwich"
    """Hugging Face dataset repository ID."""

    policy_repo_id: str = "aravindhs-NV/finish_sandwich_fix_processor_0114_bs32"
    """Hugging Face policy repository ID."""

    episode_ids: list[int] = field(default_factory=lambda: [0])
    """List of episode IDs to evaluate."""

    steps: int = 200
    """Maximum number of steps to evaluate per episode."""

    inference_interval: int = 10
    """Interval between inference points (in steps)."""

    save_dir: str = "./outputs/open_loop_eval_v4"
    """Directory to save trajectory plots."""

    device: str = "cuda"
    """Device to run inference on ('cuda' or 'cpu')."""

    rename_map: dict[str, str] = field(default_factory=dict)
    """Optional key rename map for observations."""


def main(config: EvalConfig):
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("Open-Loop Evaluation v4 - Using Standard LeRobot Pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_repo_id}")
    logger.info(f"Policy: {config.policy_repo_id}")
    logger.info(f"Episodes: {config.episode_ids}")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata first
    logger.info("Loading dataset metadata...")
    meta = LeRobotDatasetMetadata(config.dataset_repo_id)
    total_episodes = meta.total_episodes
    logger.info(f"Dataset has {total_episodes} total episodes")

    # Validate episode IDs
    valid_episode_ids = []
    for episode_id in config.episode_ids:
        if episode_id >= total_episodes:
            logger.warning(f"Episode {episode_id} out of range (max: {total_episodes-1}). Skipping.")
            continue
        valid_episode_ids.append(episode_id)

    if not valid_episode_ids:
        logger.error("No valid episode IDs provided. Exiting.")
        return

    # Load dataset with filtered episodes
    logger.info(f"Loading dataset with episodes: {valid_episode_ids}")
    dataset = LeRobotDataset(config.dataset_repo_id, episodes=valid_episode_ids)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # Load policy
    logger.info("Loading policy...")
    policy_class = get_policy_class("gr00t_n1d6")
    policy = policy_class.from_pretrained(config.policy_repo_id)
    policy.eval()
    policy.to(config.device)
    logger.info("Policy loaded successfully")

    # Create preprocessor and postprocessor using the SAME function as lerobot_eval.py
    # This is the KEY difference from open_loop_eval_v3!
    logger.info("Creating preprocessor/postprocessor pipelines...")
    
    preprocessor_overrides = {
        "device_processor": {"device": config.device},
    }
    if config.rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": config.rename_map}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=config.policy_repo_id,
        dataset_stats=dataset.meta.stats if hasattr(dataset.meta, "stats") else None,
        preprocessor_overrides=preprocessor_overrides,
    )
    logger.info("Pipelines created successfully")

    # Log some debug info about the processor
    try:
        from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import Gr00tN1d6ProcessStep
        for step in preprocessor.steps:
            if isinstance(step, Gr00tN1d6ProcessStep):
                proc = step.processor
                stats = proc.state_action_processor.statistics
                logger.info(f"Processor statistics keys: {list(stats.keys())}")
                for emb_tag in stats:
                    logger.info(f"  Embodiment '{emb_tag}' modalities: {list(stats[emb_tag].keys())}")
                break
    except Exception as e:
        logger.warning(f"Could not inspect processor stats: {e}")

    # Evaluate each episode
    all_mse = []
    all_mae = []

    for episode_id in valid_episode_ids:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating episode: {episode_id}")
        logger.info(f"{'='*40}")

        try:
            save_plot_path = str(save_dir / f"traj_{episode_id}.png")
            mse, mae = evaluate_trajectory_with_pipeline(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                episode_id=episode_id,
                steps=config.steps,
                inference_interval=config.inference_interval,
                save_plot_path=save_plot_path,
            )
            all_mse.append(mse)
            all_mae.append(mae)
        except Exception as e:
            logger.error(f"Error evaluating episode {episode_id}: {e}", exc_info=True)
            continue

    # Summary
    if all_mse:
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae)
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Average MSE across {len(all_mse)} episodes: {avg_mse:.6f}")
        print(f"Average MAE across {len(all_mae)} episodes: {avg_mae:.6f}")
        print(f"{'='*50}")
        logger.info(f"Average MSE: {avg_mse:.6f}, Average MAE: {avg_mae:.6f}")
    else:
        logger.warning("No valid episodes were evaluated.")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)

