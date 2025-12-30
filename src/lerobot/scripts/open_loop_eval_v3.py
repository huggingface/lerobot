"""Open-loop evaluation script for GR00T N1.6 models on LeRobot v3 datasets.

This script evaluates GR00T N1.6 policies on LeRobot v3 datasets from Hugging Face Hub.
It supports:
- Loading v3 datasets directly from Hub
- Evaluating model predictions with open-loop action prediction
- Computing metrics (MSE, MAE) and generating trajectory comparison plots

Usage examples:
--------------
- Evaluate a model on a dataset:
    python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --policy-repo-id=nvkartik/gr00t_n1d6-finish_sandwich-2bs \
        --episode-ids=0 \
        --steps=200 \
        --save-dir=./eval_outputs

- Visualize dataset trajectories only (no model evaluation):
    python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --episode-ids=0 \
        --visualize-only=True \
        --save-dir=./eval_outputs

- Evaluate multiple episodes:
    python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --policy-repo-id=nvkartik/gr00t_n1d6-finish_sandwich-2bs \
        --episode-ids=0 1 2 \
        --steps=400 \
        --action-horizon=16 \
        --save-dir=./outputs/eval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from matplotlib import pyplot as plt
from PIL import Image

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray | None,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time (None for visualize-only mode)
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference (not used if pred_action_across_time is None)
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logger.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys (format similar to SO100 example)
    all_modalities = state_keys + action_keys
    fig.suptitle(
        f"Trajectory {traj_id} - Modalities: {', '.join(all_modalities)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")

        # Only plot predictions if available
        if pred_action_across_time is not None:
            ax.plot(pred_action_across_time[:, action_idx], label="pred action")
            # put a dot every ACTION_HORIZON (only when predictions are available)
            for j in range(0, actual_steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def convert_lerobot_to_groot_batch(
    lerobot_batch: dict[str, Any],
    camera_keys: list[str],
    state_key: str = "observation.state",
    language_key: str = "task",
) -> dict[str, Any]:
    """
    Convert LeRobot batch format to GR00T input format.

    Args:
        lerobot_batch: Batch from LeRobotDataset
        camera_keys: List of camera keys
        state_key: Key for state data
        language_key: Key for language/task data

    Returns:
        Dictionary in GR00T format with video, state, and language keys
    """
    groot_batch = {}

    # Convert video/images
    groot_batch["video"] = {}
    for cam_key in camera_keys:
        if cam_key in lerobot_batch:
            img = lerobot_batch[cam_key]
            # Convert to numpy array if needed
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            elif not isinstance(img, np.ndarray):
                # If it's not a tensor or numpy array, try to convert it
                img = np.array(img)

            # Handle different image formats
            # LeRobot stores images as (C, H, W) - need to convert to (H, W, C) for PIL
            if img.ndim == 3:
                # Single image: could be (C, H, W) or (H, W, C)
                if img.shape[0] in [1, 3, 4]:  # Likely (C, H, W) format
                    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                # Now img is (H, W, C) -> (1, H, W, C)
                img = img[None, :]
            elif img.ndim == 4:
                # Already batched: could be (B, C, H, W) or (B, H, W, C)
                if img.shape[1] in [1, 3, 4] and img.shape[1] < img.shape[2]:  # Likely (B, C, H, W)
                    img = np.transpose(img, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            # Ensure uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            groot_batch["video"][cam_key] = img

    # Convert state
    groot_batch["state"] = {}
    if state_key in lerobot_batch:
        state = lerobot_batch[state_key]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # Handle different state formats
        if state.ndim == 1:
            # Single state: (D,) -> (1, D)
            state = state[None, :]
        elif state.ndim == 2:
            # Already batched: (B, D) or (T, D)
            pass
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        # Use a generic key for state
        groot_batch["state"]["state"] = state.astype(np.float32)

    # Convert language/task
    groot_batch["language"] = {}
    if language_key in lerobot_batch:
        task = lerobot_batch[language_key]
        if isinstance(task, str):
            # Single task string -> list of lists format
            groot_batch["language"][language_key] = [[task]]
        elif isinstance(task, (list, tuple)):
            # Already in list format
            if len(task) > 0 and isinstance(task[0], str):
                groot_batch["language"][language_key] = [[t] for t in task]
            else:
                groot_batch["language"][language_key] = task
        else:
            logger.warning(f"Unexpected language format: {type(task)}, skipping")

    return groot_batch


def get_episode_frames(dataset: LeRobotDataset, episode_id: int) -> list[dict[str, Any]]:
    """
    Get all frames for a specific episode from LeRobotDataset.

    Args:
        dataset: LeRobotDataset instance
        episode_id: Episode index

    Returns:
        List of frame dictionaries (processed through dataset.__getitem__)
    """
    dataset._ensure_hf_dataset_loaded()

    # First, find all frame indices belonging to this episode
    episode_indices = []
    for idx in range(len(dataset.hf_dataset)):
        if dataset.hf_dataset[idx]["episode_index"] == episode_id:
            episode_indices.append(idx)

    # Now get processed frames using dataset.__getitem__
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


def visualize_dataset_trajectory(
    dataset: LeRobotDataset,
    episode_id: int,
    steps: int = 200,
    save_plot_path: str | None = None,
) -> None:
    """
    Visualize ground truth trajectories from the dataset without model evaluation.

    Args:
        dataset: LeRobotDataset instance
        episode_id: Episode index to visualize
        steps: Maximum number of steps to visualize
        save_plot_path: Path to save the plot
    """
    # Get all frames for this episode
    episode_frames = get_episode_frames(dataset, episode_id)
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    traj_length = len(episode_frames)
    actual_steps = min(steps, traj_length)
    logger.info(f"Visualizing {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})")

    state_key = "observation.state"
    action_key = "action"

    # Extract state and action keys
    state_keys = ["state"]
    action_keys = ["action"]

    # Get state and action dimensions
    state_dim = (
        dataset.meta.features.get(state_key, {}).get("shape", [0])[0]
        if state_key in dataset.meta.features
        else 0
    )
    action_dim = (
        dataset.meta.features.get(action_key, {}).get("shape", [0])[0]
        if action_key in dataset.meta.features
        else 0
    )

    logger.info(f"State keys: {state_keys}, Action keys: {action_keys}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Extract ground truth trajectories
    state_joints_across_time = []
    gt_action_across_time = []

    # Infer dimensions from first frame if not available from metadata
    inferred_state_dim = state_dim
    inferred_action_dim = action_dim

    for step_count in range(actual_steps):
        if step_count >= len(episode_frames):
            break

        frame = episode_frames[step_count]

        # Extract state
        if state_key in frame:
            state = frame[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if inferred_state_dim == 0 and state.size > 0:
                inferred_state_dim = state.size if state.ndim == 0 else state.shape[-1]
            state_joints_across_time.append(state.flatten() if state.ndim > 1 else state)
        elif inferred_state_dim > 0:
            # Use zeros if state not available but we know the dimension
            state_joints_across_time.append(np.zeros(inferred_state_dim))

        # Extract action
        if action_key in frame:
            action = frame[action_key]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if inferred_action_dim == 0 and action.size > 0:
                inferred_action_dim = action.size if action.ndim == 0 else action.shape[-1]
            gt_action_across_time.append(action.flatten() if action.ndim > 1 else action)
        elif inferred_action_dim > 0:
            # Use zeros if action not available but we know the dimension
            gt_action_across_time.append(np.zeros(inferred_action_dim))

    # Convert to numpy arrays
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)

    # Plot trajectory results (only ground truth, no predictions)
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=None,  # No predictions in visualize-only mode
        traj_id=episode_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=1,  # Not used when pred_action_across_time is None
        save_plot_path=save_plot_path,
    )


def evaluate_single_trajectory(
    policy: Any,
    dataset: LeRobotDataset,
    episode_id: int,
    steps: int = 200,
    action_horizon: int = 16,
    inference_interval: int | None = None,
    save_plot_path: str | None = None,
) -> tuple[float, float]:
    """
    Evaluate a single trajectory/episode.

    Args:
        policy: GR00T N1.6 policy instance
        dataset: LeRobotDataset instance
        episode_id: Episode index to evaluate
        steps: Maximum number of steps to evaluate
        action_horizon: Action horizon for inference (number of steps predicted at once)
        inference_interval: Interval between inference points. If None, uses action_horizon.
        save_plot_path: Path to save the plot

    Returns:
        Tuple of (MSE, MAE) metrics
    """
    # Use action_horizon as default inference interval if not specified
    if inference_interval is None:
        inference_interval = action_horizon
    # Get all frames for this episode
    episode_frames = get_episode_frames(dataset, episode_id)
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    traj_length = len(episode_frames)
    actual_steps = min(steps, traj_length)
    logger.info(f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})")

    # Pre-allocate arrays to handle overlapping predictions correctly
    pred_action_across_time = [None] * actual_steps
    gt_action_across_time = []

    camera_keys = dataset.meta.camera_keys
    state_key = "observation.state"
    action_key = "action"
    language_key = "task"

    # Extract state and action keys from dataset metadata
    # Try to get from policy config if available, otherwise use defaults
    state_keys = []
    action_keys = []

    # Get state keys from dataset features
    if hasattr(dataset.meta, "features") and state_key in dataset.meta.features:
        # Try to infer state keys from the dataset structure
        # For now, use a generic "state" key
        state_keys = ["state"]
    else:
        state_keys = ["state"]

    # Get action keys from dataset features
    if hasattr(dataset.meta, "features") and action_key in dataset.meta.features:
        action_keys = ["action"]
    else:
        action_keys = ["action"]

    # Get state and action dimensions from dataset features
    state_dim = (
        dataset.meta.features.get(state_key, {}).get("shape", [0])[0]
        if state_key in dataset.meta.features
        else 0
    )
    action_dim = (
        dataset.meta.features.get(action_key, {}).get("shape", [0])[0]
        if action_key in dataset.meta.features
        else 0
    )

    logger.info(f"State keys: {state_keys}, Action keys: {action_keys}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"Inference interval: {inference_interval} steps, Action horizon: {action_horizon} steps")

    # Step through episode at inference_interval intervals
    for step_count in range(0, actual_steps, inference_interval):
        if step_count >= len(episode_frames):
            break

        logger.info(f"Inferencing at step: {step_count}")

        # Get current frame
        frame = episode_frames[step_count]

        # Convert to GR00T format
        groot_batch = convert_lerobot_to_groot_batch(frame, camera_keys, state_key, language_key)

        # Prepare batch for policy in vlm_content format
        policy_batch = {}

        # Extract and convert images to PIL format for vlm_content
        pil_images = []
        for _cam_key, img_array in groot_batch.get("video", {}).items():
            # img_array shape: (T, H, W, C) - take first frame for inference
            img = img_array[0] if img_array.ndim == 4 else img_array
            # Convert to PIL Image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_images.append(Image.fromarray(img))

        # Extract language/task
        task_text = "complete the task"  # Default task
        for _lang_key, lang_data in groot_batch.get("language", {}).items():
            if isinstance(lang_data, list) and len(lang_data) > 0:
                if isinstance(lang_data[0], list) and len(lang_data[0]) > 0:
                    task_text = lang_data[0][0]
                elif isinstance(lang_data[0], str):
                    task_text = lang_data[0]
            elif isinstance(lang_data, str):
                task_text = lang_data
            break

        # Create conversation format for the VLM
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_text},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]

        # Apply chat template to get properly formatted text
        # The collator expects text to be the output of apply_chat_template
        from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import build_processor

        if not hasattr(evaluate_single_trajectory, "_processor"):
            evaluate_single_trajectory._processor = build_processor(
                "nvidia/Eagle-Block2A-2B-v2", {"trust_remote_code": True}
            )
        formatted_text = evaluate_single_trajectory._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Create vlm_content in the format expected by the model's collator
        policy_batch["vlm_content"] = {
            "text": formatted_text,
            "images": pil_images,
            "conversation": conversation,
        }

        # Extract and prepare state
        for _state_key_name, state_array in groot_batch.get("state", {}).items():
            # Ensure proper shape: (T, D) -> (1, D) for single timestep inference
            if state_array.ndim == 2:
                state_tensor = torch.from_numpy(state_array[0:1]).to(policy.device).float()
            else:
                state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(policy.device).float()
            policy_batch["state"] = state_tensor

            # Provide raw_state for relative->absolute action conversion
            # The processor's decode_action uses this to convert relative actions to absolute
            # state_array has shape (T, D) - use the full state for proper conversion
            policy_batch["raw_state"] = {"state": state_array}
            break  # Use first state key

        # Set embodiment_id (default to 0 for generic embodiment)
        policy_batch["embodiment_id"] = torch.tensor([0], device=policy.device, dtype=torch.long)

        # Run inference
        try:
            with torch.no_grad():
                # Use predict_action_chunk for action prediction
                action_chunk = policy.predict_action_chunk(policy_batch)
                # action_chunk shape: (B, action_horizon, action_dim)
                action_chunk_np = action_chunk.cpu().numpy()[0]  # Remove batch dimension

                # Handle dimension mismatch between model output and dataset action dimensions
                # The model may output more dimensions than the dataset expects (e.g., 23 vs 6)
                # This can happen when using a base model processor with different embodiment config
                # Try to find the best matching slice by checking value ranges
                if action_chunk_np.shape[1] > action_dim and action_dim > 0:
                    original_dim = action_chunk_np.shape[1]
                    # The model outputs more dims than GT - need to find the correct slice
                    # For behavior_r1_pro, the first 3 dims are "base" velocity commands (small values)
                    # The actual joint positions start at dim 3
                    # Heuristic: skip the first few dims if they're in a different range
                    # For now, try offset=3 (skip base dims) as the most common case
                    best_offset = 3 if original_dim >= action_dim + 3 else 0
                    action_chunk_np = action_chunk_np[:, best_offset : best_offset + action_dim]

                # NOTE: predict_action_chunk already calls processor.decode_action() which:
                # 1. Unnormalizes actions using pretrained model's statistics
                # 2. Converts relative->absolute if state is provided
                # Therefore, we should NOT unnormalize again here!
                #
                # The following code was causing DOUBLE UNNORMALIZATION and is now disabled:
                #
                # if (
                #     hasattr(dataset.meta, "stats")
                #     and dataset.meta.stats is not None
                #     and action_key in dataset.meta.stats
                # ):
                #     action_stats = dataset.meta.stats[action_key]
                #     action_min = action_stats.get("min", None)
                #     action_max = action_stats.get("max", None)
                #     if action_min is not None and action_max is not None:
                #         # This was WRONG: actions are already unnormalized by decode_action!
                #         action_chunk_np = (action_chunk_np + 1.0) / 2.0 * action_range + action_min
                pass  # Actions are already unnormalized by predict_action_chunk -> decode_action

        except Exception as e:
            logger.error(f"Error during inference at step {step_count}: {e}")
            # Use zeros as fallback
            action_chunk_np = np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Collect predicted actions (handle overlapping predictions)
        for j in range(min(action_horizon, actual_steps - step_count)):
            action_idx = step_count + j
            if action_idx < actual_steps:
                # Later predictions overwrite earlier ones (more recent inference takes precedence)
                pred_action_across_time[action_idx] = action_chunk_np[j]

    # Collect ground truth actions independently for all steps (like state joints)
    for frame in episode_frames[:actual_steps]:
        if action_key in frame:
            gt_action = frame[action_key]
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.cpu().numpy()
            if gt_action.ndim == 1:
                gt_action_across_time.append(gt_action)
            elif gt_action.ndim == 0:
                gt_action_across_time.append(np.array([gt_action]))
            else:
                gt_action_across_time.append(gt_action.flatten()[:action_dim])
        else:
            # No ground truth action available
            gt_action_across_time.append(np.zeros(action_dim, dtype=np.float32))

    # Convert to numpy arrays
    # Filter out None values (steps that weren't predicted) and convert to array
    # If there are None values, fill them with zeros or the last valid prediction
    valid_pred_actions = []
    last_valid_action = None
    for _i, action in enumerate(pred_action_across_time[:actual_steps]):
        if action is not None:
            valid_pred_actions.append(action)
            last_valid_action = action
        elif last_valid_action is not None:
            # Use last valid prediction if available
            valid_pred_actions.append(last_valid_action)
        else:
            # No valid predictions yet, infer dimension from first valid action or use zeros
            if len(valid_pred_actions) > 0:
                # Use shape from first valid action
                valid_pred_actions.append(np.zeros_like(valid_pred_actions[0]))
            else:
                # Fallback: use action_dim if available, otherwise zeros
                fallback_dim = action_dim if action_dim > 0 else 1
                valid_pred_actions.append(np.zeros(fallback_dim, dtype=np.float32))

    pred_action_across_time = np.array(valid_pred_actions)
    gt_action_across_time = np.array(gt_action_across_time[:actual_steps])

    # Ensure shapes match
    if len(pred_action_across_time) != len(gt_action_across_time):
        min_len = min(len(pred_action_across_time), len(gt_action_across_time))
        pred_action_across_time = pred_action_across_time[:min_len]
        gt_action_across_time = gt_action_across_time[:min_len]

    if pred_action_across_time.shape != gt_action_across_time.shape:
        # Pad or truncate to match
        min_dim = min(pred_action_across_time.shape[1], gt_action_across_time.shape[1])
        pred_action_across_time = pred_action_across_time[:, :min_dim]
        gt_action_across_time = gt_action_across_time[:, :min_dim]

    # Compute metrics
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

    # Print to stdout for visibility (in addition to logging)
    print(f"\n=== Episode {episode_id} Evaluation Results ===")
    print(f"Unnormalized Action MSE: {mse:.6f}")
    print(f"Unnormalized Action MAE: {mae:.6f}")
    print(f"GT action shape: {gt_action_across_time.shape}")
    print(f"Pred action shape: {pred_action_across_time.shape}")
    print("=" * 40)

    logger.info(f"Unnormalized Action MSE across single traj: {mse}")
    logger.info(f"Unnormalized Action MAE across single traj: {mae}")
    logger.info(f"GT action shape: {gt_action_across_time.shape}")
    logger.info(f"Pred action shape: {pred_action_across_time.shape}")

    # Extract state joints for plotting
    state_joints_across_time = []
    for frame in episode_frames[:actual_steps]:
        if state_key in frame:
            state = frame[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if state.ndim == 0:
                state = np.array([state])
            state_joints_across_time.append(state.flatten()[:state_dim])
        else:
            state_joints_across_time.append(np.zeros(state_dim, dtype=np.float32))
    state_joints_across_time = np.array(state_joints_across_time)

    # Plot trajectory results
    if save_plot_path is None:
        # Default to current directory if not specified
        default_dir = Path("./eval_outputs")
        default_dir.mkdir(parents=True, exist_ok=True)
        save_plot_path = str(default_dir / f"traj_{episode_id}.png")

    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=episode_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=inference_interval,  # Use inference_interval for marking inference points
        save_plot_path=save_plot_path,
    )

    return mse, mae


@dataclass
class EvalConfig:
    """Configuration for evaluating a GR00T N1.6 policy on a LeRobot v3 dataset.

    This configuration class defines all CLI arguments for the evaluation script.
    Use --help to see all available options.

    Parameter Mapping to Isaac-GR00T (gr00t/eval/open_loop_eval.py):
    ===================================================================
    - action_horizon: Maps to ArgsConfig.action_horizon (default: 16)
      * Also determines inference interval (inference every action_horizon steps)
      * SO100 example uses: 16

    - steps: Maps to ArgsConfig.steps (default: 200)
      * SO100 example uses: 400

    - denoising_steps: Maps to ArgsConfig.denoising_steps (default: 4)
      * NOTE: Isaac-GR00T defines this but doesn't use it
      * Model uses config.num_inference_timesteps instead (typically 4)

    - inference_interval: NOT in Isaac-GR00T (always uses action_horizon)
      * LeRobot addition for flexibility, defaults to action_horizon

    Isaac-GR00T Evaluation Pattern:
    ================================
    for step_count in range(0, actual_steps, action_horizon):
        action_chunk = policy.get_action(obs)  # Returns (B, action_horizon, D)
        for j in range(action_horizon):
            pred_action_across_time.append(action_chunk[j])
    """

    dataset_repo_id: str = "izuluaga/finish_sandwich"
    """Hugging Face dataset repository ID (e.g., 'izuluaga/finish_sandwich')."""

    policy_repo_id: str = "nvkartik/gr00t_n1d6-finish_sandwich-2bs"
    """Hugging Face policy repository ID (e.g., 'nvkartik/gr00t_n1d6-finish_sandwich-2bs')."""

    episode_ids: list[int] = field(default_factory=lambda: [0])
    """List of episode IDs to evaluate (e.g., [0] or [0, 1, 2])."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length).
    Default matches Isaac-GR00T. SO100 example uses 400 steps."""

    action_horizon: int = 16
    """Action horizon for inference (number of steps predicted at once).
    Also determines the inference interval (inference happens every action_horizon steps)."""

    inference_interval: int | None = None
    """Interval between inference points in steps. If None, uses action_horizon (matches Isaac-GR00T behavior).
    NOTE: Isaac-GR00T always uses action_horizon as the inference interval. This parameter is for advanced use cases only."""

    save_dir: str = "./outputs/eval"
    """Directory to save trajectory plots (e.g., './outputs/eval')."""

    device: str = "cuda"
    """Device to run inference on ('cuda' or 'cpu')."""

    denoising_steps: int | None = None
    """Number of denoising steps for flow matching. If None, uses model's config.num_inference_timesteps.
    NOTE: Isaac-GR00T defines this parameter but doesn't use it - the model uses its config value instead.
    This parameter is included for completeness but may not affect inference."""

    visualize_only: bool = False
    """If True, only visualize dataset trajectories without evaluating the model."""


def main(config: EvalConfig):
    """Main evaluation function."""
    if config.visualize_only:
        logger.info("Starting dataset trajectory visualization")
    else:
        logger.info("Starting GR00T N1.6 evaluation on LeRobot v3 dataset")
    logger.info(f"Dataset: {config.dataset_repo_id}")
    if not config.visualize_only:
        logger.info(f"Policy: {config.policy_repo_id}")
    logger.info(f"Episodes: {config.episode_ids}")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata first to check total episodes
    logger.info("Loading dataset metadata...")
    meta = LeRobotDatasetMetadata(config.dataset_repo_id)
    total_episodes = meta.total_episodes
    logger.info(f"Dataset has {total_episodes} total episodes")

    # Validate episode IDs before filtering
    valid_episode_ids = []
    for episode_id in config.episode_ids:
        if episode_id >= total_episodes:
            logger.warning(
                f"Episode ID {episode_id} is out of range (dataset has {total_episodes} episodes). Skipping."
            )
            continue
        valid_episode_ids.append(episode_id)

    if not valid_episode_ids:
        logger.error("No valid episode IDs provided. Exiting.")
        return

    # Now load dataset with filtered episodes
    logger.info(f"Loading dataset with episodes: {valid_episode_ids}")
    dataset = LeRobotDataset(config.dataset_repo_id, episodes=valid_episode_ids)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # If visualize_only, skip model loading and evaluation
    if config.visualize_only:
        logger.info("Visualize-only mode: Plotting dataset trajectories without model evaluation")
        for episode_id in valid_episode_ids:
            logger.info(f"Visualizing trajectory for episode: {episode_id}")
            try:
                save_plot_path = str(save_dir / f"traj_{episode_id}.png")
                visualize_dataset_trajectory(
                    dataset,
                    episode_id,
                    steps=config.steps,
                    save_plot_path=save_plot_path,
                )
                logger.info(f"Trajectory plot saved for episode {episode_id}")
            except Exception as e:
                logger.error(f"Error visualizing episode {episode_id}: {e}", exc_info=True)
                continue

        logger.info("Visualization complete!")
        return

    # Load policy for evaluation
    logger.info("Loading policy...")
    policy_class = get_policy_class("gr00t_n1d6")
    policy = policy_class.from_pretrained(config.policy_repo_id)
    policy.eval()
    policy.to(config.device)
    logger.info("Policy loaded successfully")

    # Evaluate each episode
    all_mse = []
    all_mae = []

    for episode_id in valid_episode_ids:
        logger.info(f"Running evaluation on episode: {episode_id}")
        try:
            save_plot_path = str(save_dir / f"traj_{episode_id}.png")
            mse, mae = evaluate_single_trajectory(
                policy,
                dataset,
                episode_id,
                steps=config.steps,
                action_horizon=config.action_horizon,
                inference_interval=config.inference_interval,
                save_plot_path=save_plot_path,
            )
            logger.info(f"MSE for episode {episode_id}: {mse}, MAE: {mae}")
            all_mse.append(mse)
            all_mae.append(mae)
        except Exception as e:
            logger.error(f"Error evaluating episode {episode_id}: {e}", exc_info=True)
            continue

    # Print summary
    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logger.info(f"Average MSE across all episodes: {avg_mse}")
        logger.info(f"Average MAE across all episodes: {avg_mae}")
    else:
        logger.info("No valid episodes were evaluated.")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(EvalConfig)
    main(config)
