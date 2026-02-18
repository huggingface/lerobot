#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test script to verify Groot policy integration with LeRobot vs the original implementation, only meant to be run locally!"""

import gc
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

pytest.importorskip("gr00t")
pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local Groot installation and is not meant for CI",
)


from gr00t.data.dataset import ModalityConfig  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.transform.base import ComposedModalityTransform  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

# GR1 humanoid dimensions (from pretrained model metadata)
# The actual GR1 robot has 44 dimensions for both state and action
# GR00TTransform will pad state to 64 and truncate action to 32
DUMMY_STATE_DIM = 44
DUMMY_ACTION_DIM = 44
DUMMY_ACTION_HORIZON = 16
IMAGE_SIZE = 256
DEVICE = "cpu"
MODEL_PATH = "nvidia/GR00T-N1.5-3B"

GR1_BODY_PARTS = {
    "left_arm": 7,
    "left_hand": 6,
    "left_leg": 6,
    "neck": 3,
    "right_arm": 7,
    "right_hand": 6,
    "right_leg": 6,
    "waist": 3,
}


def cleanup_memory():
    """Clean up GPU/MPS memory to prevent OOM errors between tests."""
    print("\nCleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Memory cleanup complete.")


def set_seed_all(seed: int):
    """Set random seed for all RNG sources to ensure reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def instantiate_lerobot_groot(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH,
) -> tuple[
    GrootPolicy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot Groot policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = GrootPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=False,
        )
        policy.config.embodiment_tag = "gr1"
    else:
        config = GrootConfig(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_HORIZON,
            chunk_size=DUMMY_ACTION_HORIZON,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
            device=DEVICE,
            embodiment_tag="gr1",
        )
        policy = GrootPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE

    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization (original GR00T doesn't normalize)
    )

    return (policy, preprocessor, postprocessor)


def instantiate_original_groot(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH,
):
    """Instantiate original Groot policy from NVIDIA's implementation."""
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor
    from gr00t.data.transform.video import VideoToNumpy, VideoToTensor
    from gr00t.model.transforms import GR00TTransform

    video_keys = ["video.ego_view"]
    state_keys = [
        "state"
    ]  # Important: Use single concatenated "state" key (not split body parts) to match preprocessing
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.left_leg",
        "action.right_leg",
        "action.neck",
        "action.waist",
    ]
    language_keys = ["annotation.human.action.task_description"]

    modality_config = {
        "video": ModalityConfig(
            delta_indices=[0],  # Current frame only
            modality_keys=video_keys,
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_keys,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(DUMMY_ACTION_HORIZON)),
            modality_keys=action_keys,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=language_keys,
        ),
    }

    modality_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_keys),
            VideoToNumpy(apply_to=video_keys),  # Convert to numpy (GR00TTransform expects numpy arrays)
            # State is already a single concatenated key, so no StateActionToTensor needed
            # Convert action from numpy to tensor
            StateActionToTensor(apply_to=action_keys),
            # Concatenate only video and actions (state is already single key)
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=[],  # Empty:state is already single key
                action_concat_order=action_keys,
            ),
            GR00TTransform(
                max_state_dim=64,
                max_action_dim=32,
                state_horizon=1,
                action_horizon=DUMMY_ACTION_HORIZON,
                training=False,
            ),
        ]
    )

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.GR1,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=DEVICE,
    )

    return policy, modality_config, modality_transform


def create_dummy_data(device=DEVICE):
    """Create dummy data for testing both implementations."""
    batch_size = 2
    prompt = "Pick up the red cube and place it in the bin"
    state = torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device)

    batch = {
        "observation.state": state,
        "action": torch.randn(
            batch_size,
            DUMMY_ACTION_HORIZON,
            DUMMY_ACTION_DIM,
            dtype=torch.float32,
            device=device,  # Action ground truth (for training)
        ),
        "observation.images.ego_view": torch.rand(
            batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            dtype=torch.float32,
            device=device,  # Images in [0, 1] range as expected by LeRobot
        ),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


def convert_lerobot_to_original_format(batch, modality_config):
    """Convert LeRobot batch format to original Groot format.

    The original Groot expects observations in this format:
    {
        "video.<camera_name>": np.ndarray (T, H, W, C) or (B, T, H, W, C)
        "state.<state_component>": np.ndarray (T, D) or (B, T, D)
        "action.<action_component>": np.ndarray (T, D) or (B, T, D)
        "annotation.<annotation_type>": str or list[str]
    }
    """
    # Original Groot expects (T, H, W, C) format for images
    # LeRobot has (B, C, H, W) format, so we need to convert
    observation = {}

    for img_key in ["ego_view"]:
        lerobot_key = f"observation.images.{img_key}"
        if lerobot_key in batch:
            img = batch[lerobot_key]
            # Convert from (B, C, H, W) to (B, T=1, H, W, C)
            img_np = img.permute(0, 2, 3, 1).unsqueeze(1).cpu().numpy()
            # Convert [0, 1] to [0, 255] uint8 as expected by original
            img_np = (img_np * 255).astype(np.uint8)
            observation[f"video.{img_key}"] = img_np

    # Important: The Original's GR00TTransform expects "state" as (B, T, D), not split body parts
    if "observation.state" in batch:
        state = batch["observation.state"]
        state_np = state.unsqueeze(1).cpu().numpy()  # (B, 1, D)
        observation["state"] = state_np

    if "action" in batch:
        action = batch["action"]
        action_np = action.cpu().numpy()

        start_idx = 0
        for part_name, part_dim in GR1_BODY_PARTS.items():
            end_idx = start_idx + part_dim
            observation[f"action.{part_name}"] = action_np[:, :, start_idx:end_idx]
            start_idx = end_idx

    if "task" in batch:
        task_list = batch["task"]
        # GR00TTransform expects language with (B, T) shape for batched data
        # Create a (B, T=1) array where each element is the string directly
        bsz = len(task_list)
        task_array = np.empty((bsz, 1), dtype=object)
        for i in range(bsz):
            task_array[i, 0] = task_list[i]  # Assign string directly to each (i, 0) position
        observation["annotation.human.action.task_description"] = task_array

    return observation


def test_groot_original_vs_lerobot_pretrained():
    """Test Groot original implementation vs LeRobot implementation with pretrained weights."""
    print("Test: Groot Original vs LeRobot with Pretrained Weights (Inference)")

    set_seed_all(42)

    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_groot(
        from_pretrained=True
    )
    original_policy, modality_config, modality_transform = instantiate_original_groot(from_pretrained=True)

    batch = create_dummy_data()
    batch_lerobot = deepcopy(batch)

    print("\n[LeRobot] Running inference...")
    lerobot_policy.eval()
    batch_lerobot_processed = lerobot_preprocessor(batch_lerobot)

    # Important: Reset seed immediately before inference to ensure identical RNG state
    torch.manual_seed(42)

    with torch.no_grad():
        lerobot_actions = lerobot_policy.select_action(batch_lerobot_processed)

    print("\n[Original] Running inference...")
    original_policy.model.eval()
    observation = convert_lerobot_to_original_format(batch, modality_config)
    original_obs_transformed = modality_transform(deepcopy(observation))

    # Important: Reset seed immediately before inference to ensure identical RNG state
    torch.manual_seed(42)

    with torch.no_grad():
        original_model_output = original_policy.model.get_action(original_obs_transformed)
        original_actions_raw = original_model_output["action_pred"]  # [2, 16, 32]
    # Take first timestep
    original_actions = original_actions_raw[:, 0, :].to(lerobot_actions.device).to(lerobot_actions.dtype)

    print("Action Comparison:")
    diff = lerobot_actions - original_actions
    abs_diff = torch.abs(diff)

    for batch_idx in range(lerobot_actions.shape[0]):
        print(f"\n{'=' * 60}")
        print(f"Batch {batch_idx}")
        print(f"{'=' * 60}")
        print(f"{'Idx':<5} {'LeRobot':<14} {'Original':<14} {'Difference':<14}")
        print("-" * 60)
        for action_idx in range(lerobot_actions.shape[1]):
            lr_val = lerobot_actions[batch_idx, action_idx].item()
            orig_val = original_actions[batch_idx, action_idx].item()
            diff_val = abs(lr_val - orig_val)
            sign = "+" if (lr_val - orig_val) > 0 else "-"
            print(f"{action_idx:<5} {lr_val:>13.6f} {orig_val:>13.6f} {sign}{diff_val:>12.6f}")

    max_diff = abs_diff.max().item()
    tolerance = 0.001
    assert torch.allclose(lerobot_actions, original_actions, atol=tolerance), (
        f"Actions differ by more than tolerance ({tolerance}): max diff = {max_diff:.6f}"
    )
    print(f"\nSuccess: Actions match within tolerance ({tolerance})!")

    del lerobot_policy, lerobot_preprocessor, lerobot_postprocessor
    del original_policy, modality_config, modality_transform
    del batch, batch_lerobot, observation
    cleanup_memory()


def test_groot_forward_pass_comparison():
    """Test forward pass comparison between LeRobot and Original Groot implementations."""
    print("Test: Forward Pass Comparison (Training Mode)")

    set_seed_all(42)

    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_groot(
        from_pretrained=True
    )
    original_policy, modality_config, modality_transform = instantiate_original_groot(from_pretrained=True)

    batch = create_dummy_data()
    lerobot_policy.eval()
    original_policy.model.eval()

    print("\n[LeRobot] Running forward pass...")
    batch_lerobot = deepcopy(batch)
    batch_lerobot_processed = lerobot_preprocessor(batch_lerobot)

    set_seed_all(42)
    with torch.no_grad():
        lerobot_loss, lerobot_metrics = lerobot_policy.forward(batch_lerobot_processed)

    print(f"  Loss: {lerobot_loss.item():.6f}")

    print("\n[Original] Running forward pass...")
    observation = convert_lerobot_to_original_format(batch, modality_config)
    transformed_obs = modality_transform(observation)

    if "action" not in transformed_obs:
        action_for_forward = batch_lerobot_processed["action"]
        action_mask_for_forward = batch_lerobot_processed["action_mask"]

        # Match action horizon if needed
        if action_for_forward.shape[1] != original_policy.model.action_horizon:
            if action_for_forward.shape[1] < original_policy.model.action_horizon:
                pad_size = original_policy.model.action_horizon - action_for_forward.shape[1]
                last_action = action_for_forward[:, -1:, :]
                padding = last_action.repeat(1, pad_size, 1)
                action_for_forward = torch.cat([action_for_forward, padding], dim=1)

                mask_padding = torch.zeros(
                    action_mask_for_forward.shape[0],
                    pad_size,
                    action_mask_for_forward.shape[2],
                    dtype=action_mask_for_forward.dtype,
                    device=action_mask_for_forward.device,
                )
                action_mask_for_forward = torch.cat([action_mask_for_forward, mask_padding], dim=1)
            else:
                action_for_forward = action_for_forward[:, : original_policy.model.action_horizon, :]
                action_mask_for_forward = action_mask_for_forward[
                    :, : original_policy.model.action_horizon, :
                ]

        transformed_obs["action"] = action_for_forward
        transformed_obs["action_mask"] = action_mask_for_forward

    set_seed_all(42)
    with torch.no_grad():
        original_outputs = original_policy.model.forward(transformed_obs)

    original_loss = original_outputs["loss"]
    print(f"  Loss: {original_loss.item():.6f}")

    loss_diff = abs(lerobot_loss.item() - original_loss.item())
    loss_rel_diff = loss_diff / (abs(original_loss.item()) + 1e-8) * 100

    print("\nLoss Values:")
    print(f"  LeRobot: {lerobot_loss.item():.6f}")
    print(f"  Original: {original_loss.item():.6f}")
    print(f"  Absolute difference: {loss_diff:.6f}")
    print(f"  Relative difference: {loss_rel_diff:.2f}%")

    del lerobot_policy, lerobot_preprocessor, lerobot_postprocessor
    del original_policy, modality_config, modality_transform
    del batch, batch_lerobot, observation, transformed_obs
    cleanup_memory()
