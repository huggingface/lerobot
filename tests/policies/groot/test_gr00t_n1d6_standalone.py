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

"""Standalone test script for LeRobot's Groot N1.6 policy.

This test can be run independently to generate outputs for comparison with the
original Isaac-GR00T implementation. Run this in the LeRobot environment:

    cd ~/gr00t_lerobot/lerobot && source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-groot
    python tests/policies/groot/test_gr00t_n1d6_standalone.py

This test verifies the LeRobot implementation's inference and forward pass.
"""

import gc

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config
from lerobot.policies.gr00t_n1d6.modeling_gr00t_n1d6 import Gr00tN1d6Policy
from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import Gr00tN1d6Processor, VLAStepData
from lerobot.policies.gr00t_n1d6.utils import EmbodimentTag
from lerobot.utils.utils import auto_select_torch_device

# Define constants for dummy data
# NOTE: When loading from pretrained N1.6 model, it expects max_state_dim=128 and max_action_dim=128
# The model pads internally to 128. Actual dimensions come from GR1 pretrained config.
DUMMY_STATE_DIM = 29  # Fallback dimension if norm_params not available
DUMMY_ACTION_DIM = 29  # Fallback dimension if norm_params not available
DUMMY_ACTION_HORIZON = 16  # N1.6 action_horizon
DUMMY_CHUNK_SIZE = 40  # N1.6 chunk_size (max_action_horizon)
IMAGE_SIZE = 224  # N1.6 default image size
DEVICE = auto_select_torch_device()
MODEL_PATH = "nvidia/GR00T-N1.6-3B"
BATCH_SIZE = 2
SEED = 42


def cleanup_memory():
    """Clean up GPU/MPS memory to prevent OOM errors."""
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


def instantiate_lerobot_gr00t_n1d6(
    model_path: str = MODEL_PATH,
) -> tuple[Gr00tN1d6Policy, Gr00tN1d6Processor]:
    """Instantiate LeRobot Groot N1.6 policy with processor using GR1 embodiment."""
    # Load processor from pretrained model (gets GR1 modality_configs and statistics)
    processor = Gr00tN1d6Processor.from_pretrained(model_path)
    processor.eval()

    # Get GR1 modality configs and norm_params from processor
    modality_configs = processor.modality_configs
    norm_params = processor.state_action_processor.norm_params
    embodiment_tag = "gr1"

    # Debug: Print available keys
    if embodiment_tag in norm_params:
        print(f"Available norm_params keys for {embodiment_tag}: {list(norm_params[embodiment_tag].keys())}")
        if "action" in norm_params[embodiment_tag]:
            print(f"Available action keys in norm_params: {list(norm_params[embodiment_tag]['action'].keys())}")
        if "state" in norm_params[embodiment_tag]:
            print(f"Available state keys in norm_params: {list(norm_params[embodiment_tag]['state'].keys())}")
    print(f"Modality config action keys: {modality_configs[embodiment_tag]['action'].modality_keys}")
    print(f"Modality config state keys: {modality_configs[embodiment_tag]['state'].modality_keys}")

    # Get actual dimensions from GR1 norm_params for config
    # Sum up all state dimensions (before sin/cos encoding)
    total_state_dim = 0
    if embodiment_tag in norm_params and "state" in norm_params[embodiment_tag]:
        for key in modality_configs[embodiment_tag]["state"].modality_keys:
            if key in norm_params[embodiment_tag]["state"]:
                dim_val = norm_params[embodiment_tag]["state"][key]["dim"]
                if isinstance(dim_val, torch.Tensor):
                    dim_val = dim_val.item()
                total_state_dim += int(dim_val)
                print(f"  State key '{key}': dim={int(dim_val)}, total={total_state_dim}")

    # Sum up all action dimensions
    total_action_dim = 0
    if embodiment_tag in norm_params and "action" in norm_params[embodiment_tag]:
        for key in modality_configs[embodiment_tag]["action"].modality_keys:
            if key in norm_params[embodiment_tag]["action"]:
                dim_val = norm_params[embodiment_tag]["action"][key]["dim"]
                if isinstance(dim_val, torch.Tensor):
                    dim_val = dim_val.item()
                total_action_dim += int(dim_val)
                print(f"  Action key '{key}': dim={int(dim_val)}, total={total_action_dim}")
    
    # Fallback if computation failed
    if total_state_dim == 0:
        print(f"Warning: total_state_dim is 0, using fallback {DUMMY_STATE_DIM}")
        total_state_dim = DUMMY_STATE_DIM
    if total_action_dim == 0:
        print(f"Warning: total_action_dim is 0, using fallback {DUMMY_ACTION_DIM}")
        total_action_dim = DUMMY_ACTION_DIM
    
    print(f"Final GR1 dimensions: state_dim={total_state_dim}, action_dim={total_action_dim}")

    # Get video key from modality configs
    video_keys = modality_configs[embodiment_tag]["video"].modality_keys
    video_key = video_keys[0].replace("video.", "").replace("observation.images.", "")

    # Create config with GR1 embodiment tag
    config = Gr00tN1d6Config(
        base_model_path=model_path,
        n_action_steps=DUMMY_ACTION_HORIZON,
        chunk_size=DUMMY_CHUNK_SIZE,
        action_horizon=DUMMY_ACTION_HORIZON,
        max_state_dim=128,  # Model uses 128 internally
        max_action_dim=128,  # Model uses 128 internally
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        embodiment_tag=embodiment_tag,
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(total_state_dim,),
            ),
            f"observation.images.{video_key}": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMAGE_SIZE, IMAGE_SIZE),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(total_action_dim,),
            ),
        },
    )
    policy = Gr00tN1d6Policy(config)
    policy.to(DEVICE)

    return (policy, processor)


def create_dummy_data_lerobot(
    batch_size: int = BATCH_SIZE,
    for_training: bool = True,
    modality_configs=None,
    norm_params=None,
    embodiment_tag: str = "gr1",
):
    """Create dummy data in VLAStepData format compatible with LeRobot processor.

    IMPORTANT: This function generates random data in the EXACT same order as the
    original Isaac-GR00T test to ensure identical inputs with the same seed.

    Args:
        batch_size: Number of samples in batch
        for_training: Whether to include actions (for training) or not (for inference)
        modality_configs: Modality configs from processor to determine correct keys
        norm_params: Normalization parameters from processor (contains raw dimensions for each key)
        embodiment_tag: Embodiment tag string (e.g., "gr1")

    Returns:
        List of VLAStepData objects
    """
    prompt = "Pick up the red cube and place it in the bin"

    # Get keys from modality configs if provided
    if modality_configs is not None:
        video_keys = modality_configs["video"].modality_keys
        state_keys = modality_configs["state"].modality_keys
        action_keys = modality_configs["action"].modality_keys
    else:
        # Default keys for GR1
        video_keys = ["ego_view"]
        state_keys = ["state"]
        action_keys = ["joints"]

    # IMPORTANT: Generate random data in the SAME order as original test for reproducibility
    # Original generates: video (B,T,H,W,C), then state (B,T,D) - all in one batch array
    
    # Generate video data for all batch items at once (B, T=1, H, W, C)
    video_data_all = {}
    for key in video_keys:
        clean_key = key.replace("video.", "").replace("observation.images.", "")
        # Generate exactly like original: (batch_size, 1, H, W, C)
        video_data_all[clean_key] = (np.random.rand(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)

    # Generate state data for all batch items at once (B, T=1, D)
    state_data_all = {}
    for key in state_keys:
        clean_key = key.replace("state.", "")
        # Get raw dimension from norm_params
        if norm_params and embodiment_tag in norm_params and "state" in norm_params[embodiment_tag]:
            if key in norm_params[embodiment_tag]["state"]:
                dim = int(norm_params[embodiment_tag]["state"][key]["dim"].item())
            else:
                dim = DUMMY_STATE_DIM
        else:
            dim = DUMMY_STATE_DIM
        # Generate exactly like original: (batch_size, 1, dim)
        state_data_all[clean_key] = np.random.randn(batch_size, 1, dim).astype(np.float32)

    # Generate action data for all batch items at once if training
    action_data_all = {}
    if for_training:
        for key in action_keys:
            clean_key = key.replace("action.", "")
            # Get raw dimension from norm_params
            if norm_params and embodiment_tag in norm_params and "action" in norm_params[embodiment_tag]:
                if key in norm_params[embodiment_tag]["action"]:
                    dim = int(norm_params[embodiment_tag]["action"][key]["dim"].item())
                else:
                    dim = DUMMY_ACTION_DIM
            else:
                dim = DUMMY_ACTION_DIM
            # Generate for all batch items: (batch_size, horizon, dim)
            action_data_all[clean_key] = np.random.randn(batch_size, DUMMY_ACTION_HORIZON, dim).astype(np.float32)

    # Create VLAStepData for each batch item, slicing from batch arrays
    step_data_list = []
    for batch_idx in range(batch_size):
        # Slice images for this batch item: (H, W, C) per timestep
        images = {}
        for key in video_keys:
            clean_key = key.replace("video.", "").replace("observation.images.", "")
            # Convert from (B, T, H, W, C) to list of (H, W, C) for this batch item
            images[clean_key] = [video_data_all[clean_key][batch_idx, t] for t in range(1)]

        # Slice states for this batch item: (T, D)
        states = {}
        for key in state_keys:
            clean_key = key.replace("state.", "")
            # Convert from (B, T, D) to (T, D) for this batch item
            states[clean_key] = state_data_all[clean_key][batch_idx]  # Shape: (T=1, D)

        # Slice actions for this batch item: (horizon, D)
        actions = {}
        if for_training:
            for key in action_keys:
                clean_key = key.replace("action.", "")
                # Convert from (B, horizon, D) to (horizon, D) for this batch item
                actions[clean_key] = action_data_all[clean_key][batch_idx]

        step_data = VLAStepData(
            images=images,
            states=states,
            actions=actions,
            text=prompt,
            embodiment=EmbodimentTag.GR1,  # Use GR1 to match policy
        )
        step_data_list.append(step_data)

    return step_data_list


def preprocess_batch(processor: Gr00tN1d6Processor, step_data_list: list[VLAStepData]) -> dict:
    """Preprocess a batch of VLAStepData using the processor and collator.

    Args:
        processor: Gr00tN1d6Processor instance
        step_data_list: List of VLAStepData objects

    Returns:
        Dictionary with processed inputs ready for model
    """
    # Store raw states for relative->absolute action conversion
    raw_states_list = []
    for step_data in step_data_list:
        raw_states_list.append(step_data.states)

    # Process each item through the processor
    processed_items = []
    for step_data in step_data_list:
        processed = processor([{"content": step_data}])
        processed_items.append(processed)

    # Use collator to batch
    batch = processor.collator(processed_items)

    # Extract the inputs from BatchFeature
    inputs = batch["inputs"]

    # Add raw_state for relative->absolute action conversion in predict_action_chunk
    # Stack raw states: each is dict[str, np.ndarray(T, D)] -> batch to dict[str, np.ndarray(B, T, D)]
    batched_raw_states = {}
    for key in raw_states_list[0].keys():
        batched_raw_states[key] = np.stack([s[key] for s in raw_states_list], axis=0)
    inputs["raw_state"] = batched_raw_states

    return inputs


def run_lerobot_inference(policy: Gr00tN1d6Policy, processor: Gr00tN1d6Processor, step_data_list: list[VLAStepData]):
    """Run inference and return actions.

    Args:
        policy: Gr00tN1d6Policy instance
        processor: Gr00tN1d6Processor instance
        step_data_list: List of VLAStepData objects

    Returns:
        Tensor of predicted actions with shape (batch_size, action_dim)
    """
    policy.eval()
    processor.eval()

    # Preprocess the batch
    batch = preprocess_batch(processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Ensure identical RNG state before inference
    torch.manual_seed(SEED)

    with torch.no_grad():
        lerobot_action = policy.select_action(batch)

    return lerobot_action


def run_lerobot_forward(policy: Gr00tN1d6Policy, processor: Gr00tN1d6Processor, step_data_list: list[VLAStepData]):
    """Run forward pass and return loss.

    Args:
        policy: Gr00tN1d6Policy instance
        processor: Gr00tN1d6Processor instance
        step_data_list: List of VLAStepData objects

    Returns:
        Loss value as float
    """
    policy.train()
    processor.train()

    # Preprocess the batch
    batch = preprocess_batch(processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    set_seed_all(SEED)
    lerobot_loss, lerobot_metrics = policy.forward(batch)

    return lerobot_loss.item()


def main():
    """Main test function that runs inference and forward pass."""
    print("=" * 70)
    print("=== Gr00t N1.6 LeRobot Test Results ===")
    print("=" * 70)
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Embodiment: GR1")
    print(f"Action horizon: {DUMMY_ACTION_HORIZON}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Seed: {SEED}")
    print()

    # Set seed for reproducibility
    set_seed_all(SEED)

    # Instantiate policy and processor
    print("Instantiating LeRobot Gr00tN1d6Policy...")
    lerobot_policy, lerobot_processor = instantiate_lerobot_gr00t_n1d6()
    print("✓ Policy and processor instantiated\n")

    # Get modality configs and processor configs from processor
    modality_configs = lerobot_processor.modality_configs["gr1"]
    norm_params = lerobot_processor.state_action_processor.norm_params
    embodiment_tag = "gr1"

    # --- Inference Test ---
    print("-" * 70)
    print("--- Inference Test ---")
    print("-" * 70)

    # Create dummy data for inference (no actions needed)
    step_data_list = create_dummy_data_lerobot(
        for_training=False,
        modality_configs=modality_configs,
        norm_params=norm_params,
        embodiment_tag=embodiment_tag,
    )

    print("Running inference...")
    lerobot_action = run_lerobot_inference(lerobot_policy, lerobot_processor, step_data_list)

    # Get action dimensions per joint group from norm_params (same as original)
    action_keys = modality_configs["action"].modality_keys
    action_dims = {}
    for key in action_keys:
        if key in norm_params[embodiment_tag]["action"]:
            dim_val = norm_params[embodiment_tag]["action"][key]["dim"]
            if isinstance(dim_val, torch.Tensor):
                dim_val = dim_val.item()
            action_dims[key] = int(dim_val)

    # Print per-joint-group like the original test
    print(f"\nAction shape: {lerobot_action.shape}")
    offset = 0
    for action_key in action_keys:
        dim = action_dims.get(action_key, 7)
        action_slice = lerobot_action[:, offset:offset + dim].cpu().numpy()
        
        print(f"\nAction key: {action_key}")
        print(f"Action shape: {action_slice.shape}")
        print("Action values (first 5 dims per batch):")
        for batch_idx in range(action_slice.shape[0]):
            action_values = action_slice[batch_idx, :5]
            print(f"  Batch {batch_idx}: [{', '.join(f'{v:.6f}' for v in action_values)}]")

        print("\nFull action values:")
        for batch_idx in range(action_slice.shape[0]):
            action_values = action_slice[batch_idx]
            print(f"  Batch {batch_idx}: [{', '.join(f'{v:.6f}' for v in action_values)}]")
        
        offset += dim

    # --- Forward Pass Test ---
    print("\n" + "-" * 70)
    print("--- Forward Pass Test ---")
    print("-" * 70)

    # Create dummy data for training (with actions)
    step_data_list = create_dummy_data_lerobot(
        for_training=True,
        modality_configs=modality_configs,
        norm_params=norm_params,
        embodiment_tag=embodiment_tag,
    )

    print("Running forward pass...")
    lerobot_loss = run_lerobot_forward(lerobot_policy, lerobot_processor, step_data_list)

    print(f"\nLoss: {lerobot_loss:.6f}")

    # Cleanup
    del lerobot_policy, lerobot_processor
    cleanup_memory()

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

