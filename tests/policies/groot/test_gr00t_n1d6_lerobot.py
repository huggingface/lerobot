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

"""Test script for LeRobot's Groot N1.6 policy forward and inference passes."""

import gc

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config
from lerobot.policies.gr00t_n1d6.modeling_gr00t_n1d6 import Gr00tN1d6Policy
from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import Gr00tN1d6Processor, VLAStepData
from lerobot.policies.gr00t_n1d6.utils import EmbodimentTag
from lerobot.utils.utils import auto_select_torch_device
from tests.utils import require_cuda

# pytestmark = pytest.mark.skipif(
#     os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
#     reason="This test requires local Groot N1.6 installation and is not meant for CI",
# )


# Define constants for dummy data
# NOTE: When loading from pretrained N1.6 model, it expects max_state_dim=128 and max_action_dim=128
# We use smaller values (29) for user-facing config, but the model pads internally to 128
DUMMY_STATE_DIM = 29  # User-specified state dimension (will be padded to 128 internally)
DUMMY_ACTION_DIM = 29  # User-specified action dimension (will be padded to 128 internally)
PRETRAINED_MAX_STATE_DIM = 128  # Pretrained model's internal max state dimension
PRETRAINED_MAX_ACTION_DIM = 128  # Pretrained model's internal max action dimension
DUMMY_ACTION_HORIZON = 16  # N1.6 action_horizon
DUMMY_CHUNK_SIZE = 40  # N1.6 chunk_size (max_action_horizon)
IMAGE_SIZE = 224  # N1.6 default image size
DEVICE = auto_select_torch_device()
MODEL_PATH = "nvidia/GR00T-N1.6-3B"


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


def instantiate_lerobot_gr00t_n1d6(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH,
) -> tuple[Gr00tN1d6Policy, Gr00tN1d6Processor]:
    """Instantiate LeRobot Groot N1.6 policy with processor."""
    if from_pretrained:
        # Create config and load pretrained weights
        config = Gr00tN1d6Config(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_HORIZON,
            chunk_size=DUMMY_CHUNK_SIZE,
            action_horizon=DUMMY_ACTION_HORIZON,
            max_state_dim=DUMMY_STATE_DIM,
            max_action_dim=DUMMY_ACTION_DIM,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            embodiment_tag="new_embodiment",
            input_features={
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(DUMMY_STATE_DIM,),
                ),
                "observation.images.ego_view": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, IMAGE_SIZE, IMAGE_SIZE),
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(DUMMY_ACTION_DIM,),
                ),
            },
        )
        policy = Gr00tN1d6Policy(config)
    else:
        config = Gr00tN1d6Config(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_HORIZON,
            chunk_size=DUMMY_CHUNK_SIZE,
            action_horizon=DUMMY_ACTION_HORIZON,
            max_state_dim=DUMMY_STATE_DIM,
            max_action_dim=DUMMY_ACTION_DIM,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            embodiment_tag="new_embodiment",
            input_features={
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(DUMMY_STATE_DIM,),
                ),
                "observation.images.ego_view": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, IMAGE_SIZE, IMAGE_SIZE),
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(DUMMY_ACTION_DIM,),
                ),
            },
        )
        policy = Gr00tN1d6Policy(config)

    policy.to(DEVICE)

    # Create processor with modality configs for the embodiment
    modality_configs = {
        "new_embodiment": {
            "state": {
                "modality_keys": ["joints"],
                "delta_indices": list(range(1)),  # Single observation step
            },
            "action": {
                "modality_keys": ["joints"],
                "delta_indices": list(range(DUMMY_ACTION_HORIZON)),
            },
            "video": {
                "modality_keys": ["ego_view"],
                "delta_indices": list(range(1)),
            },
        }
    }

    # Create dummy statistics for normalization
    statistics = {
        "new_embodiment": {
            "state": {
                "joints": {
                    "min": [-1.0] * DUMMY_STATE_DIM,
                    "max": [1.0] * DUMMY_STATE_DIM,
                    "mean": [0.0] * DUMMY_STATE_DIM,
                    "std": [1.0] * DUMMY_STATE_DIM,
                    "q01": [-1.0] * DUMMY_STATE_DIM,
                    "q99": [1.0] * DUMMY_STATE_DIM,
                }
            },
            "action": {
                "joints": {
                    "min": [-1.0] * DUMMY_ACTION_DIM,
                    "max": [1.0] * DUMMY_ACTION_DIM,
                    "mean": [0.0] * DUMMY_ACTION_DIM,
                    "std": [1.0] * DUMMY_ACTION_DIM,
                    "q01": [-1.0] * DUMMY_ACTION_DIM,
                    "q99": [1.0] * DUMMY_ACTION_DIM,
                }
            },
        }
    }

    processor = Gr00tN1d6Processor(
        modality_configs=modality_configs,
        statistics=statistics,
        max_state_dim=DUMMY_STATE_DIM,
        max_action_dim=DUMMY_ACTION_DIM,
        max_action_horizon=DUMMY_CHUNK_SIZE,
        use_relative_action=False,  # Disable relative action for simpler testing
        formalize_language=True,
    )
    processor.eval()

    return (policy, processor)


def create_dummy_data(device=DEVICE, batch_size: int = 2, for_training: bool = True):
    """Create a dummy data batch for testing.

    For N1.6, we create data in the VLAStepData format that the processor expects.
    """
    prompt = "Pick up the red cube and place it in the bin"

    # Create dummy images as numpy arrays (H, W, C) format for processor
    images = {
        "ego_view": [
            (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)
            for _ in range(1)  # Single observation step
        ]
    }

    # Create dummy states
    states = {
        "joints": np.random.randn(1, DUMMY_STATE_DIM).astype(np.float32),  # (T, D)
    }

    # Create dummy actions (only for training)
    if for_training:
        actions = {
            "joints": np.random.randn(DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM).astype(np.float32),
        }
    else:
        actions = {}

    # Create VLAStepData for each batch item
    step_data_list = []
    for _ in range(batch_size):
        step_data = VLAStepData(
            images=images,
            states=states,
            actions=actions,
            text=prompt,
            embodiment=EmbodimentTag.NEW_EMBODIMENT,
        )
        step_data_list.append(step_data)

    return step_data_list


def preprocess_batch(processor: Gr00tN1d6Processor, step_data_list: list[VLAStepData]) -> dict:
    """Preprocess a batch of VLAStepData using the processor and collator."""
    # Process each item through the processor
    processed_items = []
    for step_data in step_data_list:
        processed = processor([{"content": step_data}])
        processed_items.append(processed)

    # Use collator to batch
    batch = processor.collator(processed_items)

    # Extract the inputs from BatchFeature
    return batch["inputs"]


@require_cuda
def test_lerobot_gr00t_n1d6_inference():
    """Test the inference pass (select_action) of LeRobot's Groot N1.6 policy."""
    print("Test: LeRobot Groot N1.6 Inference Pass")

    set_seed_all(42)

    # Instantiate policy and processor
    lerobot_policy, lerobot_processor = instantiate_lerobot_gr00t_n1d6(from_pretrained=True)

    # Create dummy data for inference (no actions needed)
    step_data_list = create_dummy_data(for_training=False)

    print("\n[LeRobot N1.6] Running inference...")
    lerobot_policy.eval()
    lerobot_processor.eval()

    # Preprocess the batch
    batch = preprocess_batch(lerobot_processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Ensure identical RNG state before inference
    torch.manual_seed(42)

    with torch.no_grad():
        lerobot_action = lerobot_policy.select_action(batch)

    print(f"\nInference successful. Output action shape: {lerobot_action.shape}")
    print("Output actions (first 5 dims):")
    print(lerobot_action[:, :5])

    del lerobot_policy, lerobot_processor, batch
    cleanup_memory()


@require_cuda
def test_lerobot_gr00t_n1d6_forward_pass():
    """Test the forward pass of LeRobot's Groot N1.6 policy."""
    print("\n" + "=" * 50)
    print("Test: LeRobot Groot N1.6 Forward Pass (Training Mode)")

    set_seed_all(42)

    # Instantiate policy and processor
    lerobot_policy, lerobot_processor = instantiate_lerobot_gr00t_n1d6(from_pretrained=True)

    # Create dummy data for training (with actions)
    step_data_list = create_dummy_data(for_training=True)

    lerobot_policy.train()
    lerobot_processor.train()

    print("\n[LeRobot N1.6] Running forward pass...")

    # Preprocess the batch
    batch = preprocess_batch(lerobot_processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    set_seed_all(42)
    lerobot_loss, lerobot_metrics = lerobot_policy.forward(batch)

    print("\nForward pass successful.")
    print(f"  - Loss: {lerobot_loss.item():.6f}")
    print(f"  - Metrics: {lerobot_metrics}")

    del lerobot_policy, lerobot_processor, batch
    cleanup_memory()


@require_cuda
def test_lerobot_gr00t_n1d6_predict_action_chunk():
    """Test the predict_action_chunk method of LeRobot's Groot N1.6 policy."""
    print("\n" + "=" * 50)
    print("Test: LeRobot Groot N1.6 Predict Action Chunk")

    set_seed_all(42)

    # Instantiate policy and processor
    lerobot_policy, lerobot_processor = instantiate_lerobot_gr00t_n1d6(from_pretrained=True)

    # Create dummy data for inference
    step_data_list = create_dummy_data(for_training=False)

    print("\n[LeRobot N1.6] Running predict_action_chunk...")
    lerobot_policy.eval()
    lerobot_processor.eval()

    # Preprocess the batch
    batch = preprocess_batch(lerobot_processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Ensure identical RNG state before inference
    torch.manual_seed(42)

    with torch.no_grad():
        action_chunk = lerobot_policy.predict_action_chunk(batch)

    print("\nPredict action chunk successful.")
    print(f"  - Action chunk shape: {action_chunk.shape}")
    print(
        f"  - Expected shape: (batch_size={len(step_data_list)}, n_action_steps={DUMMY_ACTION_HORIZON}, action_dim={DUMMY_ACTION_DIM})"
    )

    # Verify shape
    assert action_chunk.shape[0] == len(step_data_list), "Batch size mismatch"
    assert action_chunk.shape[2] == DUMMY_ACTION_DIM, "Action dimension mismatch"

    del lerobot_policy, lerobot_processor, batch
    cleanup_memory()


@require_cuda
def test_lerobot_gr00t_n1d6_reset():
    """Test the reset method of LeRobot's Groot N1.6 policy."""
    print("\n" + "=" * 50)
    print("Test: LeRobot Groot N1.6 Reset")

    set_seed_all(42)

    # Instantiate policy and processor
    lerobot_policy, lerobot_processor = instantiate_lerobot_gr00t_n1d6(from_pretrained=True)

    # Create dummy data for inference
    step_data_list = create_dummy_data(for_training=False, batch_size=1)

    lerobot_policy.eval()
    lerobot_processor.eval()

    # Preprocess the batch
    batch = preprocess_batch(lerobot_processor, step_data_list)

    # Move batch to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Call select_action to fill the action queue
    with torch.no_grad():
        _ = lerobot_policy.select_action(batch)

    # Action queue should not be empty
    assert len(lerobot_policy._action_queue) > 0, "Action queue should not be empty after select_action"
    print(f"Action queue length after select_action: {len(lerobot_policy._action_queue)}")

    # Reset should clear the action queue
    lerobot_policy.reset()
    assert len(lerobot_policy._action_queue) == 0, "Action queue should be empty after reset"
    print("Reset successful - action queue cleared")

    del lerobot_policy, lerobot_processor, batch
    cleanup_memory()
