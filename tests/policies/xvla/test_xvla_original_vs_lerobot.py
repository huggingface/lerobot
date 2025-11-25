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

"""Test script to verify XVLA policy integration with LeRobot vs the original implementation, only meant to be run locally!"""

import gc
import os
import random
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

# Skip if transformers is not available
pytest.importorskip("transformers")

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires XVLA model access and is not meant for CI",
)

from transformers import AutoModel, AutoProcessor  # noqa: E402

from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

# Constants
DUMMY_ACTION_DIM = 7  # Standard robot arm action dimension
DUMMY_STATE_DIM = 20  # Proprioceptive state dimension
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_VIEWS = 2  # Number of camera views
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_LEROBOT = "lerobot/xvla-widowx"
MODEL_PATH_ORIGINAL = "2toINF/X-VLA-WidowX"
LIBERO_DOMAIN_ID = 0  # Domain ID for examples purposes


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


def instantiate_lerobot_xvla(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH_LEROBOT,
) -> tuple[
    Any,  # Policy
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot XVLA policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = XVLAPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=False,
        )
    else:
        config = XVLAConfig(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_DIM,
            chunk_size=DUMMY_ACTION_DIM,
            device=DEVICE,
            num_image_views=NUM_VIEWS,
        )  # add resize_imgs_with_padding=IMAGE_SIZE, IMAGE_SIZE?
        policy = XVLAPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_xvla_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization (original XVLA doesn't normalize)
    )

    return policy, preprocessor, postprocessor


def instantiate_original_xvla(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH_ORIGINAL,
):
    """Instantiate original XVLA policy from the original implementation."""
    if from_pretrained:
        processor = AutoProcessor.from_pretrained(model_path, num_views=NUM_VIEWS, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise NotImplementedError("Non-pretrained XVLA instantiation not implemented yet")

    model.to(DEVICE)
    model.eval()

    return model, processor


def create_dummy_data(device=DEVICE):
    """Create dummy data for testing both implementations."""
    batch_size = 1
    prompt = "Pick up the red block and place it in the bin"

    # Create random RGB images in [0, 255] uint8 range (as PIL images would be)
    # Then convert to [0, 1] float32 range for LeRobot
    def fake_rgb(h, w):
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        return t

    batch = {
        f"{OBS_IMAGES}.image": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.image2": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


def prepare_original_inputs(batch, processor, device=DEVICE):
    """Prepare inputs for the original XVLA model."""
    # Convert images from [0, 1] to [0, 255] uint8 for processor
    image1 = (batch[f"{OBS_IMAGES}.image"]).byte()
    image2 = (batch[f"{OBS_IMAGES}.image2"]).byte()

    # Get task instruction (use first one if batch)
    task_instruction = batch["task"][0] if isinstance(batch["task"], list) else batch["task"]

    # Process images and text through original processor
    # The processor expects a list of images per sample
    processed_inputs = processor(
        [image1[0], image2[0]],  # Process first sample only for now
        task_instruction,
    )

    # Move to correct device and dtype
    dtype = torch.float32
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in processed_inputs.items()
    }

    # Add proprio and domain_id
    inputs.update(
        {
            "proprio": batch[OBS_STATE][:1].to(device),  # First sample only
            "domain_id": torch.tensor([LIBERO_DOMAIN_ID], dtype=torch.long, device=device),
        }
    )

    return inputs


def test_xvla_preprocessor_alignment():
    """Test that LeRobot and Original XVLA preprocessors produce similar outputs."""
    print("\n" + "=" * 80)
    print("Test: XVLA Preprocessor Alignment")
    print("=" * 80)

    set_seed_all(42)

    print("\n[LeRobot] Instantiating policy and preprocessor...")
    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_xvla(
        from_pretrained=True
    )

    print("\n[Original] Instantiating model and processor...")
    original_model, original_processor = instantiate_original_xvla(from_pretrained=True)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Preprocessing...")
    lerobot_observation = lerobot_preprocessor(deepcopy(batch))
    lerobot_inputs = lerobot_policy._build_model_inputs(lerobot_observation)

    print("\n[Original] Preprocessing...")
    original_inputs = prepare_original_inputs(batch, original_processor)

    print("\nComparing preprocessor outputs:")
    print("-" * 80)

    # Compare common keys
    common_keys = set(lerobot_inputs.keys()) & set(original_inputs.keys())
    print(f"Common keys: {common_keys}")

    for key in common_keys:
        lerobot_tensor = lerobot_inputs[key]
        original_tensor = original_inputs[key]

        print(f"\n🔎 Key: {key}")
        print(f"  LeRobot shape: {lerobot_tensor.shape}")
        print(f"  Original shape: {original_tensor.shape}")

        # Handle batch size difference (we only process first sample for original)
        if lerobot_tensor.shape[0] > original_tensor.shape[0]:
            lerobot_tensor = lerobot_tensor[:1]

        if lerobot_tensor.shape == original_tensor.shape:
            if torch.allclose(lerobot_tensor, original_tensor, atol=1e-5, rtol=1e-5):
                print("  ✔️ Tensors are equal (allclose with atol=1e-5)")
            else:
                diff = torch.abs(lerobot_tensor - original_tensor)
                print("  ⚠️ Tensors differ")
                print(f"  Max diff: {diff.max().item():.6e}")
                print(f"  Mean diff: {diff.mean().item():.6e}")
                print(f"  Std diff: {diff.std().item():.6e}")
        else:
            print("  ⚠️ Shapes don't match after alignment")

    cleanup_memory()


def test_xvla_original_vs_lerobot_pretrained():
    """Test XVLA original implementation vs LeRobot implementation with pretrained weights."""
    print("\n" + "=" * 80)
    print("Test: XVLA Original vs LeRobot with Pretrained Weights (Inference)")
    print("=" * 80)

    set_seed_all(42)

    print("\n[LeRobot] Instantiating policy...")
    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_xvla(
        from_pretrained=True
    )

    print("\n[Original] Instantiating model...")
    original_model, original_processor = instantiate_original_xvla(from_pretrained=True)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Running inference...")
    lerobot_observation = lerobot_preprocessor(deepcopy(batch))
    lerobot_inputs = lerobot_policy._build_model_inputs(lerobot_observation)

    # Reset seed for inference
    torch.manual_seed(42)
    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.generate_actions(**lerobot_inputs, steps=10)
        lerobot_actions = lerobot_actions.squeeze(0).float().cpu()

    print(f"LeRobot actions shape: {lerobot_actions.shape}")
    print(f"LeRobot actions mean: {lerobot_actions.mean().item():.6f}")
    print(f"LeRobot actions std: {lerobot_actions.std().item():.6f}")

    print("\n[Original] Running inference...")
    original_inputs = prepare_original_inputs(batch, original_processor)

    # Reset seed for inference
    torch.manual_seed(42)
    with torch.no_grad():
        original_actions = original_model.generate_actions(**original_inputs, steps=10)
        original_actions = original_actions.squeeze(0).float().cpu()

    print(f"Original actions shape: {original_actions.shape}")
    print(f"Original actions mean: {original_actions.mean().item():.6f}")
    print(f"Original actions std: {original_actions.std().item():.6f}")

    print("\nAction Comparison:")
    print("-" * 80)

    # Compare actions
    if lerobot_actions.shape == original_actions.shape:
        diff = torch.abs(lerobot_actions - original_actions)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Mean absolute difference: {mean_diff:.6e}")
        print(
            f"Relative difference: {(mean_diff / (torch.abs(original_actions).mean().item() + 1e-8) * 100):.2f}%"
        )

        # Check with different tolerances
        tolerances = [1e-5, 1e-4, 1e-3, 1e-2]
        for tol in tolerances:
            is_close = torch.allclose(lerobot_actions, original_actions, atol=tol)
            status = "✔️" if is_close else "❌"
            print(f"{status} Actions close (atol={tol}): {is_close}")

        # Assert with reasonable tolerance
        tolerance = 1e-3
        assert torch.allclose(lerobot_actions, original_actions, atol=tolerance), (
            f"Actions differ by more than tolerance ({tolerance}): max diff = {max_diff:.6e}"
        )
        print(f"\n✅ Success: Actions match within tolerance ({tolerance})!")
    else:
        print(f"⚠️ Shape mismatch: LeRobot {lerobot_actions.shape} vs Original {original_actions.shape}")

    cleanup_memory()


def test_xvla_inference_reproducibility():
    """Test that XVLA inference is reproducible with the same seed."""
    print("\n" + "=" * 80)
    print("Test: XVLA Inference Reproducibility")
    print("=" * 80)

    print("\n[LeRobot] Instantiating policy...")
    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_xvla(
        from_pretrained=True
    )

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    # First inference
    print("\n[Run 1] Running inference...")
    set_seed_all(42)
    lerobot_observation = lerobot_preprocessor(deepcopy(batch))
    lerobot_inputs = lerobot_policy._build_model_inputs(lerobot_observation)
    with torch.no_grad():
        actions_1 = lerobot_policy.model.generate_actions(**lerobot_inputs, steps=10)
        actions_1 = actions_1.squeeze(0).float().cpu()

    # Second inference with same seed
    print("\n[Run 2] Running inference with same seed...")
    set_seed_all(42)
    lerobot_observation = lerobot_preprocessor(deepcopy(batch))
    lerobot_inputs = lerobot_policy._build_model_inputs(lerobot_observation)
    with torch.no_grad():
        actions_2 = lerobot_policy.model.generate_actions(**lerobot_inputs, steps=10)
        actions_2 = actions_2.squeeze(0).float().cpu()

    print("\nComparing two runs:")
    print("-" * 80)

    if torch.allclose(actions_1, actions_2, atol=1e-8):
        print("✔️ Inference is perfectly reproducible!")
    else:
        diff = torch.abs(actions_1 - actions_2)
        print("⚠️ Small differences detected:")
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")

    assert torch.allclose(actions_1, actions_2, atol=1e-6), "Inference should be reproducible!"

    cleanup_memory()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("XVLA Original vs LeRobot Comparison Test Suite")
    print("=" * 80)

    try:
        test_xvla_preprocessor_alignment()
        test_xvla_original_vs_lerobot_pretrained()
        test_xvla_inference_reproducibility()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Test failed with error: {e}")
        print("=" * 80)
        raise
