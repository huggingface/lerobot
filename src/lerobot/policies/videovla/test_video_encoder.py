#!/usr/bin/env python
"""
Test script for PI05 with video encoder (VideoPrism).

This script creates a dummy example to test the model with video encoding enabled.
"""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.videovla.configuration_pi05 import PI05VideoConfig
from lerobot.policies.videovla.modeling_pi05 import PI05VideoPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def create_dummy_batch(
    batch_size: int = 2,
    num_frames: int = 16,
    image_size: int = 224,
    num_cameras: int = 2,
    state_dim: int = 14,
    action_dim: int = 14,
    chunk_size: int = 50,
    seq_len: int = 10,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create a dummy batch for testing."""
    batch = {}

    # Create image observations with temporal dimension [B, T, C, H, W]
    for i in range(num_cameras):
        key = f"{OBS_IMAGES}.camera_{i}"
        # Images in [0, 1] range
        batch[key] = torch.rand(batch_size, num_frames, 3, image_size, image_size, device=device)

    # Create state observation [B, state_dim]
    batch[OBS_STATE] = torch.rand(batch_size, state_dim, device=device)

    # Create language tokens and attention mask [B, seq_len]
    batch["observation.language.tokens"] = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    batch["observation.language.attention_mask"] = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Create action targets [B, chunk_size, action_dim]
    batch[ACTION] = torch.rand(batch_size, chunk_size, action_dim, device=device)

    return batch


def test_video_encoder():
    """Test the PI05 model with video encoding enabled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configuration
    batch_size = 2
    num_frames = 16
    image_size = 224
    num_cameras = 2
    state_dim = 14
    action_dim = 14
    chunk_size = 50

    # Create config with video encoder enabled
    print("Creating PI05VideoConfig with video encoder...")
    config = PI05VideoConfig(
        use_video_encoder=True,
        video_num_frames=num_frames,
        videoprism_model_name="MHRDYN7/videoprism-base-f16r288",
        videoprism_image_size=288,
        freeze_video_encoder=True,
        video_padding_mode="repeat",
        video_encoder_camera_key=f"{OBS_IMAGES}.camera_0",  # Use first camera for video
        chunk_size=chunk_size,
        max_action_dim=32,
        max_state_dim=32,
        dtype="float32",  # Use float32 for testing
        device=device,
    )

    # Set up input/output features
    for i in range(num_cameras):
        key = f"{OBS_IMAGES}.camera_{i}"
        config.input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_size, image_size),
        )

    config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(state_dim,),
    )

    config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(action_dim,),
    )

    print(f"use_video_encoder: {config.use_video_encoder}")
    print(f"video_num_frames: {config.video_num_frames}")
    print(f"video_padding_mode: {config.video_padding_mode}")
    print(f"video_encoder_camera_key: {config.video_encoder_camera_key}")
    print(f"image_observation_delta_indices: {config.image_observation_delta_indices}")

    # Create model
    model = PI05VideoPolicy(config)
    model.to(device)

    # Create dummy batch
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_frames=num_frames,
        image_size=image_size,
        num_cameras=num_cameras,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        device=device,
    )

    print(f"Batch keys: {list(batch.keys())}"  )
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    # Test forward pass
    model.train()
    try:
        loss, loss_dict = model.forward(batch)
        print(f"Forward pass successful!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test inference
    model.eval()
    with torch.no_grad():
        try:
            actions = model.predict_action_chunk(batch)
            print(f"Test pass, inference pass!")
            print(f"Predicted actions shape: {actions.shape}")
        except Exception as e:
            print(f"Inference failed: {e}")
            raise

    print("All tests passed!")


def test_frame_padding():
    """Test frame padding at episode start."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create config
    config = PI05VideoConfig(
        use_video_encoder=True,
        video_num_frames=16,
        videoprism_model_name="MHRDYN7/videoprism-base-f16r288",
        freeze_video_encoder=True,
        video_padding_mode="repeat",
        chunk_size=50,
        dtype="float32",
        device=device,
    )

    # Set up minimal features
    config.input_features[f"{OBS_IMAGES}.camera_0"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),
    )
    config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(14,),
    )

    # Create model
    model = PI05VideoPolicy(config)
    model.to(device)

    # Test with fewer frames than expected (simulating episode start)
    batch = {
        f"{OBS_IMAGES}.camera_0": torch.rand(2, 5, 3, 224, 224, device=device),
        "observation.language.tokens": torch.randint(0, 1000, (2, 10), device=device),
        "observation.language.attention_mask": torch.ones(2, 10, dtype=torch.bool, device=device),
        ACTION: torch.rand(2, 50, 14, device=device),
    }

    video_frames = model._preprocess_video(batch)
    if video_frames is not None:
        print(f"Input frames: 5")
        print(f"Output video_frames shape: {video_frames.shape}")
        print(f"Expected: [2, 16, 3, 224, 224]")
        assert video_frames.shape == (2, 16, 3, 224, 224), f"Unexpected shape: {video_frames.shape}"
        print("Frame padding test PASSED!")
    else:
        print("video_frames is None (unexpected)")

    # Test with single frame
    batch[f"{OBS_IMAGES}.camera_0"] = torch.rand(2, 3, 224, 224, device=device)  # [B, C, H, W]

    video_frames = model._preprocess_video(batch)
    if video_frames is not None:
        print(f"Input: single frame [B, C, H, W]")
        print(f"Output video_frames shape: {video_frames.shape}")
        print(f"Expected: [2, 16, 3, 224, 224]")
        assert video_frames.shape == (2, 16, 3, 224, 224), f"Unexpected shape: {video_frames.shape}"
        print("Single frame expansion test PASSED!")
    else:
        print("video_frames is None (unexpected)")

    print("All tests passed!")
if __name__ == "__main__":
    # Run tests
    test_frame_padding()
    test_video_encoder()
