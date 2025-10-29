#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Validate pre-computed embeddings against on-the-fly computed embeddings.

Usage:
    python src/lerobot/datasets/generating_embeddings/validate_embeddings.py \
        --original-repo-id lerobot/utokyo_xarm_bimanual \
        --embeddings-repo-id <your_username>/utokyo_xarm_bimanual_embeddings \
        --image-encoder dinov2_vitb14 \
        --language-encoder minilm-l12 \
        --num-samples 10
"""

import argparse

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.generating_embeddings.encoders import ImageEncoder, LanguageEncoder
from lerobot.datasets.generating_embeddings.generate_embeddings import (
    get_image_encoder,
    get_language_encoder,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def validate_embeddings(
    original_repo_id: str,
    embeddings_repo_id: str,
    image_encoder: ImageEncoder,
    language_encoder: LanguageEncoder,
    num_samples: int = 10,
    device: str = "cuda",
):
    """Validate pre-computed embeddings against on-the-fly embeddings.

    Args:
        original_repo_id: Original dataset repository ID
        embeddings_repo_id: Dataset with pre-computed embeddings repository ID
        image_encoder: Image encoder instance
        language_encoder: Language encoder instance
        num_samples: Number of samples to validate
        device: Device to use for encoding
    """
    # Load both datasets
    print("Loading datasets...")
    original_dataset = LeRobotDataset(original_repo_id, download_videos=True)
    embeddings_dataset = LeRobotDataset(embeddings_repo_id, download_videos=False)

    # Verify both datasets have the same number of frames
    assert original_dataset.num_frames == embeddings_dataset.num_frames, (
        f"Frame count mismatch: original={original_dataset.num_frames}, "
        f"embeddings={embeddings_dataset.num_frames}"
    )

    camera_keys = original_dataset.meta.camera_keys

    # Check embedding features exist
    expected_features = ["task_embedding"] + [f"{cam}_embedding" for cam in camera_keys]
    for feat in expected_features:
        if feat not in embeddings_dataset.features:
            raise ValueError(f"Embedding feature not found: {feat}")

    # Select random sample indices
    sample_indices = np.random.choice(
        original_dataset.num_frames, size=min(num_samples, original_dataset.num_frames), replace=False
    )
    print(f"Validating {len(sample_indices)} samples...")

    # Track statistics
    task_similarities = []
    image_similarities = {cam: [] for cam in camera_keys}

    for idx in tqdm(sample_indices, desc="Validating"):
        idx = int(idx)

        embeddings_item = embeddings_dataset[idx]
        precomputed_task_emb = embeddings_item["task_embedding"].numpy()
        precomputed_image_embs = {cam: embeddings_item[f"{cam}_embedding"].numpy() for cam in camera_keys}

        original_item = original_dataset[idx]

        # Get task and compute embedding
        task = original_item["task"]
        # Clean up task text (same as in generate_embeddings.py)
        task_clean = task.strip().capitalize().strip(" .,!?-_")
        onthefly_task_emb = language_encoder.encode([task_clean])[0]

        # Get images and compute embeddings
        onthefly_image_embs = {}
        for cam in camera_keys:
            img = original_item[cam]
            # Convert to numpy if needed
            if isinstance(img, torch.Tensor):
                if img.ndim == 3:  # (C, H, W)
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            else:
                img_np = np.array(img)

            onthefly_image_embs[cam] = image_encoder.encode([img_np])[0]

        # Task embedding comparison
        task_sim = cosine_similarity(precomputed_task_emb, onthefly_task_emb)
        task_similarities.append(task_sim)

        # Image embedding comparison
        for cam in camera_keys:
            img_sim = cosine_similarity(precomputed_image_embs[cam], onthefly_image_embs[cam])
            image_similarities[cam].append(img_sim)

    # Results
    print("\nResults:")
    task_sim_threshold = 0.99
    img_sim_threshold = 0.99

    task_mean_sim = np.mean(task_similarities)
    task_pass = task_mean_sim >= task_sim_threshold

    print(f"  Task: {task_mean_sim:.4f} {'✓' if task_pass else '✗'}")

    for cam in camera_keys:
        cam_mean_sim = np.mean(image_similarities[cam])
        cam_pass = cam_mean_sim >= img_sim_threshold
        print(f"  {cam}: {cam_mean_sim:.4f} {'✓' if cam_pass else '✗'}")

    image_pass = all(np.mean(image_similarities[cam]) >= img_sim_threshold for cam in camera_keys)

    print()
    if task_pass and image_pass:
        print("✓ PASSED")
    else:
        print("✗ FAILED")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and compare pre-computed embeddings with on-the-fly embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/lerobot/datasets/generating_embeddings/validate_embeddings.py \\
      --original-repo-id lerobot/utokyo_xarm_bimanual \\
      --embeddings-repo-id lerobot/utokyo_xarm_bimanual_embeddings \\
      --image-encoder dinov2_vitb14 \\
      --language-encoder minilm-l12 \\
      --num-samples 20
        """,
    )
    parser.add_argument("--original-repo-id", type=str, required=True, help="Original dataset repository ID")
    parser.add_argument(
        "--embeddings-repo-id",
        type=str,
        required=True,
        help="Dataset with pre-computed embeddings repository ID",
    )
    parser.add_argument(
        "--image-encoder",
        type=str,
        default="dinov2_vitb14",
        help="Image encoder to use (default: dinov2_vitb14)",
    )
    parser.add_argument(
        "--language-encoder",
        type=str,
        default="minilm-l12",
        help="Language encoder to use (default: minilm-l12)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to validate (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding (default: cuda)",
    )

    args = parser.parse_args()

    # Load encoders
    image_encoder = get_image_encoder(args.image_encoder, device=args.device)
    language_encoder = get_language_encoder(args.language_encoder, device=args.device)

    # Validate embeddings
    validate_embeddings(
        original_repo_id=args.original_repo_id,
        embeddings_repo_id=args.embeddings_repo_id,
        image_encoder=image_encoder,
        language_encoder=language_encoder,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
