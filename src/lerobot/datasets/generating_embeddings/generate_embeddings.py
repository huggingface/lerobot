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
Generate embeddings for LeRobot datasets to make them more lightweight and efficient.

This script:
1. Loads a v3.0 LeRobot dataset from the hub
2. Computes embeddings for tasks (language commands) and frames (images)
3. Stores embeddings as new features in the dataset
4. Optionally removes video files to reduce size
5. Pushes the converted dataset to the hub

Current supported encoders:
- Image: DinoV2 (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
- Language: MiniLM (minilm-l6, minilm-l12)

The architecture is extensible - you can add more encoders by:
1. Creating a new encoder class inheriting from ImageEncoder or LanguageEncoder
2. Implementing the encode() method and embedding_dim property
3. Adding it to the get_image_encoder() or get_language_encoder() factory function

Usage example:
    python src/lerobot/datasets/generating_embeddings/generate_embeddings.py \
        --repo-id lerobot/utokyo_xarm_bimanual \
        --output-repo-id lerobot/utokyo_xarm_bimanual_embeddings \
        --image-encoder dinov2_vitb14 \
        --language-encoder minilm-l12 \
        --remove-videos \
        --push-to-hub
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.generating_embeddings.encoders import (
    DinoV2Encoder,
    ImageEncoder,
    LanguageEncoder,
    MiniLMEncoder,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_image_encoder(encoder_name: str, device: str = "cuda") -> ImageEncoder:
    """Factory function to get image encoder.

    To add a new encoder:
    1. Create a new class inheriting from ImageEncoder
    2. Implement encode() and embedding_dim property
    3. Add it to the encoders dictionary below
    """
    encoders = {
        "dinov2_vits14": lambda: DinoV2Encoder(model_name="dinov2_vits14", device=device),
        "dinov2_vitb14": lambda: DinoV2Encoder(model_name="dinov2_vitb14", device=device),
        "dinov2_vitl14": lambda: DinoV2Encoder(model_name="dinov2_vitl14", device=device),
    }

    if encoder_name not in encoders:
        raise ValueError(f"Unknown image encoder: {encoder_name}. Available options: {list(encoders.keys())}")

    return encoders[encoder_name]()


def get_language_encoder(encoder_name: str, device: str = "cuda") -> LanguageEncoder:
    """Factory function to get language encoder.

    To add a new encoder:
    1. Create a new class inheriting from LanguageEncoder
    2. Implement encode() and embedding_dim property
    3. Add it to the encoders dictionary below
    """
    encoders = {
        "minilm-l6": lambda: MiniLMEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
        ),
        "minilm-l12": lambda: MiniLMEncoder(
            model_name="sentence-transformers/all-MiniLM-L12-v2", device=device
        ),
    }

    if encoder_name not in encoders:
        raise ValueError(
            f"Unknown language encoder: {encoder_name}. Available options: {list(encoders.keys())}"
        )

    return encoders[encoder_name]()


def generate_embeddings_for_dataset(
    repo_id: str,
    output_repo_id: str,
    image_encoder: ImageEncoder,
    language_encoder: LanguageEncoder,
    remove_videos: bool = False,
    local_dir: Path | None = None,
    output_local_dir: Path | None = None,
    push_to_hub: bool = False,
):
    """Generate embeddings for a LeRobot dataset.

    Args:
        repo_id: Source dataset repository ID
        output_repo_id: Output dataset repository ID
        image_encoder: Image encoder instance
        language_encoder: Language encoder instance
        remove_videos: Whether to remove video files
        local_dir: Local directory for source dataset
        output_local_dir: Local directory for output dataset
        push_to_hub: Whether to push to hub after conversion
    """
    from lerobot.datasets.dataset_tools import modify_features

    print(f"Loading dataset: {repo_id}")

    dataset = LeRobotDataset(repo_id, root=local_dir, download_videos=True)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    print("Computing task embeddings...")
    unique_tasks = dataset.meta.tasks.index.tolist()
    task_embeddings = {}

    for task in tqdm(unique_tasks, desc="Encoding tasks"):
        # Clean up task text
        task_clean = task.strip().capitalize().strip(" .,!?-_")
        embedding = language_encoder.encode([task_clean])[0]
        task_embeddings[task] = embedding

    print(f"Computed {len(task_embeddings)} task embeddings")

    print("Processing frames and computing embeddings...")
    all_task_embeddings = []
    all_image_embeddings_dict = {cam_key: [] for cam_key in dataset.meta.camera_keys}

    for frame_idx in tqdm(range(dataset.num_frames), desc="Processing frames"):
        item = dataset.hf_dataset[frame_idx]
        ep_idx = item["episode_index"].item()

        task = dataset.meta.tasks.iloc[item["task_index"].item()].name
        task_emb = task_embeddings[task]
        all_task_embeddings.append(task_emb)

        for cam_key in dataset.meta.camera_keys:
            if cam_key in dataset.meta.video_keys:
                current_ts = item["timestamp"].item()
                video_frames = dataset._query_videos({cam_key: [current_ts]}, ep_idx)
                img = video_frames[cam_key]

                if isinstance(img, torch.Tensor):
                    if img.ndim == 4:
                        img = img[0]  # (T, C, H, W) -> (C, H, W)
                    elif img.ndim != 3:
                        raise ValueError(f"Unexpected video frame shape {img.shape} for camera {cam_key}")
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    img_np = np.array(img)
            else:
                img = item[cam_key]
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    else:
                        raise ValueError(f"Unexpected image shape {img.shape} for camera {cam_key}")
                else:
                    img_np = np.array(img)

            all_image_embeddings_dict[cam_key].append(img_np)

    print("Computing image embeddings...")
    image_embeddings_dict = {}
    for cam_key, images in all_image_embeddings_dict.items():
        print(f"  {cam_key}: {len(images)} images")
        embeddings = image_encoder.encode(images)
        image_embeddings_dict[cam_key] = embeddings

    all_task_embeddings = np.array(all_task_embeddings)
    for cam_key in dataset.meta.camera_keys:
        image_embeddings_dict[cam_key] = np.array(image_embeddings_dict[cam_key])

    img_emb_dim = image_encoder.embedding_dim
    lang_emb_dim = language_encoder.embedding_dim

    add_features_dict = {
        "task_embedding": (
            all_task_embeddings,
            {"dtype": "float32", "shape": [lang_emb_dim], "names": None},
        ),
    }

    for cam_key in dataset.meta.camera_keys:
        add_features_dict[f"{cam_key}_embedding"] = (
            image_embeddings_dict[cam_key],
            {"dtype": "float32", "shape": [img_emb_dim], "names": None},
        )

    print("Adding embeddings to dataset...")
    remove_features_list = None
    if remove_videos:
        remove_features_list = dataset.meta.video_keys

    output_dataset = modify_features(
        dataset=dataset,
        add_features=add_features_dict,
        remove_features=remove_features_list,
        output_dir=output_local_dir,
        repo_id=output_repo_id,
    )

    if remove_videos:
        print("Removing video files...")
        videos_dir = output_dataset.root / "videos"
        if videos_dir.exists():
            shutil.rmtree(videos_dir)

    print(f"Saved to: {output_dataset.root}")

    if push_to_hub:
        print(f"Pushing to hub: {output_repo_id}")
        output_dataset.push_to_hub(push_videos=not remove_videos)
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for LeRobot datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default encoders (DinoV2 ViT-B/14 + MiniLM-L12)
  python src/lerobot/datasets/generating_embeddings/generate_embeddings.py \\
      --repo-id lerobot/utokyo_xarm_bimanual \\
      --output-repo-id your-username/utokyo_xarm_bimanual_embeddings \\
      --image-encoder dinov2_vitb14 \\
      --language-encoder minilm-l12 \\
      --push-to-hub

  # Generate embeddings and remove videos
  python src/lerobot/datasets/generating_embeddings/generate_embeddings.py \\
      --repo-id lerobot/utokyo_xarm_bimanual \\
      --output-repo-id your-username/utokyo_xarm_bimanual_lightweight \\
      --image-encoder dinov2_vitb14 \\
      --language-encoder minilm-l12 \\
      --remove-videos \\
      --push-to-hub

Available image encoders:
  - dinov2_vits14: DinoV2 ViT-S/14 (384-dim, faster)
  - dinov2_vitb14: DinoV2 ViT-B/14 (768-dim, recommended)
  - dinov2_vitl14: DinoV2 ViT-L/14 (1024-dim, best quality)

Available language encoders:
  - minilm-l6: MiniLM-L6-v2 (384-dim, faster)
  - minilm-l12: MiniLM-L12-v2 (384-dim, recommended)
        """,
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repository ID")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Output dataset repository ID")
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
        "--remove-videos",
        action="store_true",
        help="Remove video files after generating embeddings",
    )
    parser.add_argument("--local-dir", type=str, default=None, help="Local directory for source dataset")
    parser.add_argument(
        "--output-local-dir", type=str, default=None, help="Local directory for output dataset"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted dataset to the hub",
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

    # Generate embeddings
    generate_embeddings_for_dataset(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        image_encoder=image_encoder,
        language_encoder=language_encoder,
        remove_videos=args.remove_videos,
        local_dir=Path(args.local_dir) if args.local_dir else None,
        output_local_dir=Path(args.output_local_dir) if args.output_local_dir else None,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
