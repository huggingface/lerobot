# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Train FAST tokenizer for action encoding.

This script:
1. Loads action chunks from LeRobotDataset (with episode sampling)
2. Optionally applies relative transforms (relative vs absolute actions)
3. Extracts specified action dimensions for encoding
4. Applies normalization (MEAN_STD, MIN_MAX, QUANTILES, or other modes)
5. Trains FAST tokenizer (BPE on DCT coefficients) on the action chunks
6. Saves tokenizer to output directory
7. Optionally pushes tokenizer to Hugging Face Hub
8. Reports compression statistics

Example:

```shell
lerobot-train-tokenizer \
    --repo_id=user/dataset_name \
    --action_horizon=10 \
    --max_episodes=100 \
    --sample_fraction=0.1 \
    --encoded_dims="0:6" \
    --relative_dims="0,1,2,3,4,5" \
    --use_relative_transform=true \
    --state_key="observation.state" \
    --normalization_mode="QUANTILES" \
    --vocab_size=1024 \
    --scale=10.0 \
    --output_dir="./fast_tokenizer_dataset_name" \
    --push_to_hub=true \
    --hub_repo_id="user/fast_tokenizer_dataset_name" \
    --hub_private=false
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from huggingface_hub import HfApi

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, parser
from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import NormalizerProcessorStep, encode_fast_actions, to_relative_actions
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# Revision of the FAST code that fitting runs; ``fit()`` only uses the class, not its vocabulary.
LEROBOT_FAST_REVISION = "3d4d8dd847c5333f1cb0f54d00074ba22b14a5b7"


@dataclass
class TokenizerTrainingConfig:
    """Configuration for training FAST tokenizer."""

    # LeRobot dataset repository ID
    repo_id: str
    # Root directory for dataset (default: ~/.cache/huggingface/lerobot)
    root: str | None = None
    # Number of future actions in each chunk
    action_horizon: int = 10
    # Max episodes to use (None = all episodes in dataset)
    max_episodes: int | None = None
    # Fraction of chunks to sample per episode
    sample_fraction: float = 0.1
    # Comma-separated dimension ranges to encode (e.g., "0:6,7:23")
    encoded_dims: str = "0:6,7:23"
    # Comma-separated dimension indices for relative transform (e.g., "0,1,2,3,4,5")
    relative_dims: str | None = None
    # Whether to apply relative transform (relative actions vs absolute actions)
    use_relative_transform: bool = False
    # Dataset key for state observations (default: "observation.state")
    state_key: str = OBS_STATE
    # Normalization mode (MEAN_STD, MIN_MAX, QUANTILES, QUANTILE10, IDENTITY)
    normalization_mode: str = "QUANTILES"
    # FAST vocabulary size (BPE vocab size)
    vocab_size: int = 1024
    # DCT scaling factor (default: 10.0)
    scale: float = 10.0
    # Directory to save tokenizer (default: ./fast_tokenizer_{repo_id})
    output_dir: str | None = None
    # Whether to push the tokenizer to Hugging Face Hub
    push_to_hub: bool = False
    # Hub repository ID (e.g., "username/tokenizer-name"). If None, uses output_dir name
    hub_repo_id: str | None = None
    # Whether to create a private repository on the Hub
    hub_private: bool = False


def apply_normalization(
    data: np.ndarray,
    stats: dict[str, np.ndarray],
    mode: NormalizationMode,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize fitting inputs with the same processor used by policy training.

    Args:
        data: Data to normalize [N, H, D] or [D]
        stats: Dictionary of statistics (mean, std, min, max, q01, q99, q10, q90)
        mode: Normalization mode to apply
        eps: Small epsilon for numerical stability

    Returns:
        Normalized data with the same shape as input
    """
    if mode == NormalizationMode.IDENTITY:
        return data

    required_stats = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }
    if mode not in required_stats:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    first_stat, second_stat = required_stats[mode]
    if stats.get(first_stat) is None or stats.get(second_stat) is None:
        raise ValueError(f"{mode.value} mode requires '{first_stat}' and '{second_stat}' in stats")

    tensor = torch.as_tensor(data)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    norm_map: dict[FeatureType, NormalizationMode] = {FeatureType.ACTION: mode}
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(data.shape[-1],))},
        norm_map=norm_map,
        stats={ACTION: stats},
        dtype=tensor.dtype,
        eps=eps,
    )
    normalized = normalizer(create_transition(action=tensor))[TransitionKey.ACTION]
    return cast(torch.Tensor, normalized).numpy()


def process_episode(args):
    """Process single episode and return action chunks."""
    dataset, ep_idx, action_horizon, relative_dims, sample_fraction, state_key, use_relative_transform = args

    try:
        # get episode info
        ep_info = dataset.meta.episodes[ep_idx]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]
        ep_length = to_idx - from_idx

        if ep_length < action_horizon:
            return None

        # load all frames in episode
        # if dataset has episode filtering, we need to use the mapping
        states = []
        actions = []

        absolute_to_relative_idx = dataset.absolute_to_relative_idx

        for abs_idx in range(from_idx, to_idx):
            # map absolute index to relative index if needed
            if absolute_to_relative_idx is not None:
                if abs_idx not in absolute_to_relative_idx:
                    # this episode's frames aren't in the filtered dataset
                    return None
                rel_idx = absolute_to_relative_idx[abs_idx]
            else:
                rel_idx = abs_idx

            frame = dataset.get_raw_item(rel_idx)

            # get state (could be from observation.state or other state key)
            if state_key in frame:
                state = (
                    frame[state_key].numpy()
                    if torch.is_tensor(frame[state_key])
                    else np.array(frame[state_key])
                )
            else:
                # if no state key, use zeros (no relative transform)
                state = np.zeros_like(
                    frame[ACTION].numpy() if torch.is_tensor(frame[ACTION]) else np.array(frame[ACTION])
                )

            action = frame[ACTION].numpy() if torch.is_tensor(frame[ACTION]) else np.array(frame[ACTION])

            states.append(state)
            actions.append(action)

        states = np.array(states)
        actions = np.array(actions)

        # create action chunks (sliding window)
        starts = range(len(states) - action_horizon + 1)
        action_chunks = np.stack([actions[i : i + action_horizon] for i in starts])
        if use_relative_transform and relative_dims:
            # Same transform as training's RelativeActionsProcessorStep: every action in a chunk
            # becomes an offset from the chunk's first state, for the listed dimensions only.
            mask = [dim in relative_dims for dim in range(max(relative_dims) + 1)]
            action_chunks = to_relative_actions(
                torch.from_numpy(action_chunks), torch.from_numpy(states[list(starts)]), mask
            ).numpy()

        # sample chunks
        if sample_fraction < 1.0:
            n_chunks = len(action_chunks)
            n_samples = max(1, int(n_chunks * sample_fraction))
            episode_seed = hash(ep_idx) % (2**31)
            rng = np.random.RandomState(episode_seed)
            indices = rng.choice(n_chunks, size=n_samples, replace=False)
            action_chunks = action_chunks[indices]

        return action_chunks

    except Exception:
        logger.exception("Error processing episode %s", ep_idx)
        return None


def train_fast_tokenizer(
    action_chunks: np.ndarray,
    vocab_size: int = 1024,
    scale: float = 10.0,
) -> AutoProcessor:
    """
    Train FAST tokenizer (BPE on DCT coefficients) on action chunks.

    Uses the .fit() method to train a new tokenizer on the provided data.

    Args:
        action_chunks: Array of action chunks [N, H, D] where N=num_chunks, H=horizon, D=action_dim
        vocab_size: BPE vocabulary size
        scale: DCT scaling factor for quantization

    Returns:
        Trained FAST tokenizer
    """
    logger.info(f"Training FAST tokenizer on {len(action_chunks)} action chunks...")
    logger.info(f"Action chunk shape: {action_chunks.shape}")
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"DCT scale: {scale}")

    # download the tokenizer source code (not pretrained weights)
    # we'll train a new tokenizer on our own data
    base_tokenizer = AutoProcessor.from_pretrained(
        "lerobot/fast-action-tokenizer", revision=LEROBOT_FAST_REVISION, trust_remote_code=True
    )

    # convert action_chunks array to list of arrays (expected by .fit())
    action_data_list = [action_chunks[i] for i in range(len(action_chunks))]

    # train the new tokenizer on our action data using .fit()
    # this trains the BPE tokenizer on DCT coefficients
    logger.info("Training new tokenizer (this may take a few minutes)...")
    tokenizer = base_tokenizer.fit(
        action_data_list,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=action_chunks.shape[1],  # action_horizon
        action_dim=action_chunks.shape[2],  # encoded dimensions
    )
    logger.info("✓ Tokenizer training complete!")

    # validate it works
    (encoded,) = encode_fast_actions(tokenizer, action_chunks[:1])
    logger.info(f"Sample encoding: {len(encoded)} tokens for chunk shape {action_chunks.shape[1:]}")

    return tokenizer


def compute_compression_stats(tokenizer, action_chunks: np.ndarray):
    """Compute compression statistics."""
    logger.info("\nComputing compression statistics...")

    # sample for stats (use max 1000 chunks for speed)
    sample_size = min(1000, len(action_chunks))
    sample_indices = np.random.RandomState(42).choice(len(action_chunks), size=sample_size, replace=False)
    sample_chunks = action_chunks[sample_indices]

    token_lengths = np.array([len(tokens) for tokens in encode_fast_actions(tokenizer, sample_chunks)])

    # compression ratio: (H * D) / avg_tokens
    input_size = action_chunks.shape[1] * action_chunks.shape[2]
    avg_tokens = np.mean(token_lengths)
    compression_ratio = input_size / avg_tokens

    stats = {
        "compression_ratio": float(compression_ratio),
        "mean_token_length": float(np.mean(token_lengths)),
        "p99_token_length": float(np.percentile(token_lengths, 99)),
        "min_token_length": float(np.min(token_lengths)),
        "max_token_length": float(np.max(token_lengths)),
    }

    logger.info("Compression Statistics:")
    logger.info(f"  Average compression ratio: {stats['compression_ratio']:.2f}x")
    logger.info(f"  Mean token length: {stats['mean_token_length']:.1f}")
    logger.info(f"  P99 token length: {stats['p99_token_length']:.0f}")
    logger.info(f"  Min token length: {stats['min_token_length']:.0f}")
    logger.info(f"  Max token length: {stats['max_token_length']:.0f}")

    return stats


@parser.wrap()
def train_tokenizer(cfg: TokenizerTrainingConfig) -> None:
    """
    Train FAST tokenizer for action encoding.

    Args:
        cfg: TokenizerTrainingConfig dataclass with all configuration parameters
    """
    # load dataset
    logger.info(f"Loading dataset: {cfg.repo_id}")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)
    logger.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # parse normalization mode
    try:
        norm_mode = NormalizationMode(cfg.normalization_mode)
    except ValueError as err:
        raise ValueError(
            f"Invalid normalization_mode: {cfg.normalization_mode}. "
            f"Must be one of: {', '.join([m.value for m in NormalizationMode])}"
        ) from err
    logger.info(f"Normalization mode: {norm_mode.value}")

    # parse encoded dimensions
    encoded_dim_ranges = []
    for range_str in cfg.encoded_dims.split(","):
        start, end = map(int, range_str.strip().split(":"))
        encoded_dim_ranges.append((start, end))

    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    logger.info(f"Encoding {total_encoded_dims} dimensions: {cfg.encoded_dims}")

    # parse relative dimensions
    relative_dim_list = None
    if cfg.relative_dims is not None and cfg.relative_dims.strip():
        relative_dim_list = [int(d.strip()) for d in cfg.relative_dims.split(",")]
        logger.info(f"Relative dimensions: {relative_dim_list}")
    else:
        logger.info("No relative dimensions specified")

    logger.info(f"Use relative transform: {cfg.use_relative_transform}")
    if cfg.use_relative_transform and (relative_dim_list is None or len(relative_dim_list) == 0):
        logger.warning(
            "Warning: use_relative_transform=True but no relative_dims specified. "
            "No relative transform will be applied."
        )

    logger.info(f"Action horizon: {cfg.action_horizon}")
    logger.info(f"State key: {cfg.state_key}")

    # determine episodes to process
    num_episodes = dataset.num_episodes
    if cfg.max_episodes is not None:
        num_episodes = min(cfg.max_episodes, num_episodes)

    logger.info(f"Processing {num_episodes} episodes...")

    # process episodes sequentially (to avoid pickling issues with dataset)
    all_chunks = []
    for ep_idx in range(num_episodes):
        if ep_idx % 10 == 0:
            logger.info(f"  Processing episode {ep_idx}/{num_episodes}...")

        chunks = process_episode(
            (
                dataset,
                ep_idx,
                cfg.action_horizon,
                relative_dim_list,
                cfg.sample_fraction,
                cfg.state_key,
                cfg.use_relative_transform,
            )
        )
        if chunks is not None:
            all_chunks.append(chunks)

    # concatenate all chunks
    all_chunks = np.concatenate(all_chunks, axis=0)
    logger.info(f"Collected {len(all_chunks)} action chunks")

    # extract only encoded dimensions FIRST (before normalization)
    encoded_chunks = np.concatenate(
        [all_chunks[:, :, start:end] for start, end in encoded_dim_ranges], axis=-1
    )  # [N, H, D_encoded]
    logger.info(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")

    # apply normalization to encoded dimensions
    logger.info("\nBefore normalization - overall stats:")
    logger.info(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    logger.info(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

    # get normalization stats from dataset
    norm_stats = dataset.meta.stats
    if norm_stats is not None and ACTION in norm_stats:
        action_stats = norm_stats[ACTION]

        # build encoded dimension indices
        encoded_dim_indices = np.array([i for start, end in encoded_dim_ranges for i in range(start, end)])

        # extract stats for encoded dimensions only
        encoded_stats = {}
        for stat_name, stat_values in action_stats.items():
            if isinstance(stat_values, (list, np.ndarray)):
                stat_array = np.array(stat_values)
                if len(stat_array) > max(encoded_dim_indices):
                    encoded_stats[stat_name] = stat_array[encoded_dim_indices]

        if encoded_stats:
            logger.info(f"\nNormalization stats for encoded dimensions (mode: {norm_mode.value}):")
            for stat_name, stat_values in encoded_stats.items():
                logger.info(
                    f"  {stat_name}: shape={stat_values.shape}, "
                    f"range=[{np.min(stat_values):.4f}, {np.max(stat_values):.4f}]"
                )

            # apply normalization based on mode
            try:
                encoded_chunks = apply_normalization(encoded_chunks, encoded_stats, norm_mode, eps=1e-8)
                logger.info(f"\nApplied {norm_mode.value} normalization")
            except ValueError as e:
                logger.warning(f"Warning: {e}. Using raw actions without normalization.")

            logger.info("\nAfter normalization - overall stats:")
            logger.info(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
            logger.info(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

            logger.info("\nPer-dimension stats (after normalization):")
            for d in range(encoded_chunks.shape[-1]):
                dim_data = encoded_chunks[:, :, d]
                logger.info(
                    f"  Dim {d}: min={np.min(dim_data):7.4f}, max={np.max(dim_data):7.4f}, "
                    f"mean={np.mean(dim_data):7.4f}, std={np.std(dim_data):7.4f}"
                )
        else:
            logger.warning("Warning: Could not extract stats for encoded dimensions, using raw actions")
    else:
        logger.warning("Warning: No normalization stats found in dataset, using raw actions")

    logger.info(f"Encoded chunks shape: {encoded_chunks.shape}")

    # train FAST tokenizer
    tokenizer = train_fast_tokenizer(
        encoded_chunks,
        vocab_size=cfg.vocab_size,
        scale=cfg.scale,
    )

    # compute compression statistics
    compression_stats = compute_compression_stats(tokenizer, encoded_chunks)

    # save tokenizer
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = f"fast_tokenizer_{cfg.repo_id.replace('/', '_')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_path)

    # save metadata
    metadata = {
        "repo_id": cfg.repo_id,
        "vocab_size": cfg.vocab_size,
        "scale": cfg.scale,
        "encoded_dims": cfg.encoded_dims,
        "encoded_dim_ranges": encoded_dim_ranges,
        "total_encoded_dims": total_encoded_dims,
        "relative_dims": cfg.relative_dims,
        "relative_dim_list": relative_dim_list,
        "use_relative_transform": cfg.use_relative_transform,
        "state_key": cfg.state_key,
        "normalization_mode": norm_mode.value,
        "action_horizon": cfg.action_horizon,
        "num_training_chunks": len(encoded_chunks),
        "compression_stats": compression_stats,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved FAST tokenizer to {output_path}")
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")

    # push to Hugging Face Hub if requested
    if cfg.push_to_hub:
        # determine the hub repository ID
        hub_repo_id = cfg.hub_repo_id
        if hub_repo_id is None:
            hub_repo_id = output_path.name
            logger.info(f"\nNo hub_repo_id provided, using: {hub_repo_id}")

        logger.info(f"\nPushing tokenizer to Hugging Face Hub: {hub_repo_id}")
        logger.info(f"   Private: {cfg.hub_private}")

        try:
            # use the tokenizer's push_to_hub method
            tokenizer.push_to_hub(
                repo_id=hub_repo_id,
                private=cfg.hub_private,
                commit_message=f"Upload FAST tokenizer trained on {cfg.repo_id}",
            )

            # also upload the metadata.json file separately
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_path / "metadata.json"),
                path_in_repo="metadata.json",
                repo_id=hub_repo_id,
                repo_type="model",
                commit_message="Upload tokenizer metadata",
            )

            logger.info(f"Successfully pushed tokenizer to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
            logger.error("   Make sure you're logged in with `hf auth login`")


def main():
    """CLI entry point that parses arguments and runs the tokenizer training."""
    init_logging()
    train_tokenizer()


if __name__ == "__main__":
    main()
