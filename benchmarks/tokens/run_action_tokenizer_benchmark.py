#!/usr/bin/env python
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
"""Benchmark action tokenization: reconstruction error, compression ratio, and timing.

Loads action chunks from a LeRobot dataset, encodes/decodes them with a trained action
tokenizer, and reports:
- Reconstruction: MAE, MSE, RMSE, max absolute error, per-dimension MAE
- Jerk: mean absolute jerk (original and reconstructed), jerk reconstruction MAE
- Compression: ratio (input size / mean tokens), token length stats
- Timing: mean encode/decode time per chunk

Results are saved to outputs/action_tokenizer_benchmark/<timestamp>_results.json.

Example:

```bash
python benchmarks/tokens/run_action_tokenizer_benchmark.py \
    --action-tokenizer-path=outputs/wavetoken \
    --repo-id=lerobot/pusht \
    --action-horizon=10 \
    --max-episodes=50 \
    --output-dir=outputs/action_tokenizer_benchmark
```
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

# Optional: use same helpers as train script if we want to avoid duplication
from lerobot.scripts.lerobot_train_tokenizer import (
    apply_normalization,
    process_episode,
)


def load_action_chunks(
    repo_id: str,
    root: str | None,
    action_horizon: int,
    max_episodes: int | None,
    sample_fraction: float,
    encoded_dims: str,
    delta_dims: str | None,
    use_delta_transform: bool,
    state_key: str,
    normalization_mode: NormalizationMode,
):
    """Load and normalize action chunks from a LeRobot dataset (same pipeline as training)."""
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(max_episodes, num_episodes)

    # Parse encoded dims
    encoded_dim_ranges = []
    for range_str in encoded_dims.split(","):
        start, end = map(int, range_str.strip().split(":"))
        encoded_dim_ranges.append((start, end))
    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)

    delta_dim_list = None
    if delta_dims is not None and delta_dims.strip():
        delta_dim_list = [int(d.strip()) for d in delta_dims.split(",")]

    all_chunks = []
    for ep_idx in range(num_episodes):
        chunks = process_episode(
            (
                dataset,
                ep_idx,
                action_horizon,
                delta_dim_list,
                sample_fraction,
                state_key,
                use_delta_transform,
            )
        )
        if chunks is not None:
            all_chunks.append(chunks)

    if not all_chunks:
        raise ValueError("No action chunks collected. Check action_horizon and dataset.")

    all_chunks = np.concatenate(all_chunks, axis=0)

    # Extract encoded dimensions only
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(all_chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1)

    # Normalize
    norm_stats = dataset.meta.stats
    if norm_stats is not None and ACTION in norm_stats:
        action_stats = norm_stats[ACTION]
        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)
        encoded_stats = {}
        for stat_name, stat_values in action_stats.items():
            if isinstance(stat_values, (list, np.ndarray)):
                stat_array = np.array(stat_values)
                if len(stat_array) > max(encoded_dim_indices):
                    encoded_stats[stat_name] = stat_array[encoded_dim_indices]
        if encoded_stats:
            try:
                encoded_chunks = apply_normalization(
                    encoded_chunks, encoded_stats, normalization_mode, eps=1e-8
                )
            except ValueError:
                pass

    return encoded_chunks, total_encoded_dims, action_horizon, dataset.repo_id


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray):
    """Compute reconstruction error metrics (original and reconstructed same shape [N, T, D])."""
    diff = reconstructed - original
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    max_abs_err = float(np.max(np.abs(diff)))

    # Per-dimension MAE (over N and T)
    per_dim_mae = np.mean(np.abs(diff), axis=(0, 1))
    per_dim_mae = per_dim_mae.tolist()

    return {
        "reconstruction_mae": mae,
        "reconstruction_mse": mse,
        "reconstruction_rmse": rmse,
        "reconstruction_max_abs_error": max_abs_err,
        "per_dimension_mae": per_dim_mae,
    }


def compute_jerk_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute jerk (3rd derivative of action w.r.t. time) metrics.

    Args:
        original: Action chunks [N, T, D].
        reconstructed: Reconstructed action chunks [N, T, D].

    Returns:
        Dict with mean absolute jerk for original, reconstructed, and jerk reconstruction MAE.
    """
    # Jerk = 3rd discrete difference along time axis; need T >= 4
    if original.shape[1] < 4:
        return {}
    jerk_orig = np.diff(original, n=3, axis=1)  # (N, T-3, D)
    jerk_recon = np.diff(reconstructed, n=3, axis=1)
    mae_jerk_orig = float(np.mean(np.abs(jerk_orig)))
    mae_jerk_recon = float(np.mean(np.abs(jerk_recon)))
    jerk_reconstruction_mae = float(np.mean(np.abs(jerk_recon - jerk_orig)))
    return {
        "jerk_mae_original": mae_jerk_orig,
        "jerk_mae_reconstructed": mae_jerk_recon,
        "jerk_reconstruction_mae": jerk_reconstruction_mae,
    }


def run_benchmark(
    action_chunks: np.ndarray,
    action_horizon: int,
    action_dim: int,
    tokenizer_path: str,
    max_chunks_for_reconstruction: int | None = 500,
):
    """Encode/decode action chunks and compute metrics."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

    n_chunks = len(action_chunks)
    sample_size = n_chunks
    if max_chunks_for_reconstruction is not None:
        sample_size = min(max_chunks_for_reconstruction, n_chunks)
    rng = np.random.RandomState(42)
    indices = rng.choice(n_chunks, size=sample_size, replace=False)
    sample_chunks = action_chunks[indices]

    # Encode
    token_lengths = []
    encode_times = []
    all_tokens = []
    for i in range(len(sample_chunks)):
        chunk = sample_chunks[i : i + 1]
        t0 = time.perf_counter()
        tokens = processor(chunk)[0]
        encode_times.append(time.perf_counter() - t0)
        if isinstance(tokens, list):
            token_lengths.append(len(tokens))
            all_tokens.append(tokens)
        else:
            n = tokens.shape[0] if hasattr(tokens, "shape") else len(tokens)
            token_lengths.append(n)
            all_tokens.append(tokens.tolist() if hasattr(tokens, "tolist") else list(tokens))

    # Decode (processor keeps time_horizon/action_dim from encode)
    decoded_list = []
    decode_times = []
    for i, tok_list in enumerate(all_tokens):
        t0 = time.perf_counter()
        recon = processor.decode(
            [tok_list],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )
        decode_times.append(time.perf_counter() - t0)
        decoded_list.append(recon)
    decoded = np.concatenate(decoded_list, axis=0)

    # Reconstruction metrics
    metrics = compute_reconstruction_metrics(sample_chunks, decoded)

    # Jerk metrics (3rd derivative along time)
    jerk_metrics = compute_jerk_metrics(sample_chunks, decoded)
    metrics.update(jerk_metrics)

    # Compression
    token_lengths = np.array(token_lengths)
    input_size = action_horizon * action_dim
    compression_ratio = input_size / float(np.mean(token_lengths))
    metrics["compression_ratio"] = compression_ratio
    metrics["mean_token_length"] = float(np.mean(token_lengths))
    metrics["std_token_length"] = float(np.std(token_lengths))
    metrics["min_token_length"] = int(np.min(token_lengths))
    metrics["max_token_length"] = int(np.max(token_lengths))
    metrics["p50_token_length"] = float(np.percentile(token_lengths, 50))
    metrics["p99_token_length"] = float(np.percentile(token_lengths, 99))

    # Timing (seconds per chunk)
    metrics["mean_encode_time_sec"] = float(np.mean(encode_times))
    metrics["mean_decode_time_sec"] = float(np.mean(decode_times))
    metrics["num_chunks_evaluated"] = sample_size
    metrics["total_chunks_available"] = n_chunks

    return metrics


def main(
    action_tokenizer_path: str,
    repo_id: str,
    root: str | None = None,
    action_horizon: int = 10,
    max_episodes: int | None = 100,
    sample_fraction: float = 0.2,
    encoded_dims: str = "0:6",
    delta_dims: str | None = None,
    use_delta_transform: bool = False,
    state_key: str = OBS_STATE,
    normalization_mode: str = "QUANTILES",
    max_chunks_for_reconstruction: int | None = 500,
    output_dir: str | None = None,
):
    if output_dir is None:
        output_dir = "outputs/action_tokenizer_benchmark"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        norm_mode = NormalizationMode(normalization_mode)
    except ValueError:
        norm_mode = NormalizationMode.QUANTILES

    print("Loading action chunks...")
    encoded_chunks, action_dim, horizon, _ = load_action_chunks(
        repo_id=repo_id,
        root=root,
        action_horizon=action_horizon,
        max_episodes=max_episodes,
        sample_fraction=sample_fraction,
        encoded_dims=encoded_dims,
        delta_dims=delta_dims,
        use_delta_transform=use_delta_transform,
        state_key=state_key,
        normalization_mode=norm_mode,
    )
    print(f"Loaded {len(encoded_chunks)} chunks, shape {encoded_chunks.shape} (H={horizon}, D={action_dim})")

    print("Running tokenizer benchmark...")
    metrics = run_benchmark(
        action_chunks=encoded_chunks,
        action_horizon=horizon,
        action_dim=action_dim,
        tokenizer_path=action_tokenizer_path,
        max_chunks_for_reconstruction=max_chunks_for_reconstruction,
    )

    # Attach config for reproducibility
    results = {
        "config": {
            "action_tokenizer_path": action_tokenizer_path,
            "repo_id": repo_id,
            "action_horizon": action_horizon,
            "max_episodes": max_episodes,
            "sample_fraction": sample_fraction,
            "encoded_dims": encoded_dims,
            "delta_dims": delta_dims,
            "use_delta_transform": use_delta_transform,
            "state_key": state_key,
            "normalization_mode": normalization_mode,
        },
        "metrics": metrics,
    }

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_repo = repo_id.replace("/", "_")
    out_file = output_path / f"{timestamp}_{safe_repo}_action_tokenizer_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_file}")
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"  {k}: (length {len(v)})")
        else:
            print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark action tokenization (reconstruction error, compression, timing)."
    )
    parser.add_argument(
        "--action-tokenizer-path",
        type=str,
        required=True,
        help="Path or HuggingFace repo id of the trained action tokenizer (e.g. outputs/wavetoken).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repo id (e.g. lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for LeRobot datasets.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Number of future steps per action chunk.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Max episodes to use (default: all).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Fraction of chunks to sample per episode.",
    )
    parser.add_argument(
        "--encoded-dims",
        type=str,
        default="0:6",
        help="Dimension ranges to encode (e.g. 0:6,7:14).",
    )
    parser.add_argument(
        "--delta-dims",
        type=str,
        default=None,
        help="Comma-separated dimensions for delta transform.",
    )
    parser.add_argument(
        "--use-delta-transform",
        action="store_true",
        help="Apply delta (relative) transform to specified dimensions.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=OBS_STATE,
        help="Dataset key for state (for delta transform).",
    )
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="QUANTILES",
        choices=[m.value for m in NormalizationMode],
        help="Normalization mode for actions.",
    )
    parser.add_argument(
        "--max-chunks-for-reconstruction",
        type=int,
        default=500,
        help="Max chunks to use for reconstruction metrics (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/action_tokenizer_benchmark",
        help="Directory to save results JSON (default: outputs/action_tokenizer_benchmark).",
    )
    args = parser.parse_args()
    main(
        action_tokenizer_path=args.action_tokenizer_path,
        repo_id=args.repo_id,
        root=args.root,
        action_horizon=args.action_horizon,
        max_episodes=args.max_episodes,
        sample_fraction=args.sample_fraction,
        encoded_dims=args.encoded_dims,
        delta_dims=args.delta_dims,
        use_delta_transform=args.use_delta_transform,
        state_key=args.state_key,
        normalization_mode=args.normalization_mode,
        max_chunks_for_reconstruction=args.max_chunks_for_reconstruction,
        output_dir=args.output_dir,
    )
