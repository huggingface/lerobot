#!/usr/bin/env python3
"""
Performance benchmark for PI0 tensor optimization.

This script measures the performance improvement from replacing list+torch.cat
with pre-allocated tensor operations in the PI0 policy.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch


def old_approach_embed_prefix(
    images: dict[str, torch.Tensor],
    img_masks: dict[str, torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    num_image_tokens: int = 4,
    hidden_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the old approach using list + torch.cat."""
    bsize = next(iter(images.values())).shape[0]
    device = next(iter(images.values())).device
    dtype = torch.float32

    # Old approach: list + torch.cat
    embs_list = []
    pad_masks_list = []
    att_masks_list = []

    # Process images
    for key in images.keys():
        # Simulate image embedding
        img_emb = torch.randn(bsize, num_image_tokens, hidden_size, dtype=dtype, device=device)
        embs_list.append(img_emb)
        pad_masks_list.append(torch.ones(bsize, num_image_tokens, dtype=torch.bool, device=device))
        att_masks_list.append(torch.zeros(bsize, num_image_tokens, dtype=torch.int32, device=device))

    # Process language
    lang_seq_len = lang_tokens.shape[1]
    lang_emb = torch.randn(bsize, lang_seq_len, hidden_size, dtype=dtype, device=device)
    embs_list.append(lang_emb)
    pad_masks_list.append(lang_masks)
    att_masks_list.append(torch.ones(bsize, lang_seq_len, dtype=torch.int32, device=device))

    # Concatenate all tensors
    embs = torch.cat(embs_list, dim=1)
    pad_masks = torch.cat(pad_masks_list, dim=1)
    att_masks = torch.cat(att_masks_list, dim=1)

    # Simulate past_key_values
    past_key_values = tuple(torch.randn(bsize, 2, hidden_size, device=device) for _ in range(4))

    return embs, pad_masks, past_key_values


def new_approach_embed_prefix(
    images: dict[str, torch.Tensor],
    img_masks: dict[str, torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    num_image_tokens: int = 4,
    hidden_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the new approach using pre-allocated tensors."""
    bsize = next(iter(images.values())).shape[0]
    device = next(iter(images.values())).device
    dtype = torch.float32

    # Calculate total sequence length for pre-allocation
    num_active_images = len(images)
    lang_seq_len = lang_tokens.shape[1]
    total_seq_len = num_active_images * num_image_tokens + lang_seq_len

    # Pre-allocate tensors
    embs = torch.empty((bsize, total_seq_len, hidden_size), dtype=dtype, device=device)
    pad_masks = torch.empty((bsize, total_seq_len), dtype=torch.bool, device=device)
    att_masks = torch.empty((bsize, total_seq_len), dtype=torch.int32, device=device)

    # Fill pre-allocated tensors
    start_idx = 0
    for key in images.keys():
        end_idx = start_idx + num_image_tokens
        # Simulate image embedding
        img_emb = torch.randn(bsize, num_image_tokens, hidden_size, dtype=dtype, device=device)
        embs[:, start_idx:end_idx] = img_emb
        pad_masks[:, start_idx:end_idx] = True
        att_masks[:, start_idx:end_idx] = 0
        start_idx = end_idx

    # Fill language embeddings
    end_idx = start_idx + lang_seq_len
    lang_emb = torch.randn(bsize, lang_seq_len, hidden_size, dtype=dtype, device=device)
    embs[:, start_idx:end_idx] = lang_emb
    pad_masks[:, start_idx:end_idx] = lang_masks
    att_masks[:, start_idx:end_idx] = 1

    # Simulate past_key_values
    past_key_values = tuple(torch.randn(bsize, 2, hidden_size, device=device) for _ in range(4))

    return embs, pad_masks, past_key_values


def old_approach_embed_suffix(
    state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor, proj_width: int = 1024
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the old approach for embed_suffix."""
    bsize, horizon, _ = noisy_actions.shape
    device = state.device
    dtype = state.dtype

    # Old approach: list + torch.cat
    embs_list = []
    pad_masks_list = []
    att_masks_list = []

    # State embedding
    state_emb = torch.randn(bsize, 1, proj_width, dtype=dtype, device=device)
    embs_list.append(state_emb)
    pad_masks_list.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
    att_masks_list.append(torch.ones(bsize, 1, dtype=torch.int32, device=device))

    # Time embedding
    time_emb = torch.randn(bsize, 1, proj_width, dtype=dtype, device=device)
    embs_list.append(time_emb)
    pad_masks_list.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
    att_masks_list.append(torch.ones(bsize, 1, dtype=torch.int32, device=device))

    # Action embeddings
    action_embs = torch.randn(bsize, horizon, proj_width, dtype=dtype, device=device)
    embs_list.append(action_embs)
    pad_masks_list.append(torch.ones(bsize, horizon, dtype=torch.bool, device=device))
    att_masks_list.append(torch.ones(bsize, horizon, dtype=torch.int32, device=device))

    # Concatenate
    embs = torch.cat(embs_list, dim=1)
    pad_masks = torch.cat(pad_masks_list, dim=1)
    att_masks = torch.cat(att_masks_list, dim=1)

    return embs, pad_masks, att_masks


def new_approach_embed_suffix(
    state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor, proj_width: int = 1024
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the new approach for embed_suffix."""
    bsize, horizon, _ = noisy_actions.shape
    device = state.device
    dtype = state.dtype

    # Calculate total sequence length for pre-allocation
    total_seq_len = 1 + 1 + horizon  # state + time + actions

    # Pre-allocate tensors
    embs = torch.empty((bsize, total_seq_len, proj_width), dtype=dtype, device=device)
    pad_masks = torch.ones((bsize, total_seq_len), dtype=torch.bool, device=device)
    att_masks = torch.ones((bsize, total_seq_len), dtype=torch.int32, device=device)

    # Fill pre-allocated tensors
    # State embedding
    state_emb = torch.randn(bsize, 1, proj_width, dtype=dtype, device=device)
    embs[:, 0:1] = state_emb

    # Time embedding
    time_emb = torch.randn(bsize, 1, proj_width, dtype=dtype, device=device)
    embs[:, 1:2] = time_emb

    # Action embeddings
    action_embs = torch.randn(bsize, horizon, proj_width, dtype=dtype, device=device)
    embs[:, 2:] = action_embs

    return embs, pad_masks, att_masks


def benchmark_function(func, *args, num_runs: int = 100, warmup_runs: int = 10) -> dict[str, float]:
    """Benchmark a function and return timing statistics."""
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args)

    # Actual benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


def create_test_data(
    bsize: int,
    num_images: int,
    lang_seq_len: int,
    state_dim: int,
    action_dim: int,
    horizon: int,
    device: str = "cpu",
):
    """Create test data for benchmarking."""
    images = {f"image_{i}": torch.randn(bsize, 3, 224, 224, device=device) for i in range(num_images)}
    img_masks = {
        f"image_{i}": torch.ones(bsize, 1, 1, dtype=torch.bool, device=device) for i in range(num_images)
    }
    lang_tokens = torch.randint(0, 1000, (bsize, lang_seq_len), device=device)
    lang_masks = torch.ones(bsize, lang_seq_len, dtype=torch.bool, device=device)
    state = torch.randn(bsize, state_dim, device=device)
    noisy_actions = torch.randn(bsize, horizon, action_dim, device=device)
    timestep = torch.rand(bsize, device=device)

    return images, img_masks, lang_tokens, lang_masks, state, noisy_actions, timestep


def main():
    """Run the performance benchmark."""
    print("🚀 PI0 Performance Benchmark")
    print("=" * 50)

    # Test configurations
    configs = [
        {"bsize": 1, "num_images": 1, "lang_seq_len": 10, "state_dim": 8, "action_dim": 7, "horizon": 50},
        {"bsize": 2, "num_images": 2, "lang_seq_len": 20, "state_dim": 16, "action_dim": 14, "horizon": 100},
        {"bsize": 4, "num_images": 3, "lang_seq_len": 30, "state_dim": 32, "action_dim": 28, "horizon": 150},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()

    for i, config in enumerate(configs, 1):
        print(f"Configuration {i}: {config}")
        print("-" * 40)

        # Create test data
        images, img_masks, lang_tokens, lang_masks, state, noisy_actions, timestep = create_test_data(
            device=device, **config
        )

        # Benchmark embed_prefix
        print("📊 Benchmarking embed_prefix...")
        old_prefix_stats = benchmark_function(
            old_approach_embed_prefix, images, img_masks, lang_tokens, lang_masks
        )
        new_prefix_stats = benchmark_function(
            new_approach_embed_prefix, images, img_masks, lang_tokens, lang_masks
        )

        prefix_improvement = (
            (old_prefix_stats["mean"] - new_prefix_stats["mean"]) / old_prefix_stats["mean"]
        ) * 100

        print(f"  Old approach: {old_prefix_stats['mean']:.3f} ± {old_prefix_stats['std']:.3f} ms")
        print(f"  New approach: {new_prefix_stats['mean']:.3f} ± {new_prefix_stats['std']:.3f} ms")
        print(f"  Improvement: {prefix_improvement:.1f}%")

        # Benchmark embed_suffix
        print("📊 Benchmarking embed_suffix...")
        old_suffix_stats = benchmark_function(old_approach_embed_suffix, state, noisy_actions, timestep)
        new_suffix_stats = benchmark_function(new_approach_embed_suffix, state, noisy_actions, timestep)

        suffix_improvement = (
            (old_suffix_stats["mean"] - new_suffix_stats["mean"]) / old_suffix_stats["mean"]
        ) * 100

        print(f"  Old approach: {old_suffix_stats['mean']:.3f} ± {old_suffix_stats['std']:.3f} ms")
        print(f"  New approach: {new_suffix_stats['mean']:.3f} ± {new_suffix_stats['std']:.3f} ms")
        print(f"  Improvement: {suffix_improvement:.1f}%")

        # Combined improvement
        total_old = old_prefix_stats["mean"] + old_suffix_stats["mean"]
        total_new = new_prefix_stats["mean"] + new_suffix_stats["mean"]
        total_improvement = ((total_old - total_new) / total_old) * 100

        print(f"📈 Total improvement: {total_improvement:.1f}%")
        print()

    print("✅ Benchmark completed!")


if __name__ == "__main__":
    main()
