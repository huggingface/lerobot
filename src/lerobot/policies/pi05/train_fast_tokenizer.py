"""Train FAST tokenizer for action encoding.

This script:
1. Loads action chunks from LeRobotDataset (with sampling)
2. Applies delta transforms and per-timestamp normalization
3. Trains FAST tokenizer on specified action dimensions
4. Saves tokenizer to assets directory
5. Reports compression statistics
"""

import json
import numpy as np
import tyro
from pathlib import Path
from transformers import AutoProcessor
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def apply_delta_transform(state: np.ndarray, actions: np.ndarray, delta_dims: list[int] | None) -> np.ndarray:
    """Apply delta transform to specified dimensions.
    
    Args:
        state: Current state [D]
        actions: Future actions [D]
        delta_dims: List of dimension indices to apply delta transform to
    
    Returns:
        Transformed actions [D]
    """
    if delta_dims is None or len(delta_dims) == 0:
        return actions
    
    delta_actions = actions.copy()
    for dim in delta_dims:
        delta_actions[dim] = actions[dim] - state[dim]
    
    return delta_actions


def process_episode(args):
    """Process single episode and return action chunks."""
    dataset, ep_idx, action_horizon, delta_dims, sample_fraction, state_key, use_delta_transform = args
    
    try:
        # Get episode info
        ep_info = dataset.meta.episodes[ep_idx]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]
        ep_length = to_idx - from_idx
        
        if ep_length < action_horizon:
            return None
        
        # Load all frames in episode
        # If dataset has episode filtering, we need to use the mapping
        states = []
        actions = []
        
        for abs_idx in range(from_idx, to_idx):
            # Map absolute index to relative index if needed
            if dataset._absolute_to_relative_idx is not None:
                if abs_idx not in dataset._absolute_to_relative_idx:
                    # This episode's frames aren't in the filtered dataset
                    return None
                rel_idx = dataset._absolute_to_relative_idx[abs_idx]
            else:
                rel_idx = abs_idx
            
            frame = dataset.hf_dataset[rel_idx]
            
            # Get state (could be from observation.state or other state key)
            if state_key in frame:
                state = frame[state_key].numpy() if torch.is_tensor(frame[state_key]) else np.array(frame[state_key])
            else:
                # If no state key, use zeros (no delta transform)
                state = np.zeros_like(frame["action"].numpy() if torch.is_tensor(frame["action"]) else np.array(frame["action"]))
            
            action = frame["action"].numpy() if torch.is_tensor(frame["action"]) else np.array(frame["action"])
            
            states.append(state)
            actions.append(action)
        
        states = np.array(states)
        actions = np.array(actions)
        
        # Create action chunks (sliding window)
        # All actions in a chunk are relative to the FIRST state in that chunk
        action_chunks = []
        
        for i in range(len(states) - action_horizon + 1):
            current_state = states[i]  # First state in chunk
            future_absolute_actions = actions[i:i + action_horizon]
            
            if use_delta_transform:
                # Relative actions
                delta_chunk = np.zeros_like(future_absolute_actions)
                for t in range(action_horizon):
                    delta_chunk[t] = apply_delta_transform(
                        current_state,
                        future_absolute_actions[t],
                        delta_dims,
                    )
                action_chunks.append(delta_chunk)
            else:
                # Absolute actions (NO delta)
                action_chunks.append(future_absolute_actions)
        
        if len(action_chunks) == 0:
            return None
        
        action_chunks = np.array(action_chunks)
        
        # Sample chunks
        if sample_fraction < 1.0:
            n_chunks = len(action_chunks)
            n_samples = max(1, int(n_chunks * sample_fraction))
            episode_seed = hash(ep_idx) % (2**31)
            rng = np.random.RandomState(episode_seed)
            indices = rng.choice(n_chunks, size=n_samples, replace=False)
            action_chunks = action_chunks[indices]
        
        return action_chunks
        
    except Exception as e:
        print(f"Error processing episode {ep_idx}: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"Training FAST tokenizer on {len(action_chunks)} action chunks...")
    print(f"Action chunk shape: {action_chunks.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"DCT scale: {scale}")
    
    # Download the tokenizer source code (not pretrained weights)
    # We'll train a new tokenizer on our own data
    base_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast",
        trust_remote_code=True
    )
    
    # Convert action_chunks array to list of arrays (expected by .fit())
    action_data_list = [action_chunks[i] for i in range(len(action_chunks))]
    
    # Train the new tokenizer on our action data using .fit()
    # This trains the BPE tokenizer on DCT coefficients
    print("Training new tokenizer (this may take a few minutes)...")
    tokenizer = base_tokenizer.fit(
        action_data_list,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=action_chunks.shape[1],  # action_horizon
        action_dim=action_chunks.shape[2],     # encoded dimensions
    )
    print("✓ Tokenizer training complete!")
    
    # Validate it works
    sample_chunk = action_chunks[0]
    encoded = tokenizer(sample_chunk[None])[0]
    if isinstance(encoded, list):
        encoded = np.array(encoded)
    print(f"Sample encoding: {len(encoded)} tokens for chunk shape {sample_chunk.shape}")
    
    return tokenizer


def compute_compression_stats(tokenizer, action_chunks: np.ndarray):
    """Compute compression statistics."""
    print("\nComputing compression statistics...")
    
    # Sample for stats (use max 1000 chunks for speed)
    sample_size = min(1000, len(action_chunks))
    sample_indices = np.random.RandomState(42).choice(len(action_chunks), size=sample_size, replace=False)
    sample_chunks = action_chunks[sample_indices]
    
    token_lengths = []
    for chunk in sample_chunks:
        encoded = tokenizer(chunk[None])[0]
        if isinstance(encoded, list):
            token_lengths.append(len(encoded))
        else:
            token_lengths.append(encoded.shape[0] if hasattr(encoded, 'shape') else len(encoded))
    
    token_lengths = np.array(token_lengths)
    
    # Compression ratio: (H * D) / avg_tokens
    input_size = action_chunks.shape[1] * action_chunks.shape[2]
    avg_tokens = np.mean(token_lengths)
    compression_ratio = input_size / avg_tokens
    
    stats = {
        'compression_ratio': float(compression_ratio),
        'mean_token_length': float(np.mean(token_lengths)),
        'p99_token_length': float(np.percentile(token_lengths, 99)),
        'min_token_length': float(np.min(token_lengths)),
        'max_token_length': float(np.max(token_lengths)),
    }
    
    print(f"Compression Statistics:")
    print(f"  Average compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Mean token length: {stats['mean_token_length']:.1f}")
    print(f"  P99 token length: {stats['p99_token_length']:.0f}")
    print(f"  Min token length: {stats['min_token_length']:.0f}")
    print(f"  Max token length: {stats['max_token_length']:.0f}")
    
    return stats


def main(
    repo_id: str,
    root: str | None = None,
    action_horizon: int = 10,
    max_episodes: int | None = None,
    sample_fraction: float = 0.1,
    encoded_dims: str = "0:6,7:23",
    delta_dims: str | None = None,
    use_delta_transform: bool = False,
    state_key: str = "observation.state",
    vocab_size: int = 1024,
    scale: float = 10.0,
    output_dir: str | None = None,
):
    """
    Train FAST tokenizer for action encoding.
    
    Args:
        repo_id: LeRobot dataset repository ID
        root: Root directory for dataset (default: ~/.cache/huggingface/lerobot)
        action_horizon: Number of future actions in each chunk
        max_episodes: Max episodes to use (None = all episodes in dataset)
        sample_fraction: Fraction of chunks to sample per episode
        encoded_dims: Comma-separated dimension ranges to encode (e.g., "0:6,7:23")
        delta_dims: Comma-separated dimension indices for delta transform (e.g., "0,1,2,3,4,5")
        use_delta_transform: Whether to apply delta transform (relative actions vs absolute actions)
        state_key: Dataset key for state observations (default: "observation.state")
        vocab_size: FAST vocabulary size (BPE vocab size)
        scale: DCT scaling factor (default: 10.0)
        output_dir: Directory to save tokenizer (default: ./fast_tokenizer_{repo_id})
    """
    # Load dataset
    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    # Parse encoded dimensions
    encoded_dim_ranges = []
    for range_str in encoded_dims.split(','):
        start, end = map(int, range_str.strip().split(':'))
        encoded_dim_ranges.append((start, end))
    
    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    print(f"Encoding {total_encoded_dims} dimensions: {encoded_dims}")
    
    # Parse delta dimensions
    delta_dim_list = None
    if delta_dims is not None and delta_dims.strip():
        delta_dim_list = [int(d.strip()) for d in delta_dims.split(',')]
        print(f"Delta dimensions: {delta_dim_list}")
    else:
        print("No delta dimensions specified")
    
    print(f"Use delta transform: {use_delta_transform}")
    if use_delta_transform and (delta_dim_list is None or len(delta_dim_list) == 0):
        print("Warning: use_delta_transform=True but no delta_dims specified. No delta will be applied.")
    
    print(f"Action horizon: {action_horizon}")
    print(f"State key: {state_key}")
    
    # Determine episodes to process
    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(max_episodes, num_episodes)
    
    print(f"Processing {num_episodes} episodes...")
    
    # Process episodes sequentially (to avoid pickling issues with dataset)
    all_chunks = []
    for ep_idx in range(num_episodes):
        if ep_idx % 10 == 0:
            print(f"  Processing episode {ep_idx}/{num_episodes}...")
        
        chunks = process_episode(
            (dataset, ep_idx, action_horizon, delta_dim_list, sample_fraction, state_key, use_delta_transform)
        )
        if chunks is not None:
            all_chunks.append(chunks)
    
    # Concatenate all chunks
    all_chunks = np.concatenate(all_chunks, axis=0)
    print(f"Collected {len(all_chunks)} action chunks")
    
    # Extract only encoded dimensions FIRST (before normalization)
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(all_chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1)  # [N, H, D_encoded]
    print(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")
    
    # Apply normalization to encoded dimensions only
    # NOTE: For FAST, we ALWAYS use QUANTILE normalization (no per-timestamp)
    # This clips outliers and provides consistent [-1, 1] range for DCT compression
    print(f"\nBefore normalization - overall stats:")
    print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")
    
    norm_stats = dataset.meta.stats
    if norm_stats is not None and "action" in norm_stats:
        action_stats = norm_stats["action"]
        
        # Build encoded dimension indices
        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)
        
        # Use QUANTILE normalization: clip to [q01, q99] and map to [-1, 1]
        if "q01" in action_stats and "q99" in action_stats:
            q01 = np.array(action_stats["q01"])[encoded_dim_indices]  # [D_encoded]
            q99 = np.array(action_stats["q99"])[encoded_dim_indices]  # [D_encoded]
            
            print(f"\nNormalization stats (q01, q99) for encoded dimensions:")
            for i, dim_idx in enumerate(encoded_dim_indices):
                print(f"  Orig dim {dim_idx}: q01={q01[i]:7.4f}, q99={q99[i]:7.4f}, range={q99[i]-q01[i]:7.4f}")
            
            # Clip to quantile range and normalize to [-1, 1]
            encoded_chunks = np.clip(encoded_chunks, q01, q99)
            encoded_chunks = 2.0 * (encoded_chunks - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
            print(f"\nApplied quantile normalization [q01, q99] → [-1, 1]")
            
            print(f"\nAfter normalization - overall stats:")
            print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
            print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")
            
            print(f"\nPer-dimension stats (after normalization):")
            for d in range(encoded_chunks.shape[-1]):
                dim_data = encoded_chunks[:, :, d]
                print(f"  Dim {d}: min={np.min(dim_data):7.4f}, max={np.max(dim_data):7.4f}, "
                      f"mean={np.mean(dim_data):7.4f}, std={np.std(dim_data):7.4f}")
        else:
            print("Warning: q01/q99 stats not found, using raw actions")
    else:
        print("Warning: No normalization stats found, using raw actions")
    
    print(f"Encoded chunks shape: {encoded_chunks.shape}")
    
    # Train FAST tokenizer
    tokenizer = train_fast_tokenizer(
        encoded_chunks,
        vocab_size=vocab_size,
        scale=scale,
    )
    
    # Compute compression statistics
    compression_stats = compute_compression_stats(tokenizer, encoded_chunks)
    
    # Save tokenizer
    if output_dir is None:
        output_dir = f"fast_tokenizer_{repo_id.replace('/', '_')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(output_path)
    
    # Save metadata
    metadata = {
        'repo_id': repo_id,
        'vocab_size': vocab_size,
        'scale': scale,
        'encoded_dims': encoded_dims,
        'encoded_dim_ranges': encoded_dim_ranges,
        'total_encoded_dims': total_encoded_dims,
        'delta_dims': delta_dims,
        'delta_dim_list': delta_dim_list,
        'use_delta_transform': use_delta_transform,
        'state_key': state_key,
        'action_horizon': action_horizon,
        'num_training_chunks': len(encoded_chunks),
        'compression_stats': compression_stats,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved FAST tokenizer to {output_path}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    tyro.cli(main)
