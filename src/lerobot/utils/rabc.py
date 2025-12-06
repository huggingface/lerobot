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

"""
Reward-Aligned Behavior Cloning (RA-BC) utilities.

RA-BC uses a pre-trained reward model (e.g., SARM) to compute progress-based weights
for training samples, emphasizing high-quality demonstrations and down-weighting
suboptimal ones.

Workflow:
1. Precompute weights using `examples/sarm/compute_rabc_weights.py`
2. Load weights during training with `RABCWeights`

Usage:
    weights = RABCWeights("rabc_weights.parquet")
    batch_weights = weights.compute_batch_weights(batch)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class RABCWeights:
    """
    Load and use precomputed RA-BC weights from a parquet file.
    
    Uses proper progress deltas (progress[t + chunk_size] - progress[t]) generated using: examples/sarm/compute_rabc_weights.py
    
    Args:
        weights_path: Path to parquet file with precomputed weights
        weight_column: Column name for weights (default: "rabc_weight")
        fallback_weight: Weight to use for missing frames (default: 1.0)
        device: Device to return tensors on
    """
    
    def __init__(
        self,
        weights_path: str | Path,
        weight_column: str = "rabc_weight",
        fallback_weight: float = 1.0,
        device: torch.device = None,
    ):
        self.weights_path = Path(weights_path)
        self.weight_column = weight_column
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load weights
        logging.info(f"Loading precomputed RA-BC weights from {self.weights_path}")
        self.df = pd.read_parquet(self.weights_path)
        
        # Create lookup dict: global_index -> weight
        self.weights_lookup = {}
        for _, row in self.df.iterrows():
            global_idx = int(row["global_index"])
            weight = row[weight_column]
            if not np.isnan(weight):
                self.weights_lookup[global_idx] = float(weight)
        
        logging.info(f"Loaded {len(self.weights_lookup)} frame weights")
        
        # Log statistics
        valid_weights = list(self.weights_lookup.values())
        if valid_weights:
            logging.info(
                f"Weight statistics: mean={np.mean(valid_weights):.4f}, "
                f"std={np.std(valid_weights):.4f}, "
                f"zeros={sum(1 for w in valid_weights if w == 0)}, "
                f"ones={sum(1 for w in valid_weights if w == 1)}"
            )
    
    def compute_batch_weights(self, batch: dict, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Get precomputed weights for a batch.
        
        Args:
            batch: Training batch containing "index" key with global frame indices
            epsilon: Small constant for numerical stability
            
        Returns:
            Weights tensor (batch_size,) normalized to sum to batch_size
        """
        # Get frame indices from batch
        indices = batch.get("index")
        if indices is None:
            logging.warning("RA-BC: Batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            return torch.ones(batch_size, device=self.device)
        
        # Convert to list of ints
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        # Lookup weights
        weights = []
        for idx in indices:
            idx = int(idx)
            weight = self.weights_lookup.get(idx, self.fallback_weight)
            weights.append(weight)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Normalize to sum to batch_size
        batch_size = len(weights)
        weight_sum = weights.sum() + epsilon
        weights = weights * batch_size / weight_sum
        
        return weights
    
    def _get_batch_size(self, batch: dict) -> int:
        """Determine batch size from batch."""
        for key in ["action", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    return val.shape[0]
        return 1
    
    def get_stats(self) -> dict:
        """Get weight statistics."""
        valid_weights = list(self.weights_lookup.values())
        return {
            "num_frames": len(self.weights_lookup),
            "mean": np.mean(valid_weights) if valid_weights else 0.0,
            "std": np.std(valid_weights) if valid_weights else 0.0,
            "zeros": sum(1 for w in valid_weights if w == 0),
            "ones": sum(1 for w in valid_weights if w == 1),
        }
