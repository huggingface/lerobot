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
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def get_device_memory_stats(device_idx: int = 0) -> dict[str, int]:
    """Retrieve GPU memory statistics for a specific device.
    
    Args:
        device_idx: CUDA device index to query. Defaults to 0.
    
    Returns:
        Dictionary with keys 'free', 'total', 'reserved' (in bytes).
    
    Raises:
        RuntimeError: If CUDA is not available or device doesn't exist.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(f"Device {device_idx} does not exist. Available devices: {torch.cuda.device_count()}")
    
    with torch.cuda.device(device_idx):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        free_bytes = torch.cuda.mem_get_info(device_idx)[0]
        total_bytes = torch.cuda.mem_get_info(device_idx)[1]
        reserved_bytes = torch.cuda.memory_reserved(device_idx)
    
    return {
        "free": int(free_bytes),
        "total": int(total_bytes),
        "reserved": int(reserved_bytes),
    }


def estimate_batch_size_from_memory(
    peak_memory_per_sample: int,
    available_memory: int,
    safety_margin: float = 0.9,
    min_size: int = 1,
    max_size: int = 256,
    is_main_process: bool = True,
) -> int:
    """Estimate safe batch size based on available GPU memory.
    
    Calculates batch size by dividing available memory by per-sample peak memory,
    then applies a safety margin to prevent OOM.
    
    Args:
        peak_memory_per_sample: Peak GPU memory used by a single sample (bytes).
        available_memory: Available GPU memory (bytes).
        safety_margin: Fraction of memory to use (0.0-1.0). Defaults to 0.9.
        min_size: Minimum batch size. Defaults to 1.
        max_size: Maximum batch size. Defaults to 256.
        is_main_process: Whether to log. Defaults to True.
    
    Returns:
        Estimated batch size clamped to [min_size, max_size].
    
    Raises:
        ValueError: If peak_memory_per_sample <= 0, available_memory < 0, or safety_margin not in (0.0, 1.0].
    """
    if peak_memory_per_sample <= 0:
        raise ValueError(f"peak_memory_per_sample must be > 0, got {peak_memory_per_sample}")
    
    if available_memory < 0:
        raise ValueError(f"available_memory must be >= 0, got {available_memory}")
    
    if not (0.0 < safety_margin <= 1.0):
        raise ValueError(f"safety_margin must be in (0.0, 1.0], got {safety_margin}")
    
    usable_memory = int(available_memory * safety_margin)
    estimated_batch_size = max(min_size, usable_memory // peak_memory_per_sample)
    final_batch_size = min(estimated_batch_size, max_size)
    
    if is_main_process:
        logging.info(
            f"Adaptive batch sizing: peak_memory_per_sample={peak_memory_per_sample / 1e9:.2f}GB, "
            f"available_memory={available_memory / 1e9:.2f}GB, "
            f"safety_margin={safety_margin}, "
            f"estimated_batch_size={final_batch_size}"
        )
    
    return final_batch_size