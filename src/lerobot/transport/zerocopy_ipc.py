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

"""
Zero-Copy POSIX Shared Memory Inter-Process Communication (IPC) for LeRobot.

Eliminates memory copies and bypasses the Python GIL during high-bandwidth
sensor telemetry ingestion (e.g. 4K camera frames @ 60 FPS, multi-channel tactile arrays).
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError:
    torch = None
    IterableDataset = object

try:
    import lerobot_ipc
except ImportError:
    lerobot_ipc = None


def is_zerocopy_available() -> bool:
    """Returns True if native lerobot_ipc shared memory C++ library is installed."""
    return lerobot_ipc is not None


class ZeroCopyDataset(IterableDataset):
    """
    Zero-Copy PyTorch IterableDataset stream for high-frequency sensor ingestion.
    """

    def __init__(self, channel_name: str, timeout_ms: int = 1000, max_frames: Optional[int] = None):
        if not is_zerocopy_available():
            raise ImportError(
                "lerobot_ipc package is required for zero-copy shared memory transport. "
                "Install it via `pip install lerobot_zerocopy_ipc`."
            )
        super().__init__()
        self.channel_name = channel_name
        self.timeout_ms = timeout_ms
        self.max_frames = max_frames
        self.subscriber = lerobot_ipc.Subscriber(channel_name, timeout_ms)

    def __iter__(self):
        count = 0
        while self.max_frames is None or count < self.max_frames:
            frame = self.subscriber.acquire_frame(self.timeout_ms)
            if frame is None:
                continue

            np_arr = frame["data"]
            tensor = lerobot_ipc.numpy_to_torch_zerocopy(np_arr)
            meta = frame["metadata"]

            count += 1
            yield {
                "pixel_values": tensor,
                "timestamp_ns": meta.timestamp_ns,
                "sequence_id": meta.sequence_id,
                "slot_index": meta.slot_index,
            }
