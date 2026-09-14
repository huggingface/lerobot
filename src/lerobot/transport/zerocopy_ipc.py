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


class ZeroCopyPublisher:
    """
    Zero-Copy Shared Memory Publisher for raw hardware drivers streaming sensor data.
    """

    def __init__(
        self,
        channel_name: str,
        num_slots: int = 8,
        overwrite_policy: str = "OVERWRITE_OLDEST",
    ):
        if not is_zerocopy_available():
            raise ImportError(
                "lerobot_ipc package is required for zero-copy shared memory transport. "
                "Install it via `pip install lerobot_zerocopy_ipc`."
            )
        
        policy_enum = lerobot_ipc.OverwritePolicy.OVERWRITE_OLDEST
        if overwrite_policy == "BLOCK_PRODUCER":
            policy_enum = lerobot_ipc.OverwritePolicy.BLOCK_PRODUCER
        elif overwrite_policy == "REJECT_WRITE":
            policy_enum = lerobot_ipc.OverwritePolicy.REJECT_WRITE

        self.publisher = lerobot_ipc.Publisher(channel_name, num_slots=num_slots, policy=policy_enum)
        self.channel_name = channel_name

    def write_frame(
        self,
        frame_data: np.ndarray,
        timestamp_ns: Optional[int] = None,
        timeout_ms: int = 1000,
    ) -> None:
        """
        Write numpy frame directly into shared memory slot with zero memory copy.
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        height, width = frame_data.shape[0], frame_data.shape[1]
        channels = frame_data.shape[2] if frame_data.ndim == 3 else 1

        buf = self.publisher.get_write_buffer(
            width=width,
            height=height,
            channels=channels,
            type=lerobot_ipc.SensorDataType.RGB_8U,
            timeout_ms=timeout_ms,
        )
        buf[...] = frame_data
        self.publisher.commit_write(buf.nbytes, timestamp_ns)


class ZeroCopyDataset(IterableDataset):
    """
    Zero-Copy PyTorch IterableDataset stream for single-channel sensor ingestion.
    """

    def __init__(
        self,
        channel_name: str,
        timeout_ms: int = 1000,
        max_frames: Optional[int] = None,
    ):
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


class MultiChannelZeroCopyDataset(IterableDataset):
    """
    Zero-Copy PyTorch IterableDataset stream for multi-modal sensor ingestion (e.g. multiple cameras + tactile grids).
    """

    def __init__(
        self,
        channels: List[str],
        timeout_ms: int = 1000,
        max_frames: Optional[int] = None,
    ):
        if not is_zerocopy_available():
            raise ImportError(
                "lerobot_ipc package is required for zero-copy shared memory transport. "
                "Install it via `pip install lerobot_zerocopy_ipc`."
            )
        super().__init__()
        self.channels = channels
        self.timeout_ms = timeout_ms
        self.max_frames = max_frames
        self.subscribers = {ch: lerobot_ipc.Subscriber(ch, timeout_ms) for ch in channels}

    def __iter__(self):
        count = 0
        while self.max_frames is None or count < self.max_frames:
            batch = {}
            all_ready = True
            for ch in self.channels:
                frame = self.subscribers[ch].acquire_frame(self.timeout_ms)
                if frame is None:
                    all_ready = False
                    break
                
                np_arr = frame["data"]
                tensor = lerobot_ipc.numpy_to_torch_zerocopy(np_arr)
                meta = frame["metadata"]

                batch[ch] = {
                    "data": tensor,
                    "timestamp_ns": meta.timestamp_ns,
                    "sequence_id": meta.sequence_id,
                }

            if all_ready:
                count += 1
                yield batch
