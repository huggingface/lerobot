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

"""Camera pixels and freshness metadata, separate from dataset observation features."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """A decoded frame. Treat image as read-only while consumers share it.

    received_at is local time.monotonic() at decode/receipt, not sensor capture time.
    media_time_s is the stream-relative presentation time, or None for snapshots.
    It resets on WebRTC reconnection; session_id identifies the signalling session.
    sequence increases for every published frame within one MicroDuck instance.
    A standalone WebRTC receiver numbers frames within its session instead.
    """

    image: np.ndarray
    sequence: int
    received_at: float
    media_time_s: float | None = None
    session_id: str | None = None
