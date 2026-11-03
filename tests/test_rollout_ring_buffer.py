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

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.ring_buffer import RolloutRingBuffer  # noqa: E402

pytestmark = pytest.mark.dataset


def test_time_eviction_updates_byte_count():
    buffer = RolloutRingBuffer(max_seconds=1, max_memory_mb=1, fps=2)
    frames = [{"image": np.full(256 * 1024, i, dtype=np.uint8)} for i in range(8)]

    for index, frame in enumerate(frames):
        buffer.append(frame)
        assert buffer.estimated_bytes == min(index + 1, 2) * frame["image"].nbytes
        assert len(buffer) == min(index + 1, 2)

    retained = buffer.drain()
    assert [frame["image"][0] for frame in retained] == [6, 7]
    assert buffer.estimated_bytes == 0


def test_memory_eviction_counts_only_retained_frames():
    buffer = RolloutRingBuffer(max_seconds=10, max_memory_mb=1, fps=2)
    for index in range(4):
        buffer.append({"image": np.full(400 * 1024, index, dtype=np.uint8)})

    assert buffer.estimated_bytes == 800 * 1024
    assert [frame["image"][0] for frame in buffer.drain()] == [2, 3]


def test_oversized_frame_does_not_discard_valid_history():
    buffer = RolloutRingBuffer(max_seconds=10, max_memory_mb=1, fps=2)
    frame = {"value": 1.0}
    buffer.append(frame)
    buffer.append({"image": np.zeros(2 * 1024 * 1024, dtype=np.uint8)})

    assert buffer.estimated_bytes == 8
    assert buffer.drain() == [frame]


@pytest.mark.parametrize("max_seconds", [0, 0.01])
def test_zero_frame_capacity_does_not_accumulate_bytes(max_seconds):
    buffer = RolloutRingBuffer(max_seconds=max_seconds, max_memory_mb=1, fps=2)
    for index in range(5):
        buffer.append({"value": index})

    assert len(buffer) == 0
    assert buffer.estimated_bytes == 0
    assert buffer.drain() == []
