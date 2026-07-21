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

"""Unit tests for the capture-seq alignment buffer (no aiortc needed)."""

import numpy as np

from lerobot.robots.webrtc_proxy.alignment import AlignmentBuffer


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_empty_buffer_assembles_to_none():
    buf = AlignmentBuffer()
    assert buf.assemble() is None
    assert not buf.has_state()


def test_state_without_frame_is_incomplete():
    buf = AlignmentBuffer()
    buf.add_state(seq=0, t=1.0, joints={"shoulder_pan.pos": 10.0})
    assert buf.has_state()
    assert buf.assemble() is None  # no frame with that seq yet


def test_pairs_state_and_frame_by_seq():
    buf = AlignmentBuffer()
    buf.add_frame(seq=7, frame=_frame(2))
    buf.add_state(seq=7, t=1.12, joints={"a.pos": 0.0})
    aligned = buf.assemble()
    assert aligned is not None
    assert aligned.seq == 7
    assert aligned.t == 1.12
    assert int(aligned.frame[0, 0, 0]) == 2


def test_assemble_uses_freshest_complete_seq():
    buf = AlignmentBuffer()
    buf.add_state(seq=0, t=1.0, joints={"a.pos": 1.0})
    buf.add_frame(seq=0, frame=_frame(1))
    buf.add_state(seq=1, t=2.0, joints={"a.pos": 2.0})
    buf.add_frame(seq=1, frame=_frame(2))
    aligned = buf.assemble()
    assert aligned.seq == 1
    assert aligned.joints == {"a.pos": 2.0}


def test_incomplete_newest_seq_falls_back_to_prior_complete():
    buf = AlignmentBuffer()
    buf.add_state(seq=0, t=1.0, joints={"a.pos": 1.0})
    buf.add_frame(seq=0, frame=_frame(1))
    # seq 1 frame arrives but its state dropped (state channel is unreliable) -> incomplete.
    buf.add_frame(seq=1, frame=_frame(2))
    assert buf.assemble().seq == 0  # newest COMPLETE pair, not the frame-only seq 1


def test_history_is_bounded():
    buf = AlignmentBuffer(maxlen=2)
    for i in range(5):
        buf.add_frame(seq=i, frame=_frame(i))
        buf.add_state(seq=i, t=float(i), joints={"a.pos": float(i)})
    aligned = buf.assemble()
    assert aligned.seq == 4
    assert int(aligned.frame[0, 0, 0]) == 4
