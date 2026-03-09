# Copyright 2026 The HuggingFace Inc. team.
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

from __future__ import annotations

import numpy as np


def test_rtc_frozen_prefix_roundtrip_bytes() -> None:
    t = 7
    a = 13
    x = np.random.randn(t, a).astype(np.float32)

    payload = x.tobytes(order="C")
    y = np.frombuffer(payload, dtype=np.float32)
    assert y.size == t * a
    y = y.reshape(t, a)

    assert y.shape == x.shape
    assert np.allclose(y, x)


def test_actionsdense_raw_payload_sizes_match() -> None:
    # This mirrors the assumption used by the client and server:
    # `actions_f32` is packed float32 with shape (t, a).
    t = 5
    a = 4
    nbytes_expected = t * a * 4

    x = np.zeros((t, a), dtype=np.float32, order="C")
    assert len(x.tobytes(order="C")) == nbytes_expected
