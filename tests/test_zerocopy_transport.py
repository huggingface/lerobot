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

import time
import pytest
import numpy as np

from lerobot.transport import (
    is_zerocopy_available,
    ZeroCopyPublisher,
    ZeroCopyDataset,
    MultiChannelZeroCopyDataset,
)

@pytest.mark.skipif(not is_zerocopy_available(), reason="lerobot_ipc library not installed")
def test_zerocopy_transport_end_to_end():
    ch_name = "/test_lerobot_transport_ch"
    pub = ZeroCopyPublisher(channel_name=ch_name, num_slots=4)
    ds = ZeroCopyDataset(channel_name=ch_name, timeout_ms=1000, max_frames=1)
    
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 127
    pub.write_frame(dummy_frame)

    samples = list(ds)
    assert len(samples) == 1
    sample = samples[0]
    
    tensor = sample["pixel_values"]
    assert tensor.shape == (480, 640, 3)
    assert tensor[0, 0, 0].item() == 127
    print("test_zerocopy_transport_end_to_end PASSED")


@pytest.mark.skipif(not is_zerocopy_available(), reason="lerobot_ipc library not installed")
def test_multichannel_zerocopy_transport():
    ch_cam = "/test_lerobot_multicam"
    ch_tactile = "/test_lerobot_multitactile"

    pub_cam = ZeroCopyPublisher(channel_name=ch_cam, num_slots=4)
    pub_tactile = ZeroCopyPublisher(channel_name=ch_tactile, num_slots=4)

    ds = MultiChannelZeroCopyDataset(channels=[ch_cam, ch_tactile], timeout_ms=1000, max_frames=1)

    pub_cam.write_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    pub_tactile.write_frame(np.zeros((16, 16, 3), dtype=np.uint8))

    samples = list(ds)
    assert len(samples) == 1
    
    assert ch_cam in samples[0]
    assert ch_tactile in samples[0]
    print("test_multichannel_zerocopy_transport PASSED")


if __name__ == "__main__":
    test_zerocopy_transport_end_to_end()
    test_multichannel_zerocopy_transport()
