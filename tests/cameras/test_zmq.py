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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_zmq.py::test_read_loop_does_not_publish_after_stop_requested
# ```

from threading import Event
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("zmq", reason="pyzmq is required (install lerobot[pyzmq-dep])")

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig  # noqa: E402


def test_read_loop_does_not_publish_after_stop_requested():
    """A read landing after a stop was requested must not repopulate the frame buffer.

    `_stop_read_thread` gives up joining after 2s while `recv_string` blocks for up to
    `timeout_ms`, so a late frame would otherwise resurrect the buffer that was just
    cleared and be seen as a fresh frame by the next connect attempt.
    """
    config = ZMQCameraConfig(server_address="127.0.0.1", width=640, height=480, warmup_s=0)
    camera = ZMQCamera(config)
    camera.stop_event = Event()

    def read_then_request_stop():
        # the stop lands while this read is in flight
        camera.stop_event.set()
        return np.zeros((480, 640, 3), np.uint8)

    with patch.object(camera, "_read_from_hardware", side_effect=read_then_request_stop):
        camera._read_loop()

    assert camera.latest_frame is None
    assert camera.latest_timestamp is None
    assert not camera.new_frame_event.is_set()
