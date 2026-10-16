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
# pytest tests/cameras/test_zmq.py::test_failed_connect_stops_the_read_thread
# ```

import pytest

pytest.importorskip("zmq", reason="pyzmq is required (install lerobot[pyzmq-dep])")

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig  # noqa: E402

# A port with no publisher, so the warmup read never gets a frame.
_SILENT_PORT = 55917


def test_failed_connect_stops_the_read_thread():
    """A `connect()` that fails after the read thread started must not leave it running.

    The failure path calls `_cleanup()`, which closed the socket and terminated the
    context but never asked the thread to stop -- so the thread stayed blocked in
    `recv_string()` on a socket being closed from another thread, and a retry loop
    accumulated live threads.
    """
    config = ZMQCameraConfig(
        server_address="127.0.0.1",
        port=_SILENT_PORT,
        width=640,
        height=480,
        timeout_ms=500,
        warmup_s=1,
    )
    camera = ZMQCamera(config)

    with pytest.raises(RuntimeError, match="Failed to connect"):
        camera.connect()

    assert camera.thread is None
    assert camera.stop_event is None
    assert not camera.is_connected
    assert camera.latest_frame is None
