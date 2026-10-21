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

import socket
import threading

import pytest

from lerobot.env_server.server import EnvServer, ServerConfig
from lerobot.transport.zenoh import ZenohConfig


@pytest.fixture
def server():
    # Live-server tests need the `remote` extra; the base-dependency tier skips them.
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    server = EnvServer(
        ServerConfig(
            deployment="toy_test", zenoh=ZenohConfig(listen_endpoints=[endpoint]), presence_grace_s=0.5
        )
    )
    thread = threading.Thread(target=server.serve)
    thread.start()
    assert server.ready.wait(5)
    yield server, endpoint
    server.stop.set()
    thread.join(5)
    assert not thread.is_alive()
