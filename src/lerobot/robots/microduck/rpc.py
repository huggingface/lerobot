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

"""Minimal client for the Microduck daemons' wire protocol.

Every Microduck daemon (``robotd``, ``updaterd``, ...) speaks JSON-RPC 2.0 framed as NDJSON (one JSON
object per line) over a Unix socket. From a laptop the socket is reached through an ssh forward, e.g.
``ssh -N -L /tmp/robotd.sock:/run/robotd.sock microduck@<ip>``, or through a TCP forward
(``ssh -N -L 7000:/run/robotd.sock microduck@<ip>``).

The daemons serve one request at a time per connection, and a subscription takes its connection over,
so callers open one connection per lane (commands, state stream).
"""

import contextlib
import itertools
import json
import socket
import threading
from typing import Any


class DuckRpcError(RuntimeError):
    """Raised when a daemon answers a request with a JSON-RPC error."""

    def __init__(self, method: str, error: dict[str, Any]):
        self.code = error.get("code")
        self.data = error.get("data")
        super().__init__(f"{method} failed ({self.code}): {error.get('message')}")


class DuckRpc:
    """One NDJSON JSON-RPC connection to a Microduck daemon.

    ``address`` is either a Unix socket path (``/tmp/robotd.sock``) or a TCP ``host:port``.
    """

    def __init__(self, address: str, timeout_s: float = 5.0):
        self.address = address
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._reader: Any = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def open(self) -> None:
        if self.address.startswith("/") or ":" not in self.address:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            target: Any = self.address
        else:
            host, port = self.address.rsplit(":", 1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            target = (host, int(port))
        sock.settimeout(self.timeout_s)
        try:
            sock.connect(target)
        except OSError:
            sock.close()
            raise
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
            self._sock = None

    def _send(self, message: dict[str, Any]) -> None:
        if self._sock is None:
            raise ConnectionError(f"No connection to {self.address}")
        self._sock.sendall((json.dumps(message) + "\n").encode())

    def read_message(self) -> dict[str, Any]:
        """Blocks until the next line arrives and returns it decoded."""
        if self._reader is None:
            raise ConnectionError(f"No connection to {self.address}")
        line = self._reader.readline()
        if not line:
            raise ConnectionError(f"{self.address} closed the connection")
        return json.loads(line)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Sends a notification: no id, no reply. Used for continuous intents (move, head)."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            self._send(message)

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Sends a request and returns its ``result``, skipping any notification received meanwhile."""
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            self._send(message)
            while True:
                reply = self.read_message()
                if reply.get("id") != request_id:
                    continue
                if "error" in reply:
                    raise DuckRpcError(method, reply["error"])
                return reply.get("result")
