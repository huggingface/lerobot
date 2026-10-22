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

"""An owned SSH forward to robotd; explicit socket addresses remain usable."""

import contextlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import IO


class MicroDuckConnection:
    def __init__(self, host: str, user: str, port: int, timeout_s: float):
        self.host, self.user, self.port, self.timeout_s = host, user, port, timeout_s
        self._process: subprocess.Popen | None = None
        self._directory: str | None = None
        self._stderr: IO[bytes] | None = None
        self._resources = contextlib.ExitStack()

    def open(self) -> str:
        if self._process is not None:
            raise RuntimeError("The Microduck SSH connection is already open")
        ssh = shutil.which("ssh")
        if ssh is None:
            raise FileNotFoundError("Install OpenSSH to connect to Microduck automatically")
        try:
            # A private random directory keeps the socket within macOS's 104-byte path limit.
            self._directory = self._resources.enter_context(
                tempfile.TemporaryDirectory(prefix="md-", dir="/tmp")  # nosec B108: mkdtemp creates mode 0700
            )
            address = str(Path(self._directory) / "robot.sock")
            self._stderr = self._resources.enter_context(tempfile.TemporaryFile())  # noqa: SIM115 - owned by ExitStack
            self._process = subprocess.Popen(
                [
                    ssh,
                    "-N",
                    "-T",
                    "-p",
                    str(self.port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=none",
                    "-o",
                    "ServerAliveInterval=5",
                    "-o",
                    "ServerAliveCountMax=2",
                    "-o",
                    f"ConnectTimeout={max(1, int(self.timeout_s))}",
                    "-L",
                    f"{address}:/run/robotd.sock",
                    "--",
                    f"{self.user}@{self.host}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
            deadline = time.monotonic() + self.timeout_s
            while self._process.poll() is None:
                if Path(address).is_socket():
                    return address
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"SSH connection to {self.host} did not become ready")
                time.sleep(0.05)
            self._stderr.seek(0)
            details = self._stderr.read().decode(errors="replace").strip()
            raise ConnectionError(
                f"Could not connect to Microduck at {self.host}: {details}. "
                "Verify the host with ssh and configure SSH key authentication first; "
                "LeRobot's automatic connection cannot prompt for passwords or host approval."
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
        self._resources.close()
        self._stderr = None
        self._directory = None
