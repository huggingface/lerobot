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

"""Exercise owned SSH lifecycle without contacting hardware or changing SSH configuration."""

import socket
from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.robots.microduck.connection import MicroDuckConnection


def test_owned_forward_is_private_and_is_removed_on_close(monkeypatch):
    commands = []
    listeners = []
    processes = []

    def spawn(command, **kwargs):
        commands.append(command)
        address = command[command.index("-L") + 1].split(":")[0]
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(address)
        listeners.append(listener)
        process = Mock()
        process.poll.return_value = None
        processes.append(process)
        return process

    monkeypatch.setattr("lerobot.robots.microduck.connection.subprocess.Popen", spawn)
    connection = MicroDuckConnection("192.0.2.1", "microduck", 2222, 1)
    try:
        first = connection.open()
        assert Path(first).is_socket()
        assert Path(first).parent.stat().st_mode & 0o777 == 0o700
        assert commands[0][-2:] == ["--", "microduck@192.0.2.1"]
        assert "BatchMode=yes" in commands[0]
        assert "StrictHostKeyChecking=yes" in commands[0]
        connection.close()
        assert not Path(first).exists()
        processes[0].terminate.assert_called_once()
        second = connection.open()
        assert first != second
    finally:
        connection.close()
        for listener in listeners:
            listener.close()
    connection.close()


def test_authentication_failure_is_actionable_and_cleans_up(monkeypatch):
    address = None

    def spawn(command, **kwargs):
        nonlocal address
        address = command[command.index("-L") + 1].split(":")[0]
        kwargs["stderr"].write(b"Permission denied (publickey)")
        kwargs["stderr"].flush()
        process = Mock()
        process.poll.return_value = 255
        return process

    monkeypatch.setattr("lerobot.robots.microduck.connection.subprocess.Popen", spawn)
    connection = MicroDuckConnection("192.0.2.1", "microduck", 22, 1)
    with pytest.raises(ConnectionError, match="SSH key authentication"):
        connection.open()
    assert not Path(address).parent.exists()
    assert connection._process is None


def test_startup_timeout_terminates_owned_process(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr("lerobot.robots.microduck.connection.subprocess.Popen", lambda *a, **kw: process)
    connection = MicroDuckConnection("192.0.2.1", "microduck", 22, 0.001)
    with pytest.raises(TimeoutError):
        connection.open()
    process.terminate.assert_called_once()
    assert connection._directory is None


def test_missing_ssh_binary_cleans_up(monkeypatch):
    def spawn(*args, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr("lerobot.robots.microduck.connection.subprocess.Popen", spawn)
    connection = MicroDuckConnection("192.0.2.1", "microduck", 22, 1)
    with pytest.raises(FileNotFoundError):
        connection.open()
    assert connection._directory is None
    assert connection._stderr is None
