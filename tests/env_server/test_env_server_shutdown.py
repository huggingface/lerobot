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

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.env_server.client import EnvClient


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not deliver POSIX SIGTERM callbacks")
@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("interrupt_wait", [False, True])
def test_cli_signal_closes_active_world(tmp_path, interrupt_wait, shutdown_signal):
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    config = tmp_path / "server.yaml"
    config.write_text(f'sim:\n  type: toy\nzenoh:\n  listen_endpoints: ["{endpoint}"]\n')
    arm = tmp_path / "arm"
    injected = tmp_path / "injected"
    closed = tmp_path / "closed"
    restored = tmp_path / "restored"
    # The trace controls signal timing. It leaves Event.wait and the CLI unchanged.
    (tmp_path / "sitecustomize.py").write_text(
        """
import atexit
import inspect
import os
from pathlib import Path
import signal
import sys
import threading

from lerobot.sims.backend import Backend

arm = Path(os.environ['SHUTDOWN_ARM'])
injected = Path(os.environ['SHUTDOWN_INJECTED'])
closed = Path(os.environ['SHUTDOWN_CLOSED'])
restored = Path(os.environ['SHUTDOWN_RESTORED'])
original_handler = signal.getsignal(signal.SIGTERM)
def check_handler():
    if signal.getsignal(signal.SIGTERM) == original_handler:
        restored.touch()
atexit.register(check_handler)
source, first_line = inspect.getsourcelines(threading.Event.wait)
locked_line = first_line + next(i for i, line in enumerate(source) if 'signaled = self._flag' in line)
fired = False

def profile(frame, event, arg):
    if event == 'return' and frame.f_code is Backend.close.__code__:
        with closed.open('a') as marker:
            marker.write('closed\\n')

def trace(frame, event, arg):
    global fired
    if (not fired and event == 'line' and frame.f_code is threading.Event.wait.__code__
        and frame.f_lineno == locked_line and frame.f_back is not None
        and frame.f_back.f_code.co_name == 'serve' and arm.exists()):
        fired = True
        assert frame.f_locals['self']._cond._is_owned()
        injected.touch()
        os.kill(os.getpid(), int(os.environ['SHUTDOWN_SIGNAL']))
    return trace

sys.setprofile(profile)
sys.settrace(trace)
"""
    )
    env = os.environ.copy()
    # Import the checked-out source even when pytest uses an editable installation elsewhere.
    source = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), source, env.get("PYTHONPATH", "")])
    env.update(
        SHUTDOWN_ARM=str(arm),
        SHUTDOWN_INJECTED=str(injected),
        SHUTDOWN_CLOSED=str(closed),
        SHUTDOWN_RESTORED=str(restored),
        SHUTDOWN_SIGNAL=str(int(shutdown_signal)),
    )
    client = EnvClient(endpoint, timeout_s=5)
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "lerobot.scripts.lerobot_env_server", "--config", str(config)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert process.poll() is None
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", int(endpoint.rsplit(":", 1)[1]))) == 0:
                        break
                time.sleep(0.02)
            else:
                pytest.fail("The environment server did not listen within ten seconds")
            client.open(1, "lockstep", seeds=[42])
            result = client.request("step", actions=np.zeros((1, 2), np.float32))["result"]
            assert result["step"].tolist() == [1]
            if interrupt_wait:
                arm.touch()
            else:
                process.send_signal(shutdown_signal)
            assert process.wait(timeout=5) == 0
            assert injected.exists() == interrupt_wait
            assert restored.exists()
            # The descriptor probe and the active session both close their backend.
            assert closed.read_text().splitlines() == ["closed", "closed"]
        finally:
            client.transport.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
