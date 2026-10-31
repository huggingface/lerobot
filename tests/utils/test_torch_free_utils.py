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

"""The helpers a control loop shares with `lerobot-rollout` run without PyTorch."""

import subprocess
import sys

from tests.torch_free import assert_runs_without_torch


def test_a_numpy_tick_runs_without_torch():
    assert_runs_without_torch(
        """
        import numpy as np

        from lerobot.utils.action_dispatch import dispatch_tick
        from lerobot.utils.action_interpolator import ActionInterpolator

        actions = [np.array([0.0]), np.array([2.0])]
        sent = []


        class Source:
            def dispatch_allowed(self):
                return True

            def acknowledge_hold(self):
                pass

            def get_action(self, obs_frame):
                return actions.pop(0) if actions else None

            def record_dispatch(self):
                pass


        source, interpolator = Source(), ActionInterpolator(multiplier=2)
        for _ in range(3):
            dispatch_tick(source, interpolator, send=lambda target: sent.append(float(target[0])))
        assert sent == [0.0, 1.0, 2.0], sent
        """
    )


def test_the_status_logs_run_without_torch():
    assert_runs_without_torch(
        """
        import logging

        from lerobot.utils.cycle_timer import CycleTimer
        from lerobot.utils.latency import LatencyWindow
        from lerobot.utils.rollout_status import _log_status

        logging.getLogger().setLevel(logging.INFO)
        timer = CycleTimer(30)
        _log_status(timer, timer.snapshot(), LatencyWindow(), "cuda:0", 0.0)
        """
    )


def test_the_status_logs_skip_the_gpu_memory_while_another_thread_imports_torch(tmp_path):
    (tmp_path / "torch.py").write_text(
        """
import __main__
from types import SimpleNamespace

__main__.release.wait()
cuda = SimpleNamespace(is_initialized=lambda: True, mem_get_info=lambda device: (1, 3))
"""
    )
    code = """
import sys
import threading
import time

from lerobot.utils.rollout_status import _gpu_memory_bytes

sys.path.insert(0, sys.argv[1])
release = threading.Event()
importer = threading.Thread(target=__import__, args=("torch",), daemon=True)
importer.start()
while "torch" not in sys.modules:
    time.sleep(0.01)
assert _gpu_memory_bytes("cuda:0") is None
release.set()
importer.join()
assert _gpu_memory_bytes("cuda:0") == (2, 3)
assert _gpu_memory_bytes("cpu") is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
