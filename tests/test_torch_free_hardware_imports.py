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

"""Check hardware imports and public exports in fresh Python processes."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_so101_import_does_not_import_torch():
    result = _run_python(
        """
        import builtins
        import sys
        from typing import get_type_hints

        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("torch import is forbidden")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import

        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.hardware_types import RobotAction
        from lerobot.robots.so_follower.so_follower import SOFollower
        assert SO101Follower is not None
        assert SO101FollowerConfig is not None
        assert get_type_hints(SOFollower.send_action)["action"] == RobotAction
        assert "torch" not in sys.modules
        print("SO-101 import passed without torch")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SO-101 import passed" in result.stdout


def test_so101_constructor_does_not_import_torch_or_open_hardware():
    pytest.importorskip("rustypot")
    result = _run_python(
        """
        import builtins
        import json
        import sys
        import tempfile
        from pathlib import Path

        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("torch import is forbidden")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import

        def forbidden(*args, **kwargs):
            raise AssertionError("hardware access is forbidden")
        try:
            import serial
            serial.Serial = forbidden
        except ImportError:
            pass
        import rustypot
        rustypot.Bus = forbidden

        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        calibration = {
            name: {"id": index, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for index, name in enumerate(names, start=1)
        }
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "test.json").write_text(json.dumps(calibration))
            robot = SO101Follower(SO101FollowerConfig(
                port="/nonexistent/torch-free-test", id="test", calibration_dir=Path(folder), use_degrees=True
            ))
            assert len(robot.calibration) == 6
            assert not robot.is_connected
            assert "torch" not in sys.modules
        print("constructor passed without torch or hardware access")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "constructor passed" in result.stdout


def test_type_aliases_and_lazy_public_utility_exports():
    result = _run_python(
        """
        import lerobot.hardware_types as hardware_types
        import lerobot.lerobot_types as lerobot_types
        import lerobot.utils as utils
        import numpy as np
        import torch
        from typing import get_args, get_type_hints

        assert lerobot_types.RobotAction is hardware_types.RobotAction
        assert lerobot_types.RobotObservation is hardware_types.RobotObservation
        expected = {
            "ActionInterpolator",
            "auto_select_torch_device",
            "get_safe_torch_device",
            "is_torch_device_available",
        }
        assert expected <= set(utils.__all__)
        assert expected <= set(dir(utils))
        assert not expected.intersection(utils.__dict__)

        from lerobot.utils.action_interpolator import ActionInterpolator
        from lerobot.utils.device_utils import (
            auto_select_torch_device,
            get_safe_torch_device,
            is_torch_device_available,
        )
        assert utils.ActionInterpolator is ActionInterpolator
        assert utils.auto_select_torch_device is auto_select_torch_device
        assert utils.get_safe_torch_device is get_safe_torch_device
        assert utils.is_torch_device_available is is_torch_device_available
        assert expected <= set(utils.__dict__)

        namespace = {}
        exec("from lerobot.utils import *", namespace)
        assert all(namespace[name] is getattr(utils, name) for name in expected)
        try:
            utils.not_a_public_export
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown utility attribute must raise AttributeError")

        action_type = get_type_hints(lerobot_types.EnvTransition)["action"]
        assert {torch.Tensor, hardware_types.RobotAction, np.ndarray, type(None)} <= set(get_args(action_type))
        print("aliases and public exports passed")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "aliases and public exports passed" in result.stdout
