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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_so101_import_does_not_import_torch():
    result = _run_python(
        """
        import sys
        from typing import get_type_hints

        sys.modules["torch"] = None
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.hardware_types import RobotAction
        from lerobot.robots.so_follower.so_follower import SOFollower

        assert get_type_hints(SOFollower.send_action)["action"] == RobotAction
        assert sys.modules["torch"] is None
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_aliases_and_lazy_utility_exports():
    result = _run_python(
        """
        import numpy as np
        import torch
        from typing import get_args, get_type_hints

        import lerobot.hardware_types as hardware_types
        import lerobot.lerobot_types as legacy_types
        import lerobot.utils as utils

        assert legacy_types.RobotAction is hardware_types.RobotAction
        assert legacy_types.RobotObservation is hardware_types.RobotObservation

        expected = {
            "ActionInterpolator",
            "auto_select_torch_device",
            "get_safe_torch_device",
            "is_torch_device_available",
        }
        assert expected <= set(utils.__all__) & set(dir(utils))
        namespace = {}
        exec("from lerobot.utils import *", namespace)
        from lerobot.utils import action_interpolator, device_utils
        assert namespace["ActionInterpolator"] is action_interpolator.ActionInterpolator
        for name in expected - {"ActionInterpolator"}:
            assert namespace[name] is getattr(device_utils, name)
        assert expected <= utils.__dict__.keys()

        action_type = get_type_hints(legacy_types.EnvTransition)["action"]
        assert {torch.Tensor, hardware_types.RobotAction, np.ndarray, type(None)} <= set(get_args(action_type))
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lightweight_utils_remain_usable_after_tensor_export_fails():
    result = _run_python(
        """
        import sys

        sys.modules["torch"] = None
        import lerobot.utils as utils

        assert "ActionInterpolator" in dir(utils)
        assert callable(utils.init_logging)
        assert not hasattr(utils, "not_a_public_utility")
        for name in ("ActionInterpolator", "get_safe_torch_device"):
            try:
                getattr(utils, name)
            except ModuleNotFoundError as exc:
                assert exc.name == "torch", str(exc)
            else:
                raise AssertionError(f"{name} unexpectedly loaded without torch")
        # A failed tensor-dependent import must not break direct hardware imports.
        from lerobot.robots.so_follower import SO101FollowerConfig
        assert SO101FollowerConfig(port="unused").port == "unused"
        assert callable(utils.init_logging)
        assert sys.modules["torch"] is None
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tensor_export_loads_on_first_use_in_fresh_process():
    result = _run_python(
        """
        import sys
        import lerobot.utils as utils

        assert "torch" not in sys.modules
        assert "ActionInterpolator" in dir(utils)
        assert "torch" not in sys.modules
        from lerobot.utils import ActionInterpolator, get_safe_torch_device
        from lerobot.utils.action_interpolator import ActionInterpolator as direct_class
        from lerobot.utils.device_utils import get_safe_torch_device as direct_function
        assert ActionInterpolator is direct_class
        assert get_safe_torch_device is direct_function
        assert str(get_safe_torch_device("cpu")) == "cpu"
        assert utils.ActionInterpolator is ActionInterpolator
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
