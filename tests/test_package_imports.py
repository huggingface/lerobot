#!/usr/bin/env python

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

import subprocess
import sys
from pathlib import Path


def test_lerobot_package_directory_does_not_shadow_stdlib_types():
    package_dir = Path(__file__).parents[1] / "src" / "lerobot"
    command = (
        "from types import DynamicClassAttribute, MappingProxyType; "
        "from lerobot.lerobot_types import RobotAction"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=package_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
