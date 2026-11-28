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

"""Check that code runs without importing PyTorch, in a fresh process.

The test process itself has PyTorch loaded (see `conftest.py`), so each check starts a subprocess.
"""

import subprocess
import sys
import textwrap

_CHECK_UNLOADED = """
import sys
assert "torch" not in sys.modules, "torch was imported"
if sys.platform == "linux":
    with open("/proc/self/maps") as maps:
        mapped = {line.split()[-1] for line in maps if "libtorch" in line or "libc10" in line}
    assert not mapped, f"PyTorch libraries are mapped: {sorted(mapped)}"
"""


def assert_runs_without_torch(code: str, *args: str) -> None:
    """Run `code` in a fresh interpreter with PyTorch installed; it must not load PyTorch.

    `args` become `sys.argv[1:]` of the code.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code) + _CHECK_UNLOADED, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
