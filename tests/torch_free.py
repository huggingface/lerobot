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

The test process itself has PyTorch loaded (see `conftest.py`), so each check starts a subprocess. The code runs
twice: once with PyTorch installed as usual, checking that it never loads; and once with every attempt to load it
refused and recorded, so that an import wrapped in try/except, which would silently fall back, still fails.
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

_REFUSE_TORCH = """
import importlib.abc
import importlib.machinery
import sys
import traceback

_torch_attempts = []


class _RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        _torch_attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")


class _RefuseTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            spec.loader = _RefuseLoader()
        return spec


sys.meta_path.insert(0, _RefuseTorch())
"""

_CHECK_REFUSED = """
assert not _torch_attempts, "torch import attempted:\\n" + _torch_attempts[0]
"""


def assert_runs_without_torch(code: str, *args: str, cwd=None) -> None:
    """Run `code` in a fresh interpreter, with PyTorch installed and with PyTorch refused; it must load neither.

    `args` become `sys.argv[1:]` of the code.
    """
    code = textwrap.dedent(code)
    for script in (code + _CHECK_UNLOADED, _REFUSE_TORCH + code + _CHECK_REFUSED):
        result = subprocess.run(
            [sys.executable, "-c", script, *args], capture_output=True, text=True, cwd=cwd, timeout=300
        )
        assert result.returncode == 0, result.stdout + result.stderr
