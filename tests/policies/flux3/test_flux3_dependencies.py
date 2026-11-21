# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Optional VAE imports must preserve the cause of dependency failures."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("failure", ["missing", "broken_natten", "unrelated"])
def test_vae_dependency_errors(failure):
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        from lerobot.utils import import_utils

        failure = sys.argv[1]
        import_utils._natten_available = failure == "broken_natten"
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "natten" or name.startswith("natten."):
                if failure == "missing":
                    raise AssertionError("Missing NATTEN should not be imported")
                raise ImportError("incompatible NATTEN wheel")
            if (
                failure == "unrelated"
                and name == "safetensors.torch"
                and (globals or {}).get("__name__") == "lerobot.policies.flux3.f3.video_vae"
            ):
                raise ImportError("unrelated safetensors failure")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            f3 = importlib.import_module("lerobot.policies.flux3.f3")
        except ImportError as exc:
            expected = {
                "broken_natten": "incompatible NATTEN wheel",
                "unrelated": "unrelated safetensors failure",
            }
            assert failure in expected, exc
            assert str(exc) == expected[failure], exc
        else:
            assert failure == "missing"
            import torch

            attention = f3.video_vae.Natten3D(16, [3, 3, 3], 2)
            assert attention(torch.randn(1, 3, 4, 4, 16)).shape == (1, 3, 4, 4, 16)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, failure], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
