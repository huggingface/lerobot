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


@pytest.mark.parametrize("failure", ["missing", "broken_natten", "natten_without_kernels", "unrelated"])
def test_vae_dependency_errors(failure):
    script = textwrap.dedent(
        """
        import builtins
        import functools
        import importlib
        import logging
        import sys
        import types

        from lerobot.utils import import_utils

        failure = sys.argv[1]
        import_utils._natten_available = failure in ("broken_natten", "natten_without_kernels")
        if failure == "natten_without_kernels":
            # A NATTEN wheel built for another torch imports, but its fused kernels do not load.
            natten = types.ModuleType("natten")
            natten.HAS_LIBNATTEN = False
            natten.functional = types.ModuleType("natten.functional")
            natten.functional.na2d = natten.functional.na3d = None
            sys.modules.update({"natten": natten, "natten.functional": natten.functional})
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if (name == "natten" or name.startswith("natten.")) and failure != "natten_without_kernels":
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
            assert failure in ("missing", "natten_without_kernels")
            import torch

            # The public loader with fewer, smaller layers than the real VAE, to keep the test fast.
            f3.video_vae.ViTNormInferenceParams = functools.partial(
                f3.video_vae.ViTNormInferenceParams,
                embed_dim=16,
                window_size=[3, 3, 3],
                enc_depths=[1, 1, 1, 1],
                num_heads=[1, 1, 2, 2],
            )
            frame = torch.rand(1, 3, 1, 96, 96, dtype=torch.bfloat16)
            assert f3.load_video_vae(None).encode_task(frame).shape == (1, 96, 1, 3, 3)
            warnings = []
            handler = logging.Handler()
            handler.emit = warnings.append
            logging.getLogger(f3.video_vae.__name__).addHandler(handler)
            for _ in range(2):
                f3.video_vae._note_attention("cuda")
            assert len(warnings) == 1, warnings
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, failure], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
