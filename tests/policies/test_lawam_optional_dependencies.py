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

"""Verify that base installations can import policies without LaWAM extras."""

import subprocess
import sys
import textwrap


def test_lawam_import_without_optional_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import importlib.util
                import sys
                from unittest.mock import patch

                find_spec = importlib.util.find_spec
                missing = {"transformers", "diffusers"}

                def without_extras(name, *args, **kwargs):
                    if name.split(".")[0] in missing:
                        return None
                    return find_spec(name, *args, **kwargs)

                with patch.dict(sys.modules, {name: None for name in missing}), patch(
                    "importlib.util.find_spec", side_effect=without_extras
                ):
                    from lerobot.policies import PreTrainedPolicy
                    from lerobot.policies.lawam.latent_world.processor_utils import (
                        LatentWorldProcessorSpec,
                        load_latent_world_processor,
                    )

                    try:
                        load_latent_world_processor(LatentWorldProcessorSpec("unused", "<latent>"))
                    except ImportError as exc:
                        assert "lerobot[lawam]" in str(exc), str(exc)
                    else:
                        raise AssertionError("Missing transformers should raise an installation hint")
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
