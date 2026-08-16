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

import importlib.util
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _dataset_extra_pins() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        return list(tomllib.load(fh)["project"]["optional-dependencies"]["dataset"])


def _load_import_utils():
    path = REPO_ROOT / "src/lerobot/utils/import_utils.py"
    spec = importlib.util.spec_from_file_location("lerobot_import_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_extra_raises_torch_floor_to_torchcodec_abi():
    """lerobot[dataset] must not resolve torch 2.7 with torchcodec 0.11 (#4393)."""
    pins = _dataset_extra_pins()
    assert any(pin.startswith("torch>=2.11") for pin in pins), pins
    assert any(pin.startswith("torchvision>=0.26") for pin in pins), pins
    torchcodec_pins = [pin for pin in pins if pin.startswith("torchcodec>=")]
    assert torchcodec_pins, pins
    for pin in torchcodec_pins:
        assert pin.startswith("torchcodec>=0.11.0"), pin


def test_get_safe_default_video_backend_falls_back_when_torchcodec_unloadable(monkeypatch, caplog):
    import_utils = _load_import_utils()

    monkeypatch.setattr(
        import_utils.importlib.util, "find_spec", lambda name: object() if name == "torchcodec" else None
    )

    def _boom(name):
        if name == "torchcodec":
            raise OSError("undefined symbol: torch_dtype_float4_e2m1fn_x2")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(import_utils.importlib, "import_module", _boom)

    with caplog.at_level("WARNING"):
        assert import_utils.get_safe_default_video_backend() == "pyav"
    assert "cannot be loaded" in caplog.text
