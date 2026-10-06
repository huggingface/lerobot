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

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Platforms with a torchcodec 0.11.x wheel vs PyAV-only fallbacks.
TORCHCODEC_WHEEL_PLATFORMS = [
    ("linux", "x86_64", True),
    ("linux", "AMD64", True),
    ("linux", "aarch64", True),
    ("linux", "arm64", True),
    ("darwin", "arm64", True),
    ("win32", "AMD64", True),
    ("win32", "x86_64", True),
    ("darwin", "x86_64", False),
    ("win32", "ARM64", False),
    ("linux", "armv7l", False),
]


def _dataset_extra_pins() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        return list(tomllib.load(fh)["project"]["optional-dependencies"]["dataset"])


def _abi_requirements() -> dict[str, Requirement]:
    reqs = {}
    for pin in _dataset_extra_pins():
        req = Requirement(pin)
        if req.name in {"torch", "torchvision", "torchcodec"}:
            reqs[req.name] = req
    return reqs


def _marker_env(sys_platform: str, platform_machine: str) -> dict[str, str]:
    env = default_environment()
    env["sys_platform"] = sys_platform
    env["platform_machine"] = platform_machine
    env["os_name"] = "nt" if sys_platform == "win32" else "posix"
    env["platform_system"] = {"linux": "Linux", "darwin": "Darwin", "win32": "Windows"}[sys_platform]
    return env


def _load_import_utils():
    path = REPO_ROOT / "src/lerobot/utils/import_utils.py"
    spec = importlib.util.spec_from_file_location("lerobot_import_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_extra_scopes_torch_abi_tuple_to_torchcodec_wheels():
    """Stricter torch/torchvision/torchcodec pins share one wheel-platform marker (#4393)."""
    reqs = _abi_requirements()
    assert set(reqs) == {"torch", "torchvision", "torchcodec"}
    assert reqs["torch"].specifier == SpecifierSet(">=2.11,<2.12.0")
    assert reqs["torchvision"].specifier == SpecifierSet(">=0.26.0,<0.27.0")
    assert reqs["torchcodec"].specifier == SpecifierSet(">=0.11.0,<0.12.0")

    assert reqs["torch"].marker is not None
    assert reqs["torch"].marker == reqs["torchvision"].marker == reqs["torchcodec"].marker

    for sys_platform, platform_machine, expect_wheel in TORCHCODEC_WHEEL_PLATFORMS:
        env = _marker_env(sys_platform, platform_machine)
        matched = reqs["torchcodec"].marker.evaluate(env)
        assert matched is expect_wheel, (sys_platform, platform_machine, matched)


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
