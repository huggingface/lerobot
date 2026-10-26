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

from lerobot.scripts.lerobot_info import get_package_version, get_sys_info


def test_get_package_version_not_installed():
    assert get_package_version("lerobot_missing_test_pkg") == "N/A"


def test_get_package_version_installed_but_import_fails(monkeypatch, tmp_path):
    # Like torchcodec when it can't load FFmpeg: the package is installed, but importing it raises.
    (tmp_path / "lerobot_broken_test_pkg.py").write_text(
        "raise RuntimeError('Could not load libtorchcodec')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert get_package_version("lerobot_broken_test_pkg") == "Installed (import failed: RuntimeError)"


def test_get_sys_info_survives_broken_package(monkeypatch):
    import lerobot.scripts.lerobot_info as lerobot_info

    real_import_module = lerobot_info.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "torchcodec":
            raise RuntimeError("Could not load libtorchcodec")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(lerobot_info.importlib, "import_module", import_module)
    info = get_sys_info()
    assert info["Torchcodec version"].endswith("(import failed: RuntimeError)")
