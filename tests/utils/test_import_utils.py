# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import lerobot.utils.import_utils as import_utils


def test_get_safe_default_video_backend_pyav_when_absent(monkeypatch):
    monkeypatch.setattr(import_utils.importlib.util, "find_spec", lambda name: None)
    assert import_utils.get_safe_default_video_backend() == "pyav"


def test_get_safe_default_video_backend_torchcodec_when_loadable(monkeypatch):
    monkeypatch.setattr(import_utils.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(import_utils.importlib, "import_module", lambda name, *a, **k: object())
    assert import_utils.get_safe_default_video_backend() == "torchcodec"


def test_get_safe_default_video_backend_pyav_when_installed_but_unloadable(monkeypatch):
    # find_spec succeeds (the package is installed) but the runtime import fails,
    # as on Windows when the FFmpeg shared libraries torchcodec links against are
    # missing. The default must fall back to pyav instead of crashing at decode time.
    monkeypatch.setattr(import_utils.importlib.util, "find_spec", lambda name: object())

    def raise_import_error(name, *args, **kwargs):
        raise ImportError("DLL load failed while importing _core: module not found")

    monkeypatch.setattr(import_utils.importlib, "import_module", raise_import_error)
    assert import_utils.get_safe_default_video_backend() == "pyav"
