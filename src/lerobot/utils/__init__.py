# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Public API for lightweight utilities.

Exports are resolved lazily so importing a utility submodule does not
eagerly import torch through ``device_utils``.
"""

_SUBMODULE_ATTRS = {
    ".constants": [
        "ACTION",
        "DEFAULT_FEATURES",
        "DONE",
        "IMAGENET_STATS",
        "OBS_ENV_STATE",
        "OBS_IMAGE",
        "OBS_IMAGES",
        "OBS_STATE",
        "OBS_STR",
        "REWARD",
    ],
    ".decorators": ["check_if_already_connected", "check_if_not_connected"],
    ".device_utils": ["auto_select_torch_device", "get_safe_torch_device", "is_torch_device_available"],
    ".errors": ["DeviceAlreadyConnectedError", "DeviceNotConnectedError"],
    ".import_utils": ["is_package_available", "require_package"],
}

_ATTR_TO_MODULE: dict[str, str] = {}
for _mod, _attrs in _SUBMODULE_ATTRS.items():
    for _attr in _attrs:
        _ATTR_TO_MODULE[_attr] = _mod


def __getattr__(name: str):
    if name in _ATTR_TO_MODULE:
        import importlib

        mod = importlib.import_module(_ATTR_TO_MODULE[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_ATTR_TO_MODULE.keys())
