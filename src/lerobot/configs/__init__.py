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
Public API for lerobot configuration types and base config classes.

The exports are resolved lazily so lightweight scripts can import
``lerobot.configs.parser`` without also importing training transforms,
optimizers, diffusers, torch, and torchvision.
"""

_SUBMODULE_ATTRS = {
    ".dataset": ["DatasetRecordConfig"],
    ".default": ["DatasetConfig", "EvalConfig", "PeftConfig", "WandBConfig"],
    ".policies": ["PreTrainedConfig"],
    ".types": [
        "FeatureType",
        "NormalizationMode",
        "PipelineFeatureType",
        "PolicyFeature",
        "RTCAttentionSchedule",
    ],
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
