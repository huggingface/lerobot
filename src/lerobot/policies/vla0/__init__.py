# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.policies.vla0.configuration_vla0 import VLA0Config

# VLA0Policy is imported lazily to avoid loading transformers at import time
__all__ = ["VLA0Config", "VLA0Policy"]


def __getattr__(name):
    if name == "VLA0Policy":
        from lerobot.policies.vla0.modeling_vla0 import VLA0Policy

        return VLA0Policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
