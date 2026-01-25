#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

# Lazy imports to avoid conflicts with lerobot.policies.pi05.PI05Config
# when only importing subpackages like videoprism
def __getattr__(name):
    if name == "PI05Config":
        from .configuration_pi05 import PI05Config
        return PI05Config
    elif name == "PI05Policy":
        from .modeling_pi05 import PI05Policy
        return PI05Policy
    elif name == "make_pi05_pre_post_processors":
        from .processor_pi05 import make_pi05_pre_post_processors
        return make_pi05_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PI05Config", "PI05Policy", "make_pi05_pre_post_processors"]
