#!/usr/bin/env python

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

# NOTE: ``LingBotVAPolicy`` (and the Wan transformer it owns) imports ``diffusers`` as a
# hard dependency at class-definition time (it subclasses diffusers' ModelMixin/ConfigMixin).
# To keep base ``import lerobot`` working without the optional ``lingbot_va`` extra, the
# policy is exposed lazily via module ``__getattr__`` — the heavy import only happens when
# ``LingBotVAPolicy`` is actually accessed (mirroring the lazy import in policies/factory.py).
from .configuration_lingbot_va import LingBotVAConfig
from .processor_lingbot_va import make_lingbot_va_pre_post_processors

__all__ = ["LingBotVAConfig", "LingBotVAPolicy", "make_lingbot_va_pre_post_processors"]


def __getattr__(name):
    if name == "LingBotVAPolicy":
        from .modeling_lingbot_va import LingBotVAPolicy

        return LingBotVAPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
