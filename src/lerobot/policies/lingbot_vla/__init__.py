#!/usr/bin/env python

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

from .configuration_lingbot_vla import LingbotVLAConfig

# NOTE: LingbotVLAPolicy is intentionally NOT imported here. It pulls in heavy
# optional deps (transformers Qwen2.5-VL, etc.) and is loaded lazily by the policy
# factory only when policy.type == "lingbot_vla".
__all__ = ["LingbotVLAConfig"]
