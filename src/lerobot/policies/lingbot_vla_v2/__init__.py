# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

# Importing the processor module here is what registers the LingBot
# `ProcessorStep`s in the registry on `import lerobot.policies` — the same path
# every other policy takes (the outer `policies/__init__.py` stays config-only,
# which keeps `import lerobot` free of the heavy optional deps: transformers,
# qwen-vl-utils etc. arrive via `pip install 'lerobot[lingbot_vla2]'`).
from .configuration_lingbot_vla_v2 import LingbotVLAV2Config
from .modeling_lingbot_vla_v2 import LingbotVLAV2Policy
from .processor_lingbot_vla_v2 import make_lingbot_vla_v2_pre_post_processors

__all__ = [
    "LingbotVLAV2Config",
    "LingbotVLAV2Policy",
    "make_lingbot_vla_v2_pre_post_processors",
]
