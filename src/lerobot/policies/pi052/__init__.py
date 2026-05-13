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

"""π0.5 v2 — full reproduction of the π0.5 paper's hierarchical
inference recipe on lerobot.

Extends :class:`lerobot.policies.pi05.PI05Policy` with:

* recipe-driven training (PR 1's :class:`RenderMessagesStep`),
* PaliGemma ``lm_head`` cross-entropy on supervised subtask spans
  (the "high-level subtask prediction" of the paper, §IV.D),
* AR text generation at inference (:meth:`PI052Policy.select_message`),
* per-component prompt dropout (Pi 0.7 §V.E) for regularising the
  text head against missing context at inference.

See ``src/lerobot/configs/recipes/hirobot.yaml`` for the
canonical training recipe and
``examples/training/pi052_hirobot.slurm`` for the launcher.
"""

from .configuration_pi052 import PI052Config
from .modeling_pi052 import PI052Policy
from .processor_pi052 import make_pi052_pre_post_processors
from .text_processor_pi052 import PI052TextTokenizerStep

__all__ = [
    "PI052Config",
    "PI052Policy",
    "PI052TextTokenizerStep",
    "make_pi052_pre_post_processors",
]
