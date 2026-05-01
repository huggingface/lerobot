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
"""Reusable adapter primitives for ONNX export of LeRobot policies.

Per-policy export modules under ``policies/<type>/export_<type>.py`` should
prefer composing these adapters over writing bespoke ``nn.Module`` wrappers.
The available patterns are:

- :class:`DictBatchAdapter` (with :class:`DictBatchSpec`) — for policies whose
  inference forward takes a single dict batch and returns a tensor (or a
  tuple/dict from which a tensor can be indexed).
- :class:`IterativeDenoisingAdapter` — abstract base class for policies that
  perform fixed-N-step iterative sampling (DDIM, flow matching, etc.).

For policies that fit neither pattern (e.g. Diffusion's single-step UNet
wrapper, or TDMPC's CEM planning loop), subclass ``nn.Module`` directly.
"""

from .dict_batch import DictBatchAdapter, DictBatchSpec
from .iterative import IterativeDenoisingAdapter

__all__ = [
    "DictBatchAdapter",
    "DictBatchSpec",
    "IterativeDenoisingAdapter",
]
