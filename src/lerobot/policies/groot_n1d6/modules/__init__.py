#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
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
Modules for Groot N1.6 policy.

This package contains:
- dit.py: DiT and AlternateVLDiT transformer blocks
- embodiment_mlp.py: CategorySpecificMLP and MultiEmbodimentActionEncoder
"""

from lerobot.policies.gr00t_n1d6.modules.dit import (
    AdaLayerNorm,
    AlternateVLDiT,
    BasicTransformerBlock,
    DiT,
    SelfAttentionTransformer,
    TimestepEncoder,
)
from lerobot.policies.gr00t_n1d6.modules.embodiment_mlp import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    SmallMLP,
    swish,
)

__all__ = [
    # DiT modules
    "TimestepEncoder",
    "AdaLayerNorm",
    "BasicTransformerBlock",
    "DiT",
    "AlternateVLDiT",
    "SelfAttentionTransformer",
    # Embodiment MLP modules
    "swish",
    "SinusoidalPositionalEncoding",
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "SmallMLP",
    "MultiEmbodimentActionEncoder",
]
