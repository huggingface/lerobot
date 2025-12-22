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
Eagle3 model assets for Groot N1.6.

This package contains the Eagle-Block2A-2B-v2 backbone model components,
which replace the Eagle2-HG backbone used in N1.5.

Key components:
- configuration_eagle3_vl.py: Model configuration (Eagle3_VLConfig)
- modeling_eagle3_vl.py: Main VL model implementation (Eagle3_VLForConditionalGeneration)
- modeling_siglip2.py: SigLIP2 vision model with window attention and RoPE
- processing_eagle3_vl.py: Processor implementation (Eagle3_VLProcessor)
- image_processing_eagle3_vl_fast.py: Fast image processor (Eagle3_VLImageProcessorFast)
- eagle_backbone.py: EagleBackbone wrapper for action head integration
"""

from .configuration_eagle3_vl import Eagle3_VLConfig
from .eagle_backbone import EagleBackbone
from .image_processing_eagle3_vl_fast import Eagle3_VLImageProcessorFast
from .modeling_eagle3_vl import (
    Eagle3_VLForConditionalGeneration,
    Eagle3_VLPreTrainedModel,
)
from .modeling_siglip2 import (
    Siglip2VisionConfig,
    Siglip2VisionModel,
)
from .processing_eagle3_vl import Eagle3_VLProcessor

__all__ = [
    "Eagle3_VLConfig",
    "Eagle3_VLForConditionalGeneration",
    "Eagle3_VLPreTrainedModel",
    "Eagle3_VLProcessor",
    "Eagle3_VLImageProcessorFast",
    "Siglip2VisionConfig",
    "Siglip2VisionModel",
    "EagleBackbone",
]
