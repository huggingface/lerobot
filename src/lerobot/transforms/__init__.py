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

from .deprecated import (
    CoarseDropout,
    GammaCorrection,
    GaussianNoise,
    GaussianPatchBrightness,
    MotionBlur,
    PlanckianJitter,
    RandomShadow,
    RandomSubsetApply,
    SharpnessJitter,
    make_transform_from_config,
)
from .transforms import (
    BatchedCoarseDropout,
    BatchedColorJitter,
    BatchedGammaCorrection,
    BatchedGaussianNoise,
    BatchedGaussianPatchBrightness,
    BatchedIdentity,
    BatchedImageTransforms,
    BatchedMotionBlur,
    BatchedPlanckianJitter,
    BatchedRandomAffine,
    BatchedRandomRotation,
    BatchedRandomShadow,
    BatchedRandomSubsetApply,
    BatchedSharpnessJitter,
    BatchedTransform,
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    JPEGCompression,
    PerSampleTransform,
    make_batched_transform_from_config,
)

# An example of transforms effects can be found in: https://github.com/huggingface/lerobot/pull/4210

__all__ = [
    "BatchedCoarseDropout",
    "BatchedColorJitter",
    "BatchedGammaCorrection",
    "BatchedGaussianNoise",
    "BatchedGaussianPatchBrightness",
    "BatchedIdentity",
    "BatchedImageTransforms",
    "BatchedMotionBlur",
    "BatchedPlanckianJitter",
    "BatchedRandomAffine",
    "BatchedRandomRotation",
    "BatchedRandomShadow",
    "BatchedRandomSubsetApply",
    "BatchedSharpnessJitter",
    "BatchedTransform",
    "CoarseDropout",
    "GammaCorrection",
    "GaussianNoise",
    "GaussianPatchBrightness",
    "ImageTransformConfig",
    "ImageTransforms",
    "ImageTransformsConfig",
    "JPEGCompression",
    "MotionBlur",
    "PerSampleTransform",
    "PlanckianJitter",
    "RandomShadow",
    "RandomSubsetApply",
    "SharpnessJitter",
    "make_batched_transform_from_config",
    "make_transform_from_config",
]
