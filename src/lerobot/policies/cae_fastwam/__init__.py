# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""CAE-FastWAM policy: FastWAM with a Compact Action Expert and a suite-conditioned action VAE."""

from .configuration_cae_fastwam import CAEFastWAMConfig
from .modeling_cae_fastwam import CAEFastWAMPolicy
from .processor_cae_fastwam import make_cae_fastwam_pre_post_processors

__all__ = [
    "CAEFastWAMConfig",
    "CAEFastWAMPolicy",
    "make_cae_fastwam_pre_post_processors",
]
