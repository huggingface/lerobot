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

"""Backward compatibility module - re-exports from koch_utils.

This module is deprecated. Use koch_utils directly instead.
"""

# Re-export everything from koch_utils for backward compatibility
from lerobot.async_inference.koch_utils import (
    INITIAL_EE_POSE_BIMANUAL as INITIAL_EE_POSE,
    action_dict_to_tensor,
    action_tensor_to_dict,
    compute_current_ee,
    generate_linear_trajectory,
    get_action_features as get_bimanual_action_features,
)

__all__ = [
    "INITIAL_EE_POSE",
    "action_tensor_to_dict",
    "action_dict_to_tensor",
    "generate_linear_trajectory",
    "get_bimanual_action_features",
    "compute_current_ee",
]
