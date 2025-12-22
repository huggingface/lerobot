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
Configuration for Groot N1.6 policy.

Key differences from N1.5:
- Uses AlternateVLDiT instead of standard DiT
- 32 DiT layers (vs 16 in N1.5)
- Unfrozen top 4 VLM layers instead of 4-layer post-VLM adapter
- State-relative action chunks instead of absolute joint angles
- Uses Cosmos-Reason-2B variant backbone (Eagle-Block2A-2B-v2)
"""

# TODO: Implement GrootN16Config based on:
# - gr00t-orig/configs/model/gr00t_n1d6.py
# - src/lerobot/policies/groot/configuration_groot.py

raise NotImplementedError("GrootN16Config not yet implemented")

