#!/usr/bin/env python

# Copyright 2025 DexVLA Team and The HuggingFace Inc. team. All rights reserved.
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

import torch.nn as nn


class ActionProjector(nn.Module):
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.mlps = nn.ModuleList(
            [
                # nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.0),
            ]
        )

    def forward(self, x):
        x = self.global_1d_pool(x.permute(1, 0)).permute(1, 0)
        for mlp in self.mlps:
            x = mlp(x)
        return x


class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.scale_fc = nn.Linear(condition_dim, feature_dim)
        self.shift_fc = nn.Linear(condition_dim, feature_dim)

        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.weight)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, x, condition):
        # calculate scale and shift
        scale = self.scale_fc(condition)
        shift = self.shift_fc(condition)

        # film
        return x * (1 + scale) + shift
