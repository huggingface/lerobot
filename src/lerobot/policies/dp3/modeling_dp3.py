#!/usr/bin/env python
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
"""DP3: 3D Diffusion Policy.

Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via
Simple 3D Representations*, RSS 2024.

DP3 is Diffusion Policy with a point cloud in the observation conditioning, so
that is exactly how it is built here: subclass the diffusion model, add a
point-cloud encoder, widen the global conditioning vector. Everything else --
the U-Net, the noise schedule, action chunking, the queues -- is inherited
untouched. Saying it as a subclass is more honest than copying eight hundred
lines and lets both policies benefit from fixes to the shared parts.

The cloud comes from `observation.pointcloud`, which
`lerobot.processor.depth_processor.DepthToPointCloudStep` builds from depth and
intrinsics. Cameras are optional: keep them for the RGB-D variant, drop them
for the point-cloud-only variant the paper mostly reports.
"""

from collections import deque

import einops
from torch import Tensor

from lerobot.policies.common.pointcloud import PointCloudEncoder
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.utils.constants import OBS_STR

OBS_POINTCLOUD = f"{OBS_STR}.pointcloud"


class DP3Model(DiffusionModel):
    """`DiffusionModel` with a point-cloud branch in the global conditioning."""

    def __init__(self, config: DP3Config):
        # The parent calls `_extra_global_cond_dim` while sizing its U-Net, so
        # this branch's width is reported straight from the config and the
        # encoder itself is built afterwards -- submodules cannot be assigned
        # before `nn.Module.__init__` runs, and widening the vector after the
        # U-Net exists would mean constructing it twice. A Diffusion Policy
        # U-Net is not small enough to build twice on a modest GPU for tidiness.
        super().__init__(config)
        self.pointcloud_encoder = PointCloudEncoder(
            in_channels=config.pointcloud_channels,
            hidden_sizes=tuple(config.pointcloud_hidden_sizes),
            out_features=config.pointcloud_feature_dim,
            use_layernorm=config.pointcloud_use_layernorm,
        )

    def _extra_global_cond_dim(self, config: DP3Config) -> int:
        return config.pointcloud_feature_dim

    def _has_conditioning_input(self, batch: dict[str, Tensor]) -> bool:
        """A point cloud is enough; cameras are optional for DP3.

        The parent requires images or an environment state, which would reject
        the point-cloud-only configuration the paper mostly reports.
        """
        return OBS_POINTCLOUD in batch or super()._has_conditioning_input(batch)

    def _extra_global_cond_feats(self, batch: dict[str, Tensor]) -> Tensor:
        """Pool each observation step's point cloud into one feature vector."""
        cloud = batch[OBS_POINTCLOUD]
        batch_size, n_obs_steps = cloud.shape[:2]
        # (B, S, N, C) -> (B*S, N, C): every observation step is encoded on its
        # own. Mixing them would leak a later observation into an earlier step's
        # conditioning, which looks like better training and fails at inference
        # where the queue is not yet full.
        flat = einops.rearrange(cloud, "b s n c -> (b s) n c")
        features = self.pointcloud_encoder(flat)
        return einops.rearrange(features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)


class DP3Policy(DiffusionPolicy):
    """3D Diffusion Policy: Diffusion Policy conditioned on a point cloud."""

    config_class = DP3Config
    name = "dp3"

    def _make_diffusion_model(self, config: DP3Config) -> DP3Model:
        return DP3Model(config)

    def reset(self):
        super().reset()
        if self._queues is not None:
            self._queues[OBS_POINTCLOUD] = deque(maxlen=self.config.n_obs_steps)
