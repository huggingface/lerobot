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
"""Pre- and post-processor pipelines for DP3."""

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.depth_processor import DepthToPointCloudStep

from .configuration_dp3 import DP3Config


def make_dp3_pre_post_processors(
    config: DP3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build DP3's processor pipelines: the default scaffold plus unprojection.

    Identical to Diffusion Policy's, except that a `DepthToPointCloudStep` is
    inserted before normalisation when `config.pointcloud_from_depth` is set.
    That placement is deliberate:

    * **after `to_device`**, so the unprojection and the fixed-size subsample run
      wherever the policy runs rather than pulling depth maps back to the CPU;
    * **before `normalize`**, because the step already returns coordinates
      normalised into the workspace cube, and `normalize` only touches features
      declared in the config, so it leaves `observation.pointcloud` alone.

    Without this, `--policy.type=dp3` resolves a policy class but no processor,
    and a user following the docs has to assemble the pipeline by hand from a
    depth stream, intrinsics and a crop they have to guess.

    Args:
        config: the DP3 configuration, which also carries the depth step's
            settings (frame, depth scale, workspace crop, calibration).
        dataset_stats: statistics used for normalisation.

    Returns:
        The pre-processor and post-processor pipelines.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[Any] = [steps.rename_observations, steps.add_batch_dim, steps.to_device]
    if config.pointcloud_from_depth:
        input_steps.append(
            DepthToPointCloudStep(
                num_points=config.pointcloud_num_points,
                frame=config.pointcloud_frame,
                with_colour=config.pointcloud_channels == 6,
                depth_scale=config.pointcloud_depth_scale,
                workspace_centre=config.pointcloud_workspace_centre,
                workspace_extent=config.pointcloud_workspace_extent,
                intrinsics=config.pointcloud_intrinsics,
                extrinsics=config.pointcloud_extrinsics,
            )
        )
    input_steps.append(steps.normalize)

    return make_policy_processor_pipelines(
        input_steps=input_steps,
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
