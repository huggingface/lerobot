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
"""Configuration for DP3, 3D Diffusion Policy."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("dp3")
@dataclass
class DP3Config(DiffusionConfig):
    """3D Diffusion Policy (Ze et al., RSS 2024).

    DP3 is Diffusion Policy with a point cloud added to the observation
    conditioning, so this config is `DiffusionConfig` plus the handful of fields
    that describe the cloud. Everything about the diffusion process -- horizon,
    scheduler, U-Net, action chunking -- is inherited unchanged, which is both
    faithful to the paper and the reason this policy is a few hundred lines
    rather than a parallel stack.

    The point cloud arrives as `observation.pointcloud`, produced by
    `lerobot.processor.depth_processor.DepthToPointCloudStep` from
    `observation.images.{cam}_depth` and the camera intrinsics. That step is
    part of this policy's own pre-processor pipeline by default, so
    `--policy.type=dp3` works on a depth dataset without assembling a pipeline
    by hand; set `pointcloud_from_depth=False` if the cloud is already in the
    dataset. Cameras may be kept as
    well: with `image_features` non-empty the policy conditions on both, which
    is the RGB-D variant the paper reports; with no cameras it is the
    point-cloud-only variant.

    Args:
        pointcloud_num_points: points per cloud. Must match the processor step.
        pointcloud_channels: 3 for XYZ, 6 if the processor was configured with
            `with_colour=True`.
        pointcloud_feature_dim: width of the pooled point-cloud embedding that
            is concatenated into the diffusion model's global conditioning.
        pointcloud_hidden_sizes: per-point MLP widths. The paper's finding is
            that a *small* encoder beats heavier point-cloud backbones on
            manipulation, so enlarging this is reproducing something else.
        pointcloud_use_layernorm: LayerNorm inside the encoder. On by default,
            and it matters more than usual here because point coordinates are
            metres and are not otherwise normalised.
        pointcloud_from_depth: put a `DepthToPointCloudStep` in the
            pre-processor pipeline. The fields below configure it, and they are
            ignored when this is False.
        pointcloud_frame: `"camera"` (intrinsics only) or `"world"` (also needs
            extrinsics; the only way to fuse several cameras).
        pointcloud_depth_scale: multiplied into the stored depth to get metres.
            The default is **1e-3, not 1.0**, because `LeRobotDataset`
            dequantises depth to millimetres by default (`depth_output_unit`).
            Use 1.0 for a source that already reports metres.
        pointcloud_workspace_centre, pointcloud_workspace_extent: crop cube, in
            the chosen frame. Strongly recommended -- see the step's docstring
            for how much of an uncropped cloud is floor.
        pointcloud_intrinsics, pointcloud_extrinsics: per-camera calibration for
            datasets recorded before intrinsics were stored, as
            `{"camera": (fx, fy, cx, cy)}` and 4x4 matrices respectively.
            Values present in the observation take precedence.
    """

    pointcloud_num_points: int = 1024
    pointcloud_channels: int = 3
    pointcloud_feature_dim: int = 256
    pointcloud_hidden_sizes: tuple[int, ...] = (64, 128, 256)
    pointcloud_use_layernorm: bool = True

    pointcloud_from_depth: bool = True
    pointcloud_frame: str = "camera"
    pointcloud_depth_scale: float = 1e-3
    pointcloud_workspace_centre: tuple[float, float, float] | None = None
    pointcloud_workspace_extent: float | None = None
    pointcloud_intrinsics: dict[str, Any] | None = field(default=None)
    pointcloud_extrinsics: dict[str, Any] | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.pointcloud_channels not in (3, 6):
            raise ValueError(
                f"pointcloud_channels must be 3 (XYZ) or 6 (XYZ+RGB), got {self.pointcloud_channels}"
            )
        if self.pointcloud_num_points < 1:
            raise ValueError("pointcloud_num_points must be positive")
        if self.pointcloud_frame not in ("camera", "world"):
            raise ValueError(f"pointcloud_frame must be 'camera' or 'world', got {self.pointcloud_frame!r}")
        if (self.pointcloud_workspace_centre is None) != (self.pointcloud_workspace_extent is None):
            raise ValueError(
                "pointcloud_workspace_centre and pointcloud_workspace_extent must be set together"
            )

    def validate_features(self) -> None:
        # Deliberately NOT calling DiffusionConfig.validate_features: that one
        # requires at least one camera, and a point-cloud-only DP3 is a
        # legitimate and common configuration.
        if not self.robot_state_feature:
            raise ValueError(f"{self.__class__.__name__} requires `observation.state`.")
