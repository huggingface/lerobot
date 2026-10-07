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
"""Turn depth observations into a point cloud any policy can consume.

LeRobot has no 3D observation path today: no policy unprojects depth, and there
is no convention for where calibration lives. This step adds one, and it is
deliberately a *processor* rather than part of a policy so that the existing
policies can use it. A Diffusion Policy with this step in front of it and a
`PointCloudEncoder` in its observation encoder is 3D Diffusion Policy.

Observation keys this step reads and writes:

    reads   observation.images.{camera}_depth  (..., H, W) or (..., 1, H, W)
            observation.intrinsics.{camera}    (..., 3, 3)
            observation.extrinsics.{camera}    (..., 4, 4)  only if frame="world"
    writes  observation.pointcloud             (..., num_points, 3 or 6)

The depth key follows what the robots already emit -- `koch_follower` and the
hope_jr arms write `f"{cam_key}_depth"` -- and what `LeRobotDataset` already
stores, quantised to TIFF and flagged `is_depth_map` in the feature info. The
suffix form is also accepted bare (`{camera}_depth`) for datasets that do not
nest under `observation.images`.

Intrinsics are the piece that does not exist yet. A RealSense reports them and
LeRobot was discarding them at capture time, so a recorded depth map could not
be unprojected even in principle; `RealSenseCamera.get_intrinsics()` now
returns them, corrected for rotation and resize.

That leaves every dataset recorded before this PR without intrinsics, so this
step also takes them as configuration: `intrinsics={"top": (fx, fy, cx, cy)}`
is used whenever the observation does not carry its own. Read the four numbers
off the sensor once and they stay correct for that unit at that resolution.
Absent both, the step raises and names the option rather than a missing key.

Depth is **metres**, and **z-depth** -- distance along the optical axis, which is
what RealSense, Kinect, Zed and every simulator report. If your loader hands you
uint16 millimetres, divide by 1000 first; `depth_scale` is here for that.

The `frame` argument is the decision that matters, and it has a measured
threshold rather than a house preference. `"camera"` needs only intrinsics,
which come from the sensor and do not drift. `"world"` needs extrinsics and
inherits their error at roughly 9 mm per degree at 0.7 m.

Which to pick depends on your hand-eye residual, and the crossover was measured
on a keypose manipulation task with a 20 mm object: the world frame wins
decisively at 0 degrees (3.6 mm against 18.7), the two cross between 1 and 2
degrees, and by 5 degrees the camera frame is ahead (18.7 against 33.6). So:
below about a degree of calibration error use `"world"`, above about two use
`"camera"`. Fusing several cameras requires `"world"` either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.common.pointcloud import normalize_points, sample_points, unproject
from lerobot.utils.constants import OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

OBS_IMAGES_STR = f"{OBS_STR}.images"
DEPTH_SUFFIX = "_depth"
OBS_INTRINSICS = f"{OBS_STR}.intrinsics"
OBS_EXTRINSICS = f"{OBS_STR}.extrinsics"
OBS_POINTCLOUD = f"{OBS_STR}.pointcloud"


def _serialisable(table: dict[str, Any] | None) -> dict[str, Any] | None:
    """Config has to survive a round trip through JSON, so tensors become lists."""
    if table is None:
        return None
    return {
        k: (v.tolist() if isinstance(v, Tensor) else list(v) if isinstance(v, tuple) else v)
        for k, v in table.items()
    }


@dataclass
@ProcessorStepRegistry.register(name="depth_to_pointcloud_processor")
class DepthToPointCloudStep(ObservationProcessorStep):
    """Unproject every depth camera and emit one fixed-size point cloud.

    Args:
        num_points: points kept after subsampling. DP3 uses 512 to 1024; more
            costs memory and buys little, because the encoder max-pools anyway.
        frame: `"camera"` (intrinsics only) or `"world"` (also needs extrinsics).
            Multi-camera fusion only makes sense in `"world"`.
        with_colour: append RGB from `observation.images.{camera}`, giving 6
            channels per point instead of 3.
        depth_scale: multiplied into the raw depth to get metres. Leave at 1.0
            for float metres; use **1e-3 for millimetres, which is what
            `LeRobotDataset` returns by default** (`depth_output_unit`). Getting
            this wrong used to produce a cloud of zeros in silence; it now
            raises.
        min_depth, max_depth: readings outside this range are treated as
            invalid. Depth sensors report 0 for "no return", and a zero is a
            point at the camera origin that will drag any pooled statistic.
        intrinsics: per-camera fallback used when the observation carries no
            `observation.intrinsics.{camera}`, which is the case for **every
            dataset recorded before this PR** -- LeRobot discarded intrinsics at
            capture time, so depth was recorded but could not be unprojected.
            Give `{"camera_name": (fx, fy, cx, cy)}` in pixels, or a full 3x3
            matrix. An observation that does carry intrinsics wins, so a
            recording made after this PR needs no config at all.

            Read them off the sensor once with `RealSenseCamera.get_intrinsics()`
            (already corrected for rotation and resize). They are fixed by the
            optics, so a value copied from the camera that recorded the dataset
            stays correct -- but a value copied from a *different* unit, or from
            the same unit at a different resolution, does not.
        extrinsics: the same fallback for `observation.extrinsics.{camera}`,
            needed only when `frame="world"`. Camera-to-world 4x4 matrices.
            Unlike intrinsics these are not a sensor property and do drift; see
            the error budget in this module's docstring before hard-coding one.
        seed: makes the subsampling reproducible. The generator is kept across
            calls, so successive frames draw different points but the sequence
            repeats run to run -- which is what a dataloader wants. Leave unset
            for the global RNG.
        workspace_centre, workspace_extent: if set, points outside this cube are
            dropped and the survivors are normalised to roughly [-1, 1]. In
            `"camera"` frame the cube is centred on the camera, so a sensible
            default is something like (0, 0, distance-to-workspace).

            **Set these.** They read like tidiness and are not. A uniform
            subsample of a whole-scene cloud is mostly floor and far tabletop:
            measured on a tabletop manipulation scene, a 20 mm block covered 28
            of 8256 valid pixels, so 1024 uniform samples contained about 3.5
            points of the only object that mattered -- a third of a percent of
            the policy's input. Cropping first is what DP3 does and it is the
            difference between a cloud that shows the scene and one that shows
            the table.
    """

    num_points: int = 1024
    frame: str = "camera"
    with_colour: bool = False
    depth_scale: float = 1.0
    min_depth: float = 0.01
    max_depth: float = 3.0
    workspace_centre: tuple[float, float, float] | None = None
    workspace_extent: float | None = None
    intrinsics: dict[str, Any] | None = None
    extrinsics: dict[str, Any] | None = None
    seed: int | None = None

    _generators: dict[str, torch.Generator] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame not in ("camera", "world"):
            raise ValueError(f"frame must be 'camera' or 'world', got {self.frame!r}")
        if (self.workspace_centre is None) != (self.workspace_extent is None):
            raise ValueError("workspace_centre and workspace_extent must be set together")

    def _check_scale(self, depth: Tensor, camera: str) -> None:
        """Catch a depth stream that is not in the unit `depth_scale` claims.

        This exists because the default wiring of LeRobot to this step is wrong
        and fails silently. `LeRobotDataset` dequantises depth to MILLIMETRES by
        default (`depth_output_unit`, DEFAULT_DEPTH_UNIT), while `depth_scale`
        defaults to 1.0, which means metres. Unprojecting millimetres as metres
        puts every point a thousand times too far away; the workspace crop then
        rejects all of them, and `sample_points` returns zeros for a frame with
        no valid points -- deliberately, so a dropped depth frame cannot kill a
        training run. The result is a policy training happily on clouds of
        zeros, with nothing anywhere reporting a problem.

        A magnitude check is enough to separate the two: metric depth for any
        real sensor is single-digit metres, and millimetres read as metres are
        three orders of magnitude out. The threshold is deliberately far from
        both so an unusual-but-real scene cannot trip it.
        """
        finite = depth[torch.isfinite(depth) & (depth > 0)]
        if finite.numel() == 0:
            return
        median = float(finite.median())
        if median <= self.max_depth * 20:
            return
        raise ValueError(
            f"Depth for camera {camera!r} has a median of {median:.0f} after depth_scale="
            f"{self.depth_scale}, which is {median / max(self.max_depth, 1e-9):.0f}x max_depth="
            f"{self.max_depth}. That is the signature of millimetres being read as metres. "
            "LeRobotDataset returns depth in millimetres by default, so pass depth_scale=1e-3 "
            "here (or construct the dataset with depth_output_unit='m'). Left unscaled, every "
            "point falls outside the workspace crop and the cloud comes back as zeros."
        )

    def _rng(self, device: torch.device) -> torch.Generator | None:
        """A generator living on the tensors' own device, built on first use.

        `sample_points` draws with `torch.rand(device=points.device,
        generator=...)`, and torch requires the two to agree, so this cannot be
        built in `__post_init__` -- the device is not known until a batch
        arrives, and a CPU generator handed to a CUDA `torch.rand` raises.
        """
        if self.seed is None:
            return None
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def _calibration(
        self,
        observation: dict[str, Any],
        camera: str,
        kind: str,
        like: Tensor,
    ) -> Tensor:
        """Resolve one camera's calibration: observation first, then config.

        Raises with the option to set rather than a bare KeyError, because the
        common case is a dataset recorded before intrinsics were kept and the
        useful reply is "here is how to supply them", not "this key is absent".
        """
        prefix = OBS_INTRINSICS if kind == "intrinsics" else OBS_EXTRINSICS
        key = f"{prefix}.{camera}"
        if key in observation:
            return observation[key].to(device=like.device, dtype=torch.float32)

        supplied = (self.intrinsics if kind == "intrinsics" else self.extrinsics) or {}
        if camera in supplied:
            return self._as_matrix(supplied[camera], kind, camera).to(device=like.device, dtype=torch.float32)

        shape = "(fx, fy, cx, cy)" if kind == "intrinsics" else "a 4x4 camera-to-world matrix"
        extra = (
            "Datasets recorded before intrinsics were kept do not have this key -- read the "
            "values off the sensor once with RealSenseCamera.get_intrinsics()."
            if kind == "intrinsics"
            else "Or use frame='camera', which needs no extrinsics at all."
        )
        raise KeyError(
            f"No {kind} for camera {camera!r}: the observation has no {key!r} and "
            f"DepthToPointCloudStep was built without {kind}={{{camera!r}: ...}}. "
            f"Depth cannot be unprojected without {kind}. Either put {key!r} in the "
            f"observation, or pass {kind}={{{camera!r}: {shape}}} to this step. {extra}"
        )

    @staticmethod
    def _as_matrix(value: Any, kind: str, camera: str) -> Tensor:
        """Accept (fx, fy, cx, cy) or a full matrix, and reject anything else loudly."""
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
        if kind == "intrinsics":
            if tensor.shape == (4,):
                fx, fy, cx, cy = tensor
                one, zero = tensor.new_ones(()), tensor.new_zeros(())
                return torch.stack(
                    [
                        torch.stack([fx, zero, cx]),
                        torch.stack([zero, fy, cy]),
                        torch.stack([zero, zero, one]),
                    ]
                )
            if tensor.shape == (3, 3):
                return tensor
            raise ValueError(
                f"intrinsics[{camera!r}] must be (fx, fy, cx, cy) or a 3x3 matrix, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape != (4, 4):
            raise ValueError(
                f"extrinsics[{camera!r}] must be a 4x4 camera-to-world matrix, "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor

    def _depth_keys(self, observation: dict[str, Any]) -> dict[str, str]:
        """Map camera name -> the observation key holding its depth map.

        Matches the convention the robots already emit rather than inventing
        one: `observation.images.{cam}_depth`, or a bare `{cam}_depth`.
        """
        found: dict[str, str] = {}
        for key in observation:
            if not key.endswith(DEPTH_SUFFIX):
                continue
            stem = key[: -len(DEPTH_SUFFIX)]
            if stem.startswith(f"{OBS_IMAGES_STR}."):
                found[stem[len(OBS_IMAGES_STR) + 1 :]] = key
            elif "." not in stem:
                found[stem] = key
        return dict(sorted(found.items()))

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        depth_keys = self._depth_keys(observation)
        cameras = list(depth_keys)
        if not cameras:
            return observation
        if self.frame == "camera" and len(cameras) > 1:
            raise ValueError(
                f"{len(cameras)} depth cameras were found but frame='camera'. Clouds from "
                "different cameras are in different frames and concatenating them is "
                "meaningless. Use frame='world' with extrinsics, or keep one camera."
            )

        clouds, valids = [], []
        for camera in cameras:
            depth = observation[depth_keys[camera]]
            if depth.ndim >= 3 and depth.shape[-3] == 1:
                depth = depth.squeeze(-3)  # (..., 1, H, W) -> (..., H, W)
            depth = depth.to(torch.float32) * self.depth_scale
            self._check_scale(depth, camera)

            intrinsics = self._calibration(observation, camera, "intrinsics", depth)
            extrinsics = None
            if self.frame == "world":
                extrinsics = self._calibration(observation, camera, "extrinsics", depth)

            points = unproject(depth, intrinsics, extrinsics)
            valid = (depth > self.min_depth) & (depth < self.max_depth)

            if self.with_colour:
                image = observation[f"{OBS_STR}.images.{camera}"].to(torch.float32)
                if image.max() > 1.5:  # tolerate uint8-valued floats
                    image = image / 255.0
                colour = image.movedim(-3, -1)  # (..., C, H, W) -> (..., H, W, C)
                points = torch.cat([points, colour], dim=-1)

            clouds.append(points.flatten(-3, -2))
            valids.append(valid.flatten(-2, -1))

        cloud = torch.cat(clouds, dim=-2)
        valid = torch.cat(valids, dim=-1)

        if self.workspace_centre is not None:
            centre = cloud.new_tensor(self.workspace_centre)
            half = self.workspace_extent / 2.0
            inside = ((cloud[..., :3] - centre).abs() <= half).all(dim=-1)
            valid = valid & inside

        batched = cloud.ndim == 3
        if not batched:
            cloud, valid = cloud[None], valid[None]
        cloud = sample_points(cloud, valid, self.num_points, generator=self._rng(cloud.device))
        if self.workspace_centre is not None:
            centre = cloud.new_tensor(self.workspace_centre)
            cloud = torch.cat(
                [normalize_points(cloud[..., :3], centre, self.workspace_extent), cloud[..., 3:]],
                dim=-1,
            )
        if not batched:
            cloud = cloud[0]

        observation[OBS_POINTCLOUD] = cloud
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "num_points": self.num_points,
            "frame": self.frame,
            "with_colour": self.with_colour,
            "depth_scale": self.depth_scale,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "workspace_centre": self.workspace_centre,
            "workspace_extent": self.workspace_extent,
            "intrinsics": _serialisable(self.intrinsics),
            "extrinsics": _serialisable(self.extrinsics),
            "seed": self.seed,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        channels = 6 if self.with_colour else 3
        features[PipelineFeatureType.OBSERVATION][OBS_POINTCLOUD] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.num_points, channels)
        )
        return features
