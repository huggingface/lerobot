# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Observation processor step that extracts 3D object state from RGB + depth.

Given a camera's color image and aligned depth map, this step:
1. Segments objects of interest (via HSV color thresholds, or an external mask).
2. Backprojects the masked depth pixels to 3D using camera intrinsics.
3. Computes per-object: center_xyz, size_xyz, distance.

The outputs are appended to the observation as additional state keys so that
downstream policies can be conditioned on structured 3D object information
instead of learning geometry from raw pixels.

The standalone ``compute_object_state`` function can also be called directly
outside the processor pipeline (e.g. from the autonomous manipulation script).
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def fuse_depth_temporal_median(depths: list[np.ndarray]) -> np.ndarray:
    """Fuse a sequence of aligned depth frames with a per-pixel median (invalid = 0 or NaN).

    Reduces stereo speckle when the scene and camera are mostly static.

    Args:
        depths: Each array is (H, W) uint16 depth in mm (0 = invalid).

    Returns:
        Same shape/dtype as the first frame; 0 where all inputs invalid.
    """
    if not depths:
        raise ValueError("fuse_depth_temporal_median: empty depth list")
    if len(depths) == 1:
        return np.asarray(depths[0], dtype=np.uint16)

    stack = np.stack([np.asarray(d, dtype=np.float32) for d in depths], axis=0)
    stack[stack <= 0] = np.nan
    fused = np.nanmedian(stack, axis=0)
    out = np.nan_to_num(fused, nan=0.0).astype(np.uint16)
    return out

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.core import RobotObservation

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


def _backproject_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
) -> np.ndarray:
    """Backproject masked depth pixels to 3D points (N, 3) in camera frame."""
    vs, us = np.where(mask)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z_raw = depth[vs, us, 0].astype(np.float32) if depth.ndim == 3 else depth[vs, us].astype(np.float32)

    z = z_raw * depth_scale
    valid = z > 0
    z, us, vs = z[valid], us[valid], vs[valid]

    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (us.astype(np.float32) - cx) * z / fx
    y = (vs.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _robust_point_cloud_extent(
    pts: np.ndarray,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> np.ndarray:
    """Axis-aligned size from (high - low) percentiles per axis.

    Min–max extent is dominated by stereo flying pixels and mixed-depth masks; percentiles
    yield plausible object sizes for manipulation summaries and grasp planning.
    """
    n = pts.shape[0]
    if n < 8:
        return (pts.max(axis=0) - pts.min(axis=0)).astype(np.float32)
    lo = np.percentile(pts, low_pct, axis=0)
    hi = np.percentile(pts, high_pct, axis=0)
    return np.maximum((hi - lo).astype(np.float32), 1e-4)


def _segment_by_hsv(
    rgb: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area: int = 100,
) -> np.ndarray:
    """Segment objects by HSV range and return a binary mask of the largest blob."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    result = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(result, [largest], -1, 255, cv2.FILLED)
    return result


def compute_object_state(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float],
) -> dict[str, np.ndarray]:
    """Compute 3D object state from a depth map and a binary mask.

    This is a standalone function usable outside the processor pipeline,
    e.g. from the autonomous manipulation script.

    Args:
        depth: (H, W) or (H, W, 1) depth map (uint16, millimeters).
        mask: (H, W) binary mask (255 = object).
        intrinsics: Dict with ``fx, fy, cx, cy, depth_scale``.

    Returns:
        Dict with ``obj_center_xyz`` (3,), ``obj_size_xyz`` (3,),
        ``obj_distance`` (1,) as float32 arrays. All zeros if no
        valid points are found. ``obj_size_xyz`` is a robust extent
        (95th–5th percentile per axis) in meters, not raw min–max.
    """
    zeros = {
        "obj_center_xyz": np.zeros(3, dtype=np.float32),
        "obj_size_xyz": np.zeros(3, dtype=np.float32),
        "obj_distance": np.zeros(1, dtype=np.float32),
    }

    fx = float(intrinsics.get("fx", 500.0))
    fy = float(intrinsics.get("fy", 500.0))
    cx = float(intrinsics.get("cx", 320.0))
    cy = float(intrinsics.get("cy", 240.0))
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    min_z = float(intrinsics.get("min_depth_m", 0.04))
    max_z = float(intrinsics.get("max_depth_m", 5.0))

    pts = _backproject_depth(np.asarray(depth), np.asarray(mask), fx, fy, cx, cy, depth_scale)
    if pts.shape[0] < 3:
        return zeros

    z = pts[:, 2]
    keep = (z >= min_z) & (z <= max_z)
    pts = pts[keep]
    if pts.shape[0] < 3:
        return zeros

    # Drop flying-pixel tails along the ray (stereo / bad segmentation).
    z2 = pts[:, 2]
    lo_p, hi_p = np.percentile(z2, [7.0, 93.0])
    if hi_p > lo_p + 1e-4:
        m = (z2 >= lo_p) & (z2 <= hi_p)
        if int(np.count_nonzero(m)) >= 5:
            pts = pts[m]

    center = np.median(pts, axis=0).astype(np.float32)
    extent = _robust_point_cloud_extent(pts)
    distance = np.linalg.norm(center).astype(np.float32).reshape(1)
    return {
        "obj_center_xyz": center,
        "obj_size_xyz": extent,
        "obj_distance": distance,
    }


def compute_object_state_from_bbox_roi(
    depth: np.ndarray,
    bbox_xyxy: tuple[int | float, int | float, int | float, int | float],
    intrinsics: dict[str, float],
    *,
    shrink: int = 3,
) -> dict[str, np.ndarray]:
    """3D state from median depth inside the VLM bbox (ignores mask; resists SAM outliers).

    Uses a slightly shrunken ROI and percentile filtering on depth values.
    """
    zeros = {
        "obj_center_xyz": np.zeros(3, dtype=np.float32),
        "obj_size_xyz": np.zeros(3, dtype=np.float32),
        "obj_distance": np.zeros(1, dtype=np.float32),
    }

    fx = float(intrinsics.get("fx", 500.0))
    fy = float(intrinsics.get("fy", 500.0))
    cx = float(intrinsics.get("cx", 320.0))
    cy = float(intrinsics.get("cy", 240.0))
    depth_scale = float(intrinsics.get("depth_scale", 0.001))
    min_z = float(intrinsics.get("min_depth_m", 0.04))
    max_z = float(intrinsics.get("max_depth_m", 5.0))

    # ``xyxy`` matches OpenCV / this repo: x2,y2 are exclusive slice endpoints.
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    h, w = int(depth.shape[0]), int(depth.shape[1])
    x1 = max(0, min(w - 1, x1 + shrink))
    y1 = max(0, min(h - 1, y1 + shrink))
    x2 = max(0, min(w, x2 - shrink))
    y2 = max(0, min(h, y2 - shrink))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return zeros

    roi = np.asarray(depth)[y1:y2, x1:x2]
    if roi.size == 0:
        return zeros
    z_raw = roi.astype(np.float32).reshape(-1)
    z_m = z_raw * depth_scale
    valid = (z_raw > 0) & (z_m >= min_z) & (z_m <= max_z)
    z_m = z_m[valid]
    if z_m.size < 3:
        return zeros

    lo_p, hi_p = np.percentile(z_m, [12.0, 88.0])
    if hi_p > lo_p + 1e-5:
        z_m = z_m[(z_m >= lo_p) & (z_m <= hi_p)]
    if z_m.size < 3:
        return zeros

    z_med = float(np.median(z_m))
    u = 0.5 * (x1 + x2 - 1.0)
    v = 0.5 * (y1 + y2 - 1.0)
    x_cam = (u - cx) * z_med / fx
    y_cam = (v - cy) * z_med / fy
    center = np.array([x_cam, y_cam, z_med], dtype=np.float32)

    sz_x = max(float(x2 - x1) * z_med / fx, 5e-3)
    sz_y = max(float(y2 - y1) * z_med / fy, 5e-3)
    sz_z = max(float(np.percentile(z_m, 75) - np.percentile(z_m, 25)), 5e-3)
    extent = np.array([sz_x, sz_y, sz_z], dtype=np.float32)
    distance = np.linalg.norm(center).astype(np.float32).reshape(1)
    return {
        "obj_center_xyz": center,
        "obj_size_xyz": extent,
        "obj_distance": distance,
    }


@dataclass
@ProcessorStepRegistry.register(name="depth_perception")
class DepthPerceptionProcessorStep(ObservationProcessorStep):
    """Extract 3D object state from an RGB + depth observation pair.

    This step looks for ``camera_key`` (RGB) and ``{camera_key}_depth``
    (aligned depth, uint16 mm) in the observation dict.  It segments
    objects of interest, backprojects to 3D, and appends:

    * ``obj_center_xyz`` -- (3,) float32, median 3D center in camera frame (meters)
    * ``obj_size_xyz``   -- (3,) float32, robust axis extent in meters (95th–5th pct.; not min–max)
    * ``obj_distance``   -- (1,) float32, Euclidean distance from camera origin (meters)

    If no object is detected the values are zeros.

    Segmentation can come from:
    - HSV color thresholds (default, when no external mask is provided)
    - An external mask stored in ``observation["{camera_key}_mask"]`` (e.g.
      from a VLM detector).  When present, HSV segmentation is skipped.

    Parameters:
        camera_key: name of the RGB camera observation key (default ``"front"``).
        intrinsics: dict with ``fx, fy, cx, cy, depth_scale`` for the depth camera.
            If empty, a neutral pinhole model is assumed (will be inaccurate --
            call ``camera.get_depth_intrinsics()`` and pass the result here).
        hsv_lower: lower HSV bound for color segmentation (H 0-179, S/V 0-255).
        hsv_upper: upper HSV bound for color segmentation.
        min_blob_area: minimum contour area in pixels to count as an object.
    """

    camera_key: str = "front"
    intrinsics: dict[str, float] = field(default_factory=dict)
    hsv_lower: tuple[int, int, int] = (0, 120, 70)
    hsv_upper: tuple[int, int, int] = (10, 255, 255)
    min_blob_area: int = 100

    def _get_intrinsic(self, key: str, default: float) -> float:
        return float(self.intrinsics.get(key, default))

    def observation(self, observation: RobotObservation) -> RobotObservation:
        rgb_key = self.camera_key
        depth_key = f"{self.camera_key}_depth"
        mask_key = f"{self.camera_key}_mask"

        zeros_center = np.zeros(3, dtype=np.float32)
        zeros_size = np.zeros(3, dtype=np.float32)
        zeros_dist = np.zeros(1, dtype=np.float32)

        rgb = observation.get(rgb_key)
        depth = observation.get(depth_key)

        if rgb is None or depth is None:
            observation["obj_center_xyz"] = zeros_center
            observation["obj_size_xyz"] = zeros_size
            observation["obj_distance"] = zeros_dist
            return observation

        rgb_np = np.asarray(rgb)
        depth_np = np.asarray(depth)

        external_mask = observation.get(mask_key)
        if external_mask is not None:
            mask = np.asarray(external_mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = _segment_by_hsv(rgb_np, self.hsv_lower, self.hsv_upper, self.min_blob_area)

        intr = {
            "fx": self._get_intrinsic("fx", rgb_np.shape[1] * 0.8),
            "fy": self._get_intrinsic("fy", rgb_np.shape[0] * 0.8),
            "cx": self._get_intrinsic("cx", rgb_np.shape[1] / 2.0),
            "cy": self._get_intrinsic("cy", rgb_np.shape[0] / 2.0),
            "depth_scale": self._get_intrinsic("depth_scale", 0.001),
        }

        result = compute_object_state(depth_np, mask, intr)
        observation["obj_center_xyz"] = result["obj_center_xyz"]
        observation["obj_size_xyz"] = result["obj_size_xyz"]
        observation["obj_distance"] = result["obj_distance"]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_key": self.camera_key,
            "intrinsics": self.intrinsics,
            "hsv_lower": list(self.hsv_lower),
            "hsv_upper": list(self.hsv_upper),
            "min_blob_area": self.min_blob_area,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        obs = features.get(PipelineFeatureType.OBSERVATION, {})
        obs["obj_center_xyz"] = PolicyFeature(type=FeatureType.STATE, shape=(3,))
        obs["obj_size_xyz"] = PolicyFeature(type=FeatureType.STATE, shape=(3,))
        obs["obj_distance"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        features[PipelineFeatureType.OBSERVATION] = obs
        return features
