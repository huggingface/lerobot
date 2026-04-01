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
6D pose of a fiducial marker on the robot (e.g. wrist) from a single RGB image.

Mount a printed AprilTag / ArUco tag on the arm, measure its **physical edge length** in meters,
and use camera intrinsics + ``cv2.solvePnP`` to recover rotation and translation of the tag
frame in the camera frame. This is the standard lightweight approach for “arm tracking” without
training a neural net.

Requires ``opencv-contrib-python`` (``cv2.aruco``) — same as :class:`lerobot.perception.tag_tracker.TagTracker`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from lerobot.perception.tag_tracker import TagDetection, TagTracker


def intrinsics_dict_to_camera_matrix(intrinsics: dict) -> np.ndarray:
    """Build 3×3 K from ``fx, fy, cx, cy`` (depth or RGB intrinsics dict)."""
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def default_distortion_coeffs() -> np.ndarray:
    """No lens model (use when undistorted or unknown)."""
    return np.zeros((5, 1), dtype=np.float64)


def _rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return float(w / n), float(x / n), float(y / n), float(z / n)


def _marker_object_points_square(tag_size_m: float) -> np.ndarray:
    """OpenCV ArUco convention: origin at first corner, X along bottom edge, square in Z=0 plane."""
    L = float(tag_size_m)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [L, 0.0, 0.0],
            [L, L, 0.0],
            [0.0, L, 0.0],
        ],
        dtype=np.float32,
    )


@dataclass
class TagPose6D:
    """Pose of the marker coordinate frame expressed in the camera frame.

    ``P_cam = R @ P_marker + tvec`` with ``R = Rodrigues(rvec)``.
    ``tvec`` is the marker origin (first corner) position in camera coordinates (meters).
    """

    tag_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    reproj_error_px: float

    def to_jsonable(self) -> dict:
        w, x, y, z = _rotation_matrix_to_quaternion_wxyz(self.R)
        return {
            "tag_id": int(self.tag_id),
            "rvec": [float(v) for v in self.rvec.flatten()],
            "tvec_m": [float(v) for v in np.ravel(self.tvec)],
            "position_m": {
                "x": float(np.ravel(self.tvec)[0]),
                "y": float(np.ravel(self.tvec)[1]),
                "z": float(np.ravel(self.tvec)[2]),
            },
            "quaternion_wxyz": {"w": w, "x": x, "y": y, "z": z},
            "reproj_error_px": float(self.reproj_error_px),
        }


def estimate_tag_pose6d(
    detection: TagDetection,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> TagPose6D | None:
    """Estimate 6D pose of one detected square marker from its image corners.

    Args:
        detection: Corners from :meth:`TagTracker.detect` (must match OpenCV ArUco ordering).
        tag_size_m: Physical edge length of the **black square** in meters (not including white border).
        camera_matrix: 3×3 intrinsics matrix K.
        dist_coeffs: Optional 5×1 (or OpenCV-compatible) distortion coeffs; zeros if unknown.
    """
    if tag_size_m <= 0:
        return None
    dist = dist_coeffs if dist_coeffs is not None else default_distortion_coeffs()
    obj_pts = _marker_object_points_square(tag_size_m)
    img_pts = np.asarray(detection.corners_px, dtype=np.float32).reshape(4, 2)

    # SQPNP is stable for planar 4-point tags; avoid SOLVEPNP_IPPE_SQUARE here (can pick the wrong
    # solution relative to OpenCV's corner ordering). Fall back to iterative if unavailable.
    flags = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist, flags=flags)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok or rvec is None or tvec is None:
        return None

    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))

    return TagPose6D(
        tag_id=detection.tag_id,
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        R=R.astype(np.float64),
        reproj_error_px=err,
    )


def draw_pose_axes(
    rgb_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length_m: float = 0.05,
) -> None:
    """Draw RGB coordinate axes on a BGR image (mutates ``rgb_bgr`` in place)."""
    try:
        cv2.drawFrameAxes(rgb_bgr, camera_matrix, dist_coeffs, rvec, tvec, float(axis_length_m))
    except Exception:
        pass


class RobotArmFiducialTracker:
    """Detect a fiducial on the arm and estimate its 6D pose in the camera frame.

    Typical use: print a tag with ``lerobot-make-tag``, tape it to the wrist link, set
    ``tag_id`` to that ID and ``tag_size_m`` to the printed black-square size.
    """

    def __init__(
        self,
        tag_id: int,
        tag_size_m: float,
        dictionary: str = "apriltag_36h11",
    ):
        self.tag_id = int(tag_id)
        self.tag_size_m = float(tag_size_m)
        self._detector = TagTracker(dictionary=dictionary)

    def estimate_pose(
        self,
        rgb: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> TagPose6D | None:
        """Return pose of the configured ``tag_id``, or ``None`` if not visible."""
        for det in self._detector.detect(rgb):
            if det.tag_id != self.tag_id:
                continue
            return estimate_tag_pose6d(det, self.tag_size_m, camera_matrix, dist_coeffs)
        return None

    def estimate_all_poses(
        self,
        rgb: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
        tag_ids_filter: set[int] | None = None,
    ) -> list[TagPose6D]:
        """Pose for every detected tag (optionally restricted to ``tag_ids_filter``)."""
        out: list[TagPose6D] = []
        for det in self._detector.detect(rgb):
            if tag_ids_filter is not None and det.tag_id not in tag_ids_filter:
                continue
            pose = estimate_tag_pose6d(det, self.tag_size_m, camera_matrix, dist_coeffs)
            if pose is not None:
                out.append(pose)
        return out

    def poses_json(self, poses: list[TagPose6D]) -> str:
        return json.dumps({"poses": [p.to_jsonable() for p in poses]})
