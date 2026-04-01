from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TagDetection:
    tag_id: int
    corners_px: np.ndarray  # shape (4, 2) float32 in pixel coords

    @property
    def center_px(self) -> tuple[float, float]:
        c = self.corners_px.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        xs = self.corners_px[:, 0]
        ys = self.corners_px[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def to_jsonable(self) -> dict:
        return {
            "tag_id": int(self.tag_id),
            "corners_px": self.corners_px.round(2).tolist(),
            "center_px": [round(self.center_px[0], 2), round(self.center_px[1], 2)],
            "bbox_xyxy": list(map(int, self.bbox_xyxy)),
        }


def pixel_to_camera_xyz(
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: dict,
) -> np.ndarray:
    """Backproject a pixel + depth into camera XYZ (meters).

    Assumes pinhole intrinsics with fx, fy, cx, cy.
    Camera frame convention matches the depth camera alignment (OpenCV: X right, Y down, Z forward).
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    z = float(depth_m)
    x = (float(u_px) - cx) * z / fx
    y = (float(v_px) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def depth_at_pixel_m(
    depth_mm: np.ndarray,
    u_px: float,
    v_px: float,
    intrinsics: dict,
    window: int = 5,
) -> float:
    """Robust depth lookup around (u,v) using median in a small window.

    depth_mm is uint16 (millimeters) for OAK-D aligned depth.
    """
    h, w = depth_mm.shape[:2]
    u = int(round(u_px))
    v = int(round(v_px))
    half = max(0, int(window) // 2)
    x1 = max(0, u - half)
    x2 = min(w, u + half + 1)
    y1 = max(0, v - half)
    y2 = min(h, v + half + 1)

    patch = depth_mm[y1:y2, x1:x2].astype(np.float32)
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0.0

    depth_scale = float(intrinsics.get("depth_scale", 0.001))
    return float(np.median(patch) * depth_scale)


class TagTracker:
    """AprilTag tracking via OpenCV ArUco dictionaries.

    Requires an OpenCV build with the aruco module (opencv-contrib-python).
    """

    def __init__(self, dictionary: str = "apriltag_36h11"):
        if not hasattr(cv2, "aruco"):
            raise ImportError(
                "OpenCV ArUco module not found. Install opencv-contrib-python to use tag tracking."
            )

        dict_map = {
            "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "aruco_4x4_50": cv2.aruco.DICT_4X4_50,
            "aruco_4x4_100": cv2.aruco.DICT_4X4_100,
        }
        if dictionary not in dict_map:
            raise ValueError(f"Unknown dictionary '{dictionary}'. Options: {sorted(dict_map.keys())}")

        self.dictionary_name = dictionary
        self._dict = cv2.aruco.getPredefinedDictionary(dict_map[dictionary])
        self._params = cv2.aruco.DetectorParameters()
        # Slightly more robust defaults for typical indoor robotics lighting.
        try:
            self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        try:
            self._params.adaptiveThreshWinSizeMin = 3
            self._params.adaptiveThreshWinSizeMax = 53
            self._params.adaptiveThreshWinSizeStep = 4
        except Exception:
            pass
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def detect(self, rgb: np.ndarray) -> list[TagDetection]:
        """Detect tags in an RGB image.

        Args:
            rgb: (H,W,3) uint8 RGB image.
        """
        # Detect on grayscale for robustness; accept either RGB or BGR inputs.
        gray = None
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception:
            try:
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = rgb if rgb.ndim == 2 else None
        if gray is None:
            return []

        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return []

        detections: list[TagDetection] = []
        for c, i in zip(corners, ids.flatten(), strict=False):
            # OpenCV returns corners as (1, 4, 2)
            cc = np.asarray(c, dtype=np.float32).reshape(4, 2)
            detections.append(TagDetection(tag_id=int(i), corners_px=cc))
        return detections

    def detections_json(self, detections: list[TagDetection]) -> str:
        return json.dumps(
            {
                "dictionary": self.dictionary_name,
                "detections": [d.to_jsonable() for d in detections],
            }
        )

