#!/usr/bin/env python

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
Build a 3D mesh from two cameras (e.g. stereo pair or RealSense seen as OpenCV cameras).

Uses the normal camera API (OpenCV) to capture from two camera indices, computes
stereo disparity with OpenCV StereoSGBM, then point cloud and Poisson mesh with Open3D.
Works when RealSense SDK is unavailable (e.g. macOS) and the device appears as
two OpenCV cameras (e.g. left/right or color + another).

Install: pip install open3d  (lerobot core already has OpenCV)

Example:

```shell
# Use camera 0 (left) and 1 (right), save mesh to mesh.ply
lerobot-stereo-mesh --left 0 --right 1

# D455-like defaults: baseline 50mm, focal ~500 for 640 width
lerobot-stereo-mesh --left 0 --right 1 --output scene.ply
```
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def build_q_matrix(
    width: int,
    height: int,
    focal: float,
    baseline_m: float,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Build 4x4 Q matrix for reprojectImageTo3D (left camera principal point, baseline)."""
    if cx is None:
        cx = (width - 1) / 2.0
    if cy is None:
        cy = (height - 1) / 2.0
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[0, 0] = 1.0
    Q[0, 3] = -cx
    Q[1, 1] = 1.0
    Q[1, 3] = -cy
    Q[2, 3] = focal
    Q[3, 2] = -1.0 / baseline_m
    Q[3, 3] = 0.0
    return Q


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture from two OpenCV cameras (stereo) and build a 3D mesh.",
    )
    parser.add_argument(
        "--left",
        type=int,
        default=0,
        help="OpenCV camera index for left (or primary) camera (default: 0).",
    )
    parser.add_argument(
        "--right",
        type=int,
        default=1,
        help="OpenCV camera index for right camera (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mesh.ply"),
        help="Output path for the mesh file (default: mesh.ply).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width for both cameras (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height for both cameras (default: 480).",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=500.0,
        help="Approximate focal length in pixels (default: 500, typical for 640 width).",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.05,
        help="Stereo baseline in meters (default: 0.05 for D455-like).",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Maximum depth in meters for points (default: 3.0).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=9,
        help="Poisson reconstruction depth (default: 9).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Only save the point cloud as PLY (no Poisson mesh).",
    )
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is required. Install with: pip install open3d", file=sys.stderr)
        sys.exit(1)

    w, h = args.width, args.height
    config_l = OpenCVCameraConfig(
        index_or_path=args.left,
        fps=30,
        width=w,
        height=h,
        color_mode=ColorMode.RGB,
    )
    config_r = OpenCVCameraConfig(
        index_or_path=args.right,
        fps=30,
        width=w,
        height=h,
        color_mode=ColorMode.RGB,
    )

    cam_left = OpenCVCamera(config_l)
    cam_right = OpenCVCamera(config_r)
    cam_left.connect()
    cam_right.connect()

    try:
        img_left = cam_left.read()
        img_right = cam_right.read()
    finally:
        cam_left.disconnect()
        cam_right.disconnect()

    # Ensure same size
    if img_left.shape[:2] != img_right.shape[:2]:
        img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))
    if img_left.shape[:2] != (h, w):
        img_left = cv2.resize(img_left, (w, h))
        img_right = cv2.resize(img_right, (w, h))

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

    # Stereo matching (StereoSGBM returns 16-bit fixed-point disparity, scale 16)
    min_disp = 0
    num_disp = 16 * 10  # divisible by 16
    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(gray_left, gray_right)
    # OpenCV disparity is 16-bit fixed point; convert to float disparity (pixels)
    disp_float = np.float32(disp) / 16.0
    disp_float[disp_float <= min_disp] = np.nan

    # Q matrix and 3D reprojection
    Q = build_q_matrix(w, h, args.focal, args.baseline)
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)
    # points_3d is (H,W,3); units from Q (baseline in m -> meters)

    # Mask invalid and too far
    valid = np.isfinite(points_3d).all(axis=2)
    z = points_3d[:, :, 2]
    valid &= (z > 0) & (z < args.depth_trunc)

    xyz = points_3d[valid].astype(np.float64)
    rgb = img_left[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if args.no_mesh:
        o3d.io.write_point_cloud(str(args.output), pcd)
        print(f"Point cloud saved to {args.output}")
        return

    # Poisson reconstruction requires normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth
    )
    densities_arr = np.asarray(densities)
    vertices_to_remove = densities_arr < np.quantile(densities_arr, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(str(args.output), mesh)
    print(f"Mesh saved to {args.output}")


if __name__ == "__main__":
    main()
