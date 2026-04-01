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
Capture RGB + depth from an Intel RealSense camera (e.g. D455) and build a 3D mesh.

Uses the existing RealSense support in LeRobot and Open3D for point cloud and
Poisson surface reconstruction. Install with: pip install "lerobot[realsense-mesh]"
(or pip install "lerobot[intelrealsense]" open3d).

Example:

```shell
# Use first detected RealSense, save mesh to mesh.ply
lerobot-realsense-mesh

# Specify camera serial and output path
lerobot-realsense-mesh --serial 123456789 --output scene.ply

# Limit depth range (meters) and mesh density
lerobot-realsense-mesh --depth-trunc 2.0 --depth 9
```
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.scripts.lerobot_find_cameras import find_all_realsense_cameras


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture from a RealSense camera and build a 3D mesh (point cloud + Poisson reconstruction).",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="RealSense serial number or name. If omitted, uses the first detected camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mesh.ply"),
        help="Output path for the mesh file (default: mesh.ply).",
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
        help="Poisson reconstruction depth (octree depth; higher = denser mesh, default: 9).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Only save the point cloud as PLY (no Poisson mesh).",
    )
    args = parser.parse_args()

    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        print(
            "Open3D is required for mesh reconstruction. Install with: pip install open3d",
            file=sys.stderr,
        )
        sys.exit(1)

    serial = args.serial
    if serial is None:
        realsense_cameras = find_all_realsense_cameras()
        if not realsense_cameras:
            print("No RealSense cameras found. Run: lerobot-find-cameras realsense", file=sys.stderr)
            sys.exit(1)
        serial = str(realsense_cameras[0]["id"])
        print(f"Using first detected RealSense: {serial}")

    config = RealSenseCameraConfig(
        serial_number_or_name=serial,
        use_depth=True,
        width=640,
        height=480,
        fps=30,
    )
    camera = RealSenseCamera(config)
    camera.connect()

    try:
        rgb = camera.read()
        depth_map = camera.read_depth()
        intrinsics = camera.get_depth_intrinsics()
    finally:
        camera.disconnect()

    # Build Open3D intrinsic; depth is in mm, so scale to meters for Open3D (z = d / depth_scale)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
    # RealSense depth_scale is meters per raw unit (0.001 for mm); Open3D wants raw/z so use 1/scale
    depth_scale_open3d = 1.0 / intrinsics["depth_scale"]
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth_map.astype("float32"))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=depth_scale_open3d,
        depth_trunc=args.depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic,
        depth_scale=depth_scale_open3d,
        depth_trunc=args.depth_trunc,
    )

    if args.no_mesh:
        o3d.io.write_point_cloud(str(args.output), pcd)
        print(f"Point cloud saved to {args.output}")
        return

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth
    )
    # Remove low-density vertices (often from outliers)
    densities_arr = np.asarray(densities)
    vertices_to_remove = densities_arr < np.quantile(densities_arr, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(str(args.output), mesh)
    print(f"Mesh saved to {args.output}")


if __name__ == "__main__":
    main()
