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
Zero-shot autonomous manipulation: see an object, understand it, grasp it.

Combines:
- OAK-D (or any depth camera) for RGB + depth
- VLM (Florence-2 local or cloud API) for object detection
- Depth backprojection for 3D object state
- Heuristic grasp planner for waypoint generation
- Placo IK solver for joint-space execution on SO-100

Example::

    lerobot-auto-manipulate \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \\
        --task="pick up the red cube" \\
        --urdf=./SO101/so101_new_calib.urdf \\
        --vlm_backend=local \\
        --camera_to_robot_tf="0,0,0.4,0,0,0"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import json

import cv2
import numpy as np

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.utils.motion_executor import MotionExecutionConfig, execute_waypoints
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

SO100_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class VLMConfig:
    backend: str = "gemini"
    model_id: str = "microsoft/Florence-2-base"
    # Keep as strings for CLI/Draccus compatibility (empty string means "unset").
    device: str = ""
    api_key: str = ""
    cloud_model: str = "gpt-4o"
    cloud_base_url: str = ""


@dataclass
class AutoManipulateConfig:
    robot: RobotConfig = field(default_factory=lambda: RobotConfig(type="so100_follower"))
    vlm: VLMConfig = field(default_factory=VLMConfig)

    task: str = "pick up the red cube"
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    camera_to_robot_tf: str = "0,0,0.4,0,0,0"
    # Optional "x,y,z" (meters) place location; empty string means disabled.
    place_target: str = ""
    max_attempts: int = 3
    pre_grasp_height: float = 0.06
    lift_height: float = 0.08


def parse_tf_string(tf_str: str) -> np.ndarray:
    """Parse 'x,y,z,rx,ry,rz' into a 4x4 homogeneous transform.

    rx, ry, rz are rotation vector components (radians).
    """
    from scipy.spatial.transform import Rotation

    parts = [float(v.strip()) for v in tf_str.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(parts)}: {tf_str}")

    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = parts[:3]
    if any(abs(v) > 1e-8 for v in parts[3:6]):
        tf[:3, :3] = Rotation.from_rotvec(parts[3:6]).as_matrix()
    return tf


def colorize_depth(depth_mm: np.ndarray, max_range_mm: int = 2000) -> np.ndarray:
    """Convert uint16 depth (mm) to a colorized BGR image for display."""
    depth = np.asarray(depth_mm)
    clipped = np.clip(depth.astype(np.float32), 0, max_range_mm)
    normalized = (clipped / max_range_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


@parser.wrap()
def auto_manipulate(cfg: AutoManipulateConfig):
    """Main autonomous manipulation loop."""
    init_logging()

    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.grasp_planner import GraspPlanner
    from lerobot.perception.vlm_detector import VLMDetector
    from lerobot.processor.depth_perception_processor import compute_object_state

    cam_to_robot = parse_tf_string(cfg.camera_to_robot_tf)

    logger.info("Initializing VLM detector...")
    detector = VLMDetector(
        backend=cfg.vlm.backend,
        model_id=cfg.vlm.model_id,
        device=cfg.vlm.device or None,
        api_key=cfg.vlm.api_key or None,
        cloud_model=cfg.vlm.cloud_model,
        cloud_base_url=cfg.vlm.cloud_base_url or None,
    )

    logger.info("Initializing grasp planner...")
    planner = GraspPlanner(
        camera_to_robot_tf=cam_to_robot,
        pre_grasp_height_m=cfg.pre_grasp_height,
        lift_height_m=cfg.lift_height,
    )

    logger.info(f"Loading kinematics from URDF: {cfg.urdf}")
    kinematics = RobotKinematics(
        urdf_path=cfg.urdf,
        target_frame_name=cfg.ee_frame,
        joint_names=SO100_MOTOR_NAMES,
    )
    motion_exec = MotionExecutionConfig()

    logger.info("Connecting robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    camera_key = cfg.camera_key
    depth_camera = None
    for cam_name, cam in robot.cameras.items():
        if cam_name == camera_key:
            depth_camera = cam
            break

    if depth_camera is None:
        raise ValueError(
            f"Camera '{camera_key}' not found on robot. Available: {list(robot.cameras.keys())}"
        )

    intrinsics: dict[str, float] = {}
    if hasattr(depth_camera, "get_depth_intrinsics"):
        try:
            intrinsics = depth_camera.get_depth_intrinsics()
            logger.info(f"Got depth intrinsics: {intrinsics}")
        except Exception as e:
            logger.warning(f"Could not get depth intrinsics: {e}")

    try:
        logger.info(f"Task: '{cfg.task}'")
        logger.info("Starting perception-action loop...\n")

        for attempt in range(cfg.max_attempts):
            logger.info(f"=== Attempt {attempt + 1}/{cfg.max_attempts} ===")

            obs = robot.get_observation()
            rgb = obs.get(camera_key)
            depth = obs.get(f"{camera_key}_depth")

            if rgb is None:
                logger.error(f"No RGB frame from '{camera_key}'")
                continue
            if depth is None:
                logger.error(f"No depth frame from '{camera_key}_depth'")
                continue

            rgb_np = np.asarray(rgb)
            depth_np = np.asarray(depth)

            # Live view of the current observation before detection.
            try:
                depth_vis = colorize_depth(depth_np)
                h, w = rgb_np.shape[:2]
                depth_resized = cv2.resize(depth_vis, (w, h))
                combined = np.hstack(
                    [
                        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
                        depth_resized,
                    ]
                )
                status = f"Task: '{cfg.task}'  Attempt {attempt + 1}/{cfg.max_attempts}"
                cv2.putText(
                    combined,
                    status,
                    (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("LeRobot Auto Manipulate - RGB | Depth", combined)
                cv2.waitKey(1)
            except Exception:
                # Visualization is best-effort and should not break the loop.
                pass

            logger.info(f"Detecting objects matching: '{cfg.task}'")
            detections = detector.detect(rgb_np, cfg.task)

            if not detections:
                logger.warning("No objects detected. Retrying...")
                time.sleep(1.0)
                continue

            logger.info(f"Detected {len(detections)} object(s):")
            for det in detections:
                logger.info(f"  - {det.label} at {det.bbox_xyxy}")

            # JSON-style dump of detections for debugging.
            try:
                detections_json = [
                    {
                        "label": det.label,
                        "bbox_xyxy": list(det.bbox_xyxy),
                        "has_mask": det.mask is not None,
                    }
                    for det in detections
                ]
                logger.info("Detections JSON: %s", json.dumps(detections_json))
            except Exception:
                pass

            det = detections[0]
            mask = det.mask
            if mask is None:
                logger.warning("Detection has no mask, skipping.")
                continue

            obj_state = compute_object_state(depth_np, mask, intrinsics)
            center = obj_state["obj_center_xyz"]
            size = obj_state["obj_size_xyz"]
            dist = obj_state["obj_distance"]

            if np.allclose(center, 0):
                logger.warning("Could not compute 3D state (no valid depth in mask). Retrying...")
                time.sleep(1.0)
                continue

            logger.info(
                f"Object '{det.label}': center={center}, size={size}, distance={dist[0]:.3f}m"
            )

            # Visualize all detections and 3D info for the chosen object.
            try:
                vis_rgb = cv2.cvtColor(rgb_np.copy(), cv2.COLOR_RGB2BGR)

                for d in detections:
                    x1, y1, x2, y2 = d.bbox_xyxy
                    color = (0, 255, 0) if d is det else (0, 200, 255)
                    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)

                    if d is det:
                        text = f"{d.label} d={dist[0]:.2f}m"
                    else:
                        text = d.label

                    cv2.putText(
                        vis_rgb,
                        text,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                depth_vis = colorize_depth(depth_np)
                h, w = vis_rgb.shape[:2]
                depth_resized = cv2.resize(depth_vis, (w, h))
                combined = np.hstack([vis_rgb, depth_resized])
                cv2.imshow("LeRobot Auto Manipulate - RGB | Depth", combined)
                cv2.waitKey(1)
            except Exception:
                pass

            logger.info("Planning grasp...")
            waypoints = planner.plan_pick(center, size, det.label)

            logger.info(f"Executing {len(waypoints)} waypoints...")
            execute_waypoints(robot, kinematics, waypoints, SO100_MOTOR_NAMES, motion_exec)

            logger.info("Pick complete!\n")

            if cfg.place_target:
                place_xyz = np.array(
                    [float(v) for v in cfg.place_target.split(",")],
                    dtype=np.float64,
                )
                logger.info(f"Planning place at {place_xyz}...")
                place_wps = planner.plan_place(place_xyz)
                logger.info(f"Executing {len(place_wps)} place waypoints...")
                execute_waypoints(robot, kinematics, place_wps, SO100_MOTOR_NAMES, motion_exec)
                logger.info("Place complete!\n")

            break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Disconnecting robot...")
        robot.disconnect()


def main():
    auto_manipulate()


if __name__ == "__main__":
    main()
