#!/usr/bin/env python
"""
Main pipeline entry point: Camera → Perception → Reasoning → Arm Execution.

Loads configuration from config.yaml and runs the full observe → reason → act loop.

Usage::

    # Full pipeline with robot
    python main.py --task "pick up the red cube"

    # Perception-only (no robot)
    python main.py --task "pick up the red cube" --dry-run --dry-run-image sample.png

    # Override config values
    python main.py --task "pick up the bottle" --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_tf_string(tf_str: str) -> np.ndarray:
    """Parse 'x,y,z,rx,ry,rz' into a 4x4 homogeneous transform."""
    from scipy.spatial.transform import Rotation

    parts = [float(v.strip()) for v in tf_str.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(parts)}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = parts[:3]
    if any(abs(v) > 1e-8 for v in parts[3:6]):
        tf[:3, :3] = Rotation.from_rotvec(parts[3:6]).as_matrix()
    return tf


def transform_cam_to_robot(
    point_cam: np.ndarray,
    T_cam_to_robot: np.ndarray,
    convention: str = "opencv",
) -> np.ndarray:
    """Transform a 3D point from camera frame to robot base frame.

    Args:
        point_cam: [x, y, z] in camera frame (meters).
        T_cam_to_robot: 4x4 extrinsic transform.
        convention: "opencv" applies X-right,Y-down,Z-forward → X-fwd,Y-left,Z-up.

    Returns:
        [x, y, z] in robot base frame (meters).
    """
    p = np.asarray(point_cam, dtype=np.float64).reshape(3)
    if convention == "opencv":
        # Camera: X-right, Y-down, Z-forward → Robot: X-forward, Y-left, Z-up
        p = np.array([p[2], -p[0], -p[1]], dtype=np.float64)
    p_h = np.array([p[0], p[1], p[2], 1.0])
    p_robot = (T_cam_to_robot @ p_h)[:3]
    return p_robot


def pixel_to_3d(
    pixel_x: int,
    pixel_y: int,
    depth_frame: np.ndarray,
    intrinsics: dict[str, float],
    roi_size: int = 5,
) -> np.ndarray:
    """Convert a pixel + depth map to 3D point in camera frame.

    Args:
        pixel_x, pixel_y: Pixel coordinates.
        depth_frame: (H, W) uint16 depth in millimeters.
        intrinsics: Dict with fx, fy, cx, cy, depth_scale.
        roi_size: Pixels to average depth over (reduces noise).

    Returns:
        [x, y, z] in camera frame (meters).
    """
    h, w = depth_frame.shape[:2]
    half = roi_size // 2
    y0 = max(0, int(pixel_y) - half)
    y1 = min(h, int(pixel_y) + half + 1)
    x0 = max(0, int(pixel_x) - half)
    x1 = min(w, int(pixel_x) + half + 1)

    roi = depth_frame[y0:y1, x0:x1].astype(np.float64)
    valid = roi[roi > 0]
    if valid.size == 0:
        return np.zeros(3)

    depth_mm = float(np.median(valid))
    depth_m = depth_mm * intrinsics.get("depth_scale", 0.001)

    if depth_m < 0.05 or depth_m > 2.0:
        logger.warning("Depth %.3fm out of valid range [0.05, 2.0]", depth_m)
        return np.zeros(3)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x = (pixel_x - cx) * depth_m / fx
    y = (pixel_y - cy) * depth_m / fy
    z = depth_m

    return np.array([x, y, z])


class PerceptionPipeline:
    """Wraps VLM detection + depth fusion into a single observe() call."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        pcfg = cfg["perception"]

        from lerobot.perception.vlm_detector import VLMDetector

        vlm_cfg = pcfg["vlm"]
        self.detector = VLMDetector(
            backend=vlm_cfg["backend"],
            model_id=vlm_cfg.get("model_id", "microsoft/Florence-2-base"),
            cloud_model=vlm_cfg.get("cloud_model", "gpt-4o"),
            api_key=vlm_cfg.get("api_key") or None,
        )
        self.depth_fusion_frames = pcfg.get("depth_fusion_frames", 1)
        self.color_filter = pcfg.get("color_filter_min_fraction", 0.0)
        self.max_cam_dist = pcfg.get("max_object_cam_distance_m", 2.5)
        self._depth_buffer: list[np.ndarray] = []

    def observe(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: dict[str, float],
        query: str,
    ) -> tuple:
        """Run full perception: VLM detection + depth→3D.

        Returns:
            (SceneObservation, list of (detection, state_dict) pairs)
        """
        from lerobot.perception.scene_from_rgb_depth import build_scene_from_rgb_depth
        from lerobot.processor.depth_perception_processor import compute_object_state

        # Temporal depth fusion
        if self.depth_fusion_frames > 1:
            self._depth_buffer.append(depth.copy())
            if len(self._depth_buffer) > self.depth_fusion_frames:
                self._depth_buffer.pop(0)
            if len(self._depth_buffer) >= 2:
                stack = np.stack(self._depth_buffer, axis=0).astype(np.float32)
                stack[stack == 0] = np.nan
                with np.errstate(all="ignore"):
                    fused = np.nanmedian(stack, axis=0)
                fused = np.nan_to_num(fused, nan=0.0).astype(np.uint16)
                depth = fused

        scene, det_states = build_scene_from_rgb_depth(
            rgb_np=rgb,
            depth_np=depth,
            detector=self.detector,
            compute_object_state=compute_object_state,
            intrinsics=intrinsics,
            query=query,
            color_filter_min_fraction=self.color_filter if self.color_filter > 0 else None,
        )

        # Filter by max camera distance
        if self.max_cam_dist > 0:
            filtered_objects = []
            filtered_det_states = []
            for obj, ds in zip(scene.objects, det_states):
                c = np.array(obj.center_xyz)
                if np.linalg.norm(c) <= self.max_cam_dist:
                    filtered_objects.append(obj)
                    filtered_det_states.append(ds)
                else:
                    logger.warning("Dropping %s: cam distance %.2fm > max %.2fm",
                                   obj.label, np.linalg.norm(c), self.max_cam_dist)
            scene.objects = filtered_objects
            det_states = filtered_det_states

        return scene, det_states


def build_scene_description(scene, T_cam_to_robot, convention="opencv") -> dict:
    """Build the structured scene dict from the spec.

    Adds robot-frame positions to each detected object.
    """
    objects = []
    for obj in scene.objects:
        cam_pos = np.array(obj.center_xyz)
        robot_pos = transform_cam_to_robot(cam_pos, T_cam_to_robot, convention)
        objects.append({
            "label": obj.label,
            "confidence": 1.0,
            "center_pixel": None,
            "position_3d_camera": list(obj.center_xyz),
            "position_3d_robot": robot_pos.tolist(),
            "size_3d": list(obj.size_xyz),
            "depth_m": obj.distance_m,
        })
    return {
        "objects": objects,
        "task": scene.task,
        "timestamp": time.time(),
    }


def run_pipeline(cfg: dict, task: str, dry_run: bool = False, dry_run_image: str = "") -> None:
    """Main observe → reason → act loop."""
    from lerobot.agent import AgentAction, ReasoningAgent, SceneObservation

    # --- Init camera (or dry-run image) ---
    camera = None
    intrinsics = None
    if not dry_run:
        from lerobot.cameras.oakd import OAKDCamera, OAKDCameraConfig

        cam_cfg = cfg["camera"]
        w, h = cam_cfg["rgb_resolution"]
        camera_config = OAKDCameraConfig(
            device_id=cam_cfg.get("device_id", ""),
            fps=cam_cfg["fps"],
            width=w,
            height=h,
            use_depth=cam_cfg.get("use_depth", True),
            stereo_preset=cam_cfg.get("depth_preset", "FAST_ACCURACY"),
            stereo_confidence_threshold=cam_cfg.get("stereo_confidence_threshold", 200),
            warmup_s=cam_cfg.get("warmup_s", 2),
        )
        camera = OAKDCamera(camera_config)
        camera.connect()
        intrinsics = camera.get_depth_intrinsics()
        logger.info("Camera connected. Intrinsics: %s", intrinsics)
    else:
        # Dry run: load static image
        if not dry_run_image:
            raise ValueError("--dry-run-image required in dry-run mode")
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
                      "width": 640, "height": 480, "depth_scale": 0.001}

    # --- Init arm controller ---
    arm = None
    if not dry_run:
        from robot.arm_controller import ArmController, ArmControllerConfig

        robot_cfg = cfg["robot"]
        arm_config = ArmControllerConfig(
            urdf_path=robot_cfg.get("urdf", "SO101/so101_new_calib.urdf"),
            ee_frame=robot_cfg.get("ee_frame", "gripper_frame_link"),
            home_joints_deg=robot_cfg.get("home_joints"),
            gripper_max_width_mm=robot_cfg.get("gripper_max_width_mm", 80.0),
            cartesian_step_m=cfg["motion"]["cartesian_step_m"],
            min_steps_per_segment=cfg["motion"]["min_steps_per_segment"],
            inter_step_sleep_s=cfg["motion"]["inter_step_sleep_s"],
            settle_timeout_s=cfg["motion"]["settle_timeout_s"],
            settle_threshold_deg=cfg["motion"]["settle_threshold_deg"],
        )

        # Build the LeRobot robot via the camera's existing instance
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SOFollowerRobotConfig

        so_config = SOFollowerRobotConfig(
            port="/dev/tty.usbmodem5A4B0479741",
            cameras={},  # camera managed separately
            use_degrees=robot_cfg.get("use_degrees", True),
        )
        robot = make_robot_from_config(so_config)
        arm = ArmController(robot=robot, config=arm_config)
        arm.connect()
        arm.home()
        logger.info("Arm connected and homed.")

    # --- Init perception ---
    perception = PerceptionPipeline(cfg)

    # --- Init reasoning agent ---
    rcfg = cfg["reasoning"]
    agent = ReasoningAgent(
        api_key=rcfg.get("api_key") or None,
        base_url=rcfg.get("base_url") or None,
        model=rcfg.get("model", "gemini-2.5-flash"),
        is_gemini=rcfg.get("use_gemini", True),
    )

    # --- Extrinsic calibration ---
    he_cfg = cfg["hand_eye_transform"]
    T_cam_to_robot = parse_tf_string(he_cfg["camera_to_robot_tf"])
    convention = he_cfg.get("camera_frame_convention", "opencv")

    # --- Main loop ---
    max_steps = cfg.get("workspace", {}).get("max_steps", 10)
    show_viz = cfg.get("logging", {}).get("show_viz", True)

    logger.info("Starting pipeline: task='%s'", task)

    for step in range(max_steps):
        logger.info("=== Step %d/%d ===", step + 1, max_steps)

        # 1. Observe
        if dry_run:
            rgb = cv2.imread(dry_run_image)
            if rgb is None:
                raise FileNotFoundError(f"Cannot read: {dry_run_image}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = np.full(rgb.shape[:2], 500, dtype=np.uint16)  # synthetic 50cm
        else:
            rgb = camera.read()
            depth = camera.read_depth()

        scene, det_states = perception.observe(rgb, depth, intrinsics, task)
        scene_desc = build_scene_description(scene, T_cam_to_robot, convention)

        logger.info("Detected %d objects", len(scene.objects))
        for obj in scene_desc["objects"]:
            logger.info("  %s: cam=%s robot=%s",
                        obj["label"],
                        [f"{v:.3f}" for v in obj["position_3d_camera"]],
                        [f"{v:.3f}" for v in obj["position_3d_robot"]])

        # 2. Visualize
        if show_viz:
            viz = rgb.copy()
            if viz.shape[2] == 3 and viz.dtype == np.uint8:
                viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
            else:
                viz_bgr = viz
            for det, state in det_states:
                if hasattr(det, "bbox_xyxy"):
                    x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                    cv2.rectangle(viz_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(viz_bgr, det.label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Perception", viz_bgr)
            cv2.waitKey(1)

        # 3. Reason
        scene.task = task
        action = agent.reason(scene)
        logger.info("Agent action: %s (reason: %s)", action.action, action.reason)

        if action.is_terminal():
            logger.info("Task %s: %s",
                        "completed" if action.success else "failed",
                        action.reason or "")
            break

        # 4. Act
        if action.is_pick() and arm is not None:
            idx = action.object_index
            if idx < len(scene_desc["objects"]):
                obj = scene_desc["objects"][idx]
                robot_pos = obj["position_3d_robot"]
                obj_size = obj.get("size_3d", [0.04, 0.04, 0.04])
                logger.info("Picking %s at robot frame %s", obj["label"], robot_pos)
                try:
                    arm.pick(robot_pos, object_size=obj_size, object_label=obj["label"])
                except Exception as e:
                    logger.error("Pick failed: %s", e)
            else:
                logger.warning("Invalid object index %d (only %d objects)", idx, len(scene_desc["objects"]))

        elif action.is_place() and arm is not None:
            place_cam = np.array(action.place_xyz)
            place_robot = transform_cam_to_robot(place_cam, T_cam_to_robot, convention)
            logger.info("Placing at robot frame %s", place_robot.tolist())
            try:
                arm.place(place_robot.tolist())
            except Exception as e:
                logger.error("Place failed: %s", e)

        elif action.action == "retry":
            logger.info("Retrying — re-observing scene...")
            continue

        elif dry_run:
            logger.info("[DRY RUN] Would execute: %s", action.action)
            break

    # --- Cleanup ---
    if show_viz:
        cv2.destroyAllWindows()
    if arm is not None:
        arm.home()
        arm.disconnect()
    if camera is not None:
        camera.disconnect()
    logger.info("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Robot Manipulation Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--task", required=True, help="Natural language task (e.g. 'pick up the red cube')")
    parser.add_argument("--dry-run", action="store_true", help="Skip robot/camera, use static image")
    parser.add_argument("--dry-run-image", default="", help="Path to RGB image for dry-run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    run_pipeline(cfg, task=args.task, dry_run=args.dry_run, dry_run_image=args.dry_run_image)


if __name__ == "__main__":
    main()
