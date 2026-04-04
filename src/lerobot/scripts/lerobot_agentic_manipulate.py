#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Agentic manipulation: stereo camera + VLM + LLM reasoning → choose and execute actions.

Flow: Observe (stereo RGB+depth, VLM detections, 3D state) → Reason (LLM picks action)
→ Act (pick by index, place, or done/fail/retry). The LLM can choose *which* object
to pick when several are detected, or ask to retry/fail.

Coordinate frame layout (typical): Camera (front) -> Object -> Robot (behind).
The camera sees the object; the robot reaches forward to grasp it. Use
--camera_frame_convention=opencv so camera coords (X right, Y down, Z forward)
map correctly to robot coords (X forward, Y left, Z up). Calibrate
--camera_to_robot_tf with the camera position in robot frame (e.g. "0.4,0,0.1").

**Gripper-mounted camera (SO-101 eye-in-hand):** use ``--camera-mount=gripper`` and
``--gripper-camera-tf`` (camera optical frame → ``--ee-frame``, same ``x,y,z,rx,ry,rz``
format as ``parse_tf_string``). Each frame uses ``T_base_cam = FK(q) @ T_ee_cam``.
Ignore ``--camera-to-robot-tf`` translation for extrinsics in that mode (FK supplies base pose).

Example (layout: Camera in front -> Object -> Robot behind)::

    export GEMINI_API_KEY=your-key
    lerobot-agentic-manipulate \\
        --robot.type=so100_follower \\
        --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \\
        --task="pick up the red cube" \\
        --urdf=./SO101/so101_new_calib.urdf \\
        --vlm.backend=gemini \\
        # or --vlm.backend=claude --vlm.cloud_model=claude-sonnet-4-20250514 \\
        --camera_frame_convention=opencv \\
        --camera_to_robot_tf="0.4,0,0.1,0,0,0"

If the robot arm is not connected or motors are not found (e.g. "Missing motor IDs" / "Full found motor list: {}"),
ensure the arm is powered and the correct USB port is used. To test perception and reasoning without the robot, use::

    lerobot-agentic-manipulate --dry-run --dry-run-image=/path/to/image.png --task="pick up the red cube" ...

Stereo depth can be temporally fused and VLM boxes filtered by query color::

    --depth_fusion_frames=7 --color_filter_min_fraction=0.08

Stream camera RGB + depth to the Rerun viewer (opens a window; same flags as teleoperate)::

    --display_data=true
    # optional remote viewer: --display_ip=127.0.0.1 --display_port=9876

See how often the reasoning LLM runs vs. motion waypoints::

    --agent-step-mode=micro
    --log-loop-status-interval-s=2

At INFO, lines are tagged ``[agentic]``, ``[perceive]``, ``[plan]``, ``[motion]``. This entry point also quiets
per-request HTTP and one-shot Rerun/STL messages unless you use ``--log-console-level=DEBUG``.

Closed-loop “look again after each small move” (full VLM + LLM every iteration; slow but predictable)::

    --agent-step-mode=micro --vlm-every-agent-step=true --llm-every-agent-step=true

If perception goes empty **during** a micro pick, the arm **retreats one planned waypoint** (then re-observes) instead of immediately failing::

    # defaults: recover_on_lost_sight=true, lost_sight_max_retreats_per_pick=8
    --recover-on-lost-sight=false   # disable
    # verbose: every control tick (requires DEBUG console)
    --log-loop-tick-debug=true --log-console-level=DEBUG

3D preview in Rerun: SO101 **STL meshes** from the URDF (next to ``robot.urdf`` / ``meshes/``), solid **cube** objects from 3D size, ground grid, EE axes, plan path. Opens a Spatial 3D view when using a local viewer::

    --display_sim3d=true
    # with camera: --display_data=true --display_sim3d=true
    # 3D is logged every control tick (including micro pick waypoints). Timeline: ``frame``.

Bad depth / huge reaches (camera-frame centers like 10m) are usually a broken mask; we drop those over
``--max-object-cam-distance-m``. Override grasp geometry without fixing depth::

    --manual-object-xyz="0.05,-0.02,0.35" --manual-object-frame=cam
    # or pose in robot base: --manual-object-frame=base --manual-object-xyz="0.3,0.0,0.05"
    # optional size (m): --manual-object-size-xyz="0.04,0.04,0.04"

Micro waypoint mode no longer calls the LLM every tick by default; use ``--micro-reason-each-waypoint=true`` for that.

**JSON state reasoning** (VLM seeds poses; LLM edits the same JSON, then motion uses the refined pose)::

    --agent-reasoning-strategy=json_state
"""

from __future__ import annotations

import logging
import os
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import rerun as rr

from lerobot.agent import (
    AgentAction,
    ReasoningAgent,
    SceneObject,
    SceneObservation,
    build_vlm_scene_json_payload,
)
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig  # noqa: F401 — registers SO follower
from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d
from lerobot.utils.motion_executor import (
    MotionExecutionConfig,
    execute_waypoint,
    execute_waypoint_microsteps,
    execute_waypoints,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import (
    colorize_depth_mm_u16,
    init_rerun,
    log_stereo_pair_to_rerun,
    send_agentic_rerun_blueprint,
)

logger = logging.getLogger(__name__)

SO100_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class VLMConfig:
    backend: str = "gemini"
    model_id: str = "microsoft/Florence-2-base"
    device: str = ""
    api_key: str = ""
    cloud_model: str = "gpt-4o"
    cloud_base_url: str = ""


@dataclass
class AgentConfig:
    """LLM used for reasoning (observe → reason → act)."""

    model: str = "gemini-2.5-flash"
    api_key: str = ""
    base_url: str = ""
    use_gemini: bool = True


@dataclass
class AgenticManipulateConfig:
    robot: RobotConfig = field(
        default_factory=lambda: SOFollowerRobotConfig(port="", cameras={}),
    )
    vlm: VLMConfig = field(default_factory=VLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    task: str = "pick up the red cube"
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    # "fixed": use camera_to_robot_tf (world/base camera). "gripper": T_base_cam = FK(q) @ gripper_camera_tf.
    camera_mount: str = "fixed"
    camera_to_robot_tf: str = "0,0,0.4,0,0,0"
    # Eye-in-hand: homogeneous camera(optical)→ee_frame (meters, rotvec rx,ry,rz). Calibrate (e.g. Charuco).
    gripper_camera_tf: str = "0,0,0,0,0,0"
    # "opencv" = Camera (X right, Y down, Z forward) -> Robot (X forward, Y left, Z up).
    # Use for layout: Camera (front) -> Object -> Robot (behind). Without this, robot may move away from object.
    camera_frame_convention: str = "opencv"
    # If robot goes to the right of object, try --camera-flip-lateral. If to the left, omit it.
    camera_flip_lateral: bool = False
    place_target: str = ""
    max_attempts: int = 3
    max_agent_steps: int = 10
    pre_grasp_height: float = 0.06
    lift_height: float = 0.08
    # Simple workspace model (meters): project target from object pose into reachable region.
    arm_base_xyz: str = "0,0,0"
    min_reach_m: float = 0.10
    max_reach_m: float = 0.42
    min_grasp_z_m: float = 0.01
    max_grasp_z_m: float = 0.35
    # Pull first approach waypoint toward the arm base in XY (m), then align over object, then descend.
    approach_xy_retract_m: float = 0.055

    # Interaction tuning (prevents "slam into object" by avoiding center-as-grasp-target).
    grasp_clearance_m: float = 0.008
    grasp_seat_m: float = 0.004
    place_clearance_m: float = 0.010

    # Motion: Cartesian micro-steps (recommended) vs joint lerp between IK solutions.
    motion_use_cartesian: bool = True
    # Slower/safer defaults for real hardware.
    motion_cartesian_step_m: float = 0.003
    motion_min_steps_per_segment: int = 12
    motion_inter_step_sleep_s: float = 0.05
    motion_settle_threshold_deg: float = 2.0
    motion_settle_timeout_s: float = 4.5

    # Fast loop: track boxes at camera FPS, re-detect with VLM periodically, and reason periodically.
    enable_tracking: bool = True
    # How often we call models:
    # - detect_period_s: VLM detections (Gemini/Florence/etc.)
    # - reason_period_s: reasoning LLM action selection
    detect_period_s: float = 2.0
    reason_period_s: float = 2.0
    # If true, (re)initialize tracker from the *agent-selected* object when action is pick.
    # If false, tracker is initialized from the first/most confident detection.
    track_agent_choice: bool = True
    # Visualization
    show_viz: bool = True
    # Rerun viewer (same pattern as lerobot-teleoperate): spawns local viewer or connects remotely.
    display_data: bool = False
    display_sim3d: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False

    # Dry run: skip robot connection, use static image(s) to test perception + agent (no motion).
    dry_run: bool = False
    dry_run_image: str = ""  # Path to RGB image (required if dry_run=True)
    dry_run_depth: str = ""  # Optional path to depth .npy or image; if missing, synthetic depth is used

    # Perception robustness: temporal median of depth (reduces stereo speckle); HSV check on VLM boxes.
    depth_fusion_frames: int = 1
    color_filter_min_fraction: float = 0.0
    # Depth values outside [min, max] (m) are ignored when lifting mask/bbox pixels to 3D.
    perception_min_depth_m: float = 0.05
    perception_max_depth_m: float = 5.0
    # If mask-based |center_cam| exceeds this (m), retry using median depth inside the VLM bbox (SAM/OAK outliers).
    perception_bbox_fallback_if_mask_far_m: float = 2.0
    # Drop detections whose camera-frame center is still beyond this after fallback. 0 = off.
    max_object_cam_distance_m: float = 5.0

    # MCP server (lerobot-agentic-mcp): DimOS-style SSE binding; stdio is default for Cursor.
    mcp_transport: str = "stdio"  # "stdio" | "sse"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 9990
    mcp_mount_path: str = "/"

    # How the reasoning LLM consumes perception:
    # - "summary": plain-text scene (default).
    # - "json_state": VLM+depth scene as JSON; model returns updated objects + action; pick uses refined center/size.
    agent_reasoning_strategy: str = "summary"  # "summary" | "json_state"

    # Agent/action loop granularity:
    # - "macro": one agent action triggers a full pick/place execution (current behavior).
    # - "micro": agent must re-affirm the action every waypoint (approach_far → align_xy → pre_grasp → ...).
    agent_step_mode: str = "macro"  # "macro" | "micro"
    # In micro mode, execute only a small number of internal micro-steps per agent decision.
    micro_max_microsteps_per_decision: int = 12
    # If true, call the reasoning LLM after every waypoint micro-step (slow + noisy re-planning).
    micro_reason_each_waypoint: bool = False
    # If true (default), skip expensive VLM runs while a micro pick is still executing waypoints.
    micro_pause_vlm_during_motion: bool = True
    # If true: run full VLM detection every main-loop iteration (fresh image every step). Implies VLM is not
    # paused during micro motion. Pair with agent_step_mode=micro for small actions; use llm_every_agent_step
    # to also query the LLM every iteration (otherwise reason_period_s still throttles the LLM).
    vlm_every_agent_step: bool = False
    # If true: call the reasoning LLM every loop iteration (ignores reason_period_s except the first forced step).
    llm_every_agent_step: bool = False
    # Micro pick only: if the scene has no objects mid-pick, move the arm back one planned waypoint (re-acquire FOV)
    # instead of immediately asking the LLM to fail. Capped per pick to avoid infinite retreat.
    recover_on_lost_sight: bool = True
    lost_sight_max_retreats_per_pick: int = 8

    # Optional: fix grasp target from measured / known pose (planner still expects camera-frame center internally).
    manual_object_xyz: str = ""
    manual_object_frame: str = "cam"  # "cam" | "base"
    manual_object_size_xyz: str = ""
    manual_object_index: int = 0

    # Logging: write structured trace with obs/reason/act to a JSONL file.
    log_jsonl_path: str = ""
    log_scene_text: bool = True
    log_llm_raw: bool = True
    # Terminal: periodic INFO while the main loop is spinning (mode, time since last LLM, micro plan). 0 = off.
    log_loop_status_interval_s: float = 0.0
    # Log every control tick at DEBUG (noisy; use --log-console-level=DEBUG).
    log_loop_tick_debug: bool = False
    log_console_level: str = "INFO"


def parse_tf_string(tf_str: str) -> np.ndarray:
    """Parse 'x,y,z,rx,ry,rz' into a 4x4 homogeneous transform."""
    from scipy.spatial.transform import Rotation

    parts = [float(v.strip()) for v in tf_str.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(parts)}: {tf_str}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = parts[:3]
    if any(abs(v) > 1e-8 for v in parts[3:6]):
        tf[:3, :3] = Rotation.from_rotvec(parts[3:6]).as_matrix()
    return tf


def parse_xyz_string(xyz_str: str) -> np.ndarray:
    """Parse 'x,y,z' into a 3-vector."""
    parts = [float(v.strip()) for v in xyz_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values (x,y,z), got {len(parts)}: {xyz_str}")
    return np.array(parts, dtype=np.float64)


def _scene_from_det_pairs(pairs: list, task: str) -> SceneObservation:
    objects: list[SceneObject] = []
    for i, (det, st) in enumerate(pairs):
        c = st["obj_center_xyz"]
        s = st["obj_size_xyz"]
        dist = st["obj_distance"]
        objects.append(
            SceneObject(
                index=i,
                label=str(getattr(det, "label", "object")),
                center_xyz=(float(c[0]), float(c[1]), float(c[2])),
                size_xyz=(float(s[0]), float(s[1]), float(s[2])),
                distance_m=float(dist[0]),
            )
        )
    return SceneObservation(objects=objects, task=task)


def _refined_object_pose_from_json_state(
    refined_state: dict | None, object_index: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (center_xyz, size_xyz) in camera frame from last json_state model output, if valid."""
    if not refined_state:
        return None
    objs = refined_state.get("objects")
    if not isinstance(objs, list):
        return None
    for o in objs:
        if not isinstance(o, dict):
            continue
        if int(o.get("index", -1)) != int(object_index):
            continue
        c = o.get("center_xyz")
        s = o.get("size_xyz")
        if c is None or s is None or len(c) < 3 or len(s) < 3:
            return None
        return (
            np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float32),
            np.array([float(s[0]), float(s[1]), float(s[2])], dtype=np.float32),
        )
    return None


def _filter_detections_by_cam_distance(
    detections_with_state: list,
    task: str,
    max_cam_dist_m: float,
) -> tuple[list, SceneObservation]:
    """Drop detections with absurd camera-frame range (flying pixels / bad mask)."""
    if max_cam_dist_m <= 0:
        return detections_with_state, _scene_from_det_pairs(detections_with_state, task)
    kept: list = []
    for det, st in detections_with_state:
        c = np.asarray(st["obj_center_xyz"], dtype=np.float64).reshape(3)
        if np.allclose(c, 0):
            continue
        d = float(np.linalg.norm(c))
        if d > max_cam_dist_m:
            logger.warning(
                "Ignoring detection %r: |center_cam|=%.3fm > max_object_cam_distance_m=%.3fm (bad depth/mask?)",
                getattr(det, "label", "?"),
                d,
                max_cam_dist_m,
            )
            continue
        kept.append((det, st))
    return kept, _scene_from_det_pairs(kept, task)


def _apply_manual_object_geometry(
    cfg: AgenticManipulateConfig,
    planner,
    object_index: int,
    center: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace center/size for grasp planning when ``manual_object_xyz`` is set."""
    raw = (cfg.manual_object_xyz or "").strip()
    if not raw:
        return center, size
    if int(object_index) != int(cfg.manual_object_index):
        return center, size
    p = parse_xyz_string(raw).astype(np.float32)
    frame = (cfg.manual_object_frame or "cam").lower().strip()
    if frame == "base":
        T = np.asarray(planner.cam_to_robot_tf, dtype=np.float64)
        T_inv = np.linalg.inv(T)
        h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0], dtype=np.float64)
        cam_h = T_inv @ h
        center_out = cam_h[:3].astype(np.float32)
    elif frame == "cam":
        center_out = p
    else:
        raise ValueError(f"manual_object_frame must be 'cam' or 'base', got {cfg.manual_object_frame!r}")
    sz_raw = (cfg.manual_object_size_xyz or "").strip()
    if sz_raw:
        size_out = parse_xyz_string(sz_raw).astype(np.float32)
    else:
        size_out = np.asarray(size, dtype=np.float32).reshape(3)
    logger.info(
        "Manual object geometry for index %d (%s frame): center_cam=%s size=%s",
        object_index,
        frame,
        center_out,
        size_out,
    )
    return center_out, size_out


def camera_opencv_to_robot_rotation(flip_lateral: bool = False) -> np.ndarray:
    """Rotation from OpenCV camera frame to robot frame.

    OpenCV camera: X right, Y down, Z forward (into scene).
    Robot (SO-101): X forward, Y left, Z up.

    Use when layout is: Camera (front) -> Object -> Robot (behind).
    Maps: cam_Z (forward) -> robot_X (forward), cam_-Y (up) -> robot_Z (up).
    flip_lateral: if True, cam_X -> robot +Y (robot Y=right); else cam_X -> robot -Y (robot Y=left).
    """
    # Base: cam X->robot -Y, cam Y->robot -Z, cam Z->robot X
    R = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    if flip_lateral:
        R[1, 0] *= -1  # flip lateral: cam X -> robot +Y instead of -Y
    return R


def colorize_depth(depth_mm: np.ndarray, max_range_mm: int = 2000) -> np.ndarray:
    """Convert uint16 depth (mm) to BGR for display."""
    return colorize_depth_mm_u16(depth_mm, max_range_mm=max_range_mm)


def _agentic_reduce_third_party_log_noise() -> None:
    """Hide per-request HTTP and one-shot viewer/STL INFO lines unless you raise log level."""
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "lerobot.utils.urdf_visual_meshes",
        "lerobot.utils.visualization_utils",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _micro_retreat_previous_waypoint(
    robot,
    kinematics,
    waypoints: list,
    motor_names: list[str],
    motion_exec: MotionExecutionConfig,
    current_wp_index: int,
) -> tuple[int, int]:
    """Move toward the previous pick waypoint pose. Returns (new_wp_index, active_wp_micro_i=0)."""
    if current_wp_index <= 0 or not waypoints:
        return current_wp_index, 0
    new_i = current_wp_index - 1
    wp = waypoints[new_i]
    obs_r = robot.get_observation()
    cur_j = np.array([float(obs_r[f"{m}.pos"]) for m in motor_names], dtype=np.float64)
    execute_waypoint(robot, kinematics, wp, motor_names, motion_exec, current_joints=cur_j)
    return new_i, 0


def _scene_from_tracker_bbox(
    rgb_curr: np.ndarray,
    depth_fused: np.ndarray,
    tracked_bbox_xyxy: tuple[int, int, int, int],
    tracked_label: str,
    compute_object_state,
    intrinsics: dict,
    task: str,
) -> tuple[SceneObservation, list[tuple[object, dict]]]:
    """Build scene from an existing 2D track + depth (same path as the no-VLM loop tick)."""
    from lerobot.perception.vlm_detector import mask_from_bbox_grabcut

    mask = mask_from_bbox_grabcut(rgb_curr, tracked_bbox_xyxy)
    state = compute_object_state(depth_fused, mask, intrinsics)
    center = state["obj_center_xyz"]
    size = state["obj_size_xyz"]
    dist = state["obj_distance"]
    if np.allclose(center, 0):
        return SceneObservation(objects=[], task=task), []

    class _DetLike:
        def __init__(self, label: str, bbox_xyxy: tuple[int, int, int, int]):
            self.label = label
            self.bbox_xyxy = bbox_xyxy

    det_like = _DetLike(tracked_label or "tracked_object", tracked_bbox_xyxy)
    objects = [
        SceneObject(
            index=0,
            label=tracked_label or "tracked_object",
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
            size_xyz=(float(size[0]), float(size[1]), float(size[2])),
            distance_m=float(dist[0]),
        )
    ]
    return (
        SceneObservation(objects=objects, task=task),
        [(det_like, state)],
    )


def _observe_from_images(
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    detector,
    compute_object_state,
    intrinsics: dict,
    task: str,
    *,
    color_filter_min_fraction: float | None = None,
) -> tuple[SceneObservation, list[tuple[object, dict]], np.ndarray, np.ndarray]:
    """Run VLM detection and 3D state on given RGB+depth. Returns (scene, detections_with_state, rgb, depth)."""
    from lerobot.perception.scene_from_rgb_depth import build_scene_from_rgb_depth

    scene, detections_with_state = build_scene_from_rgb_depth(
        rgb_np,
        depth_np,
        detector,
        compute_object_state,
        intrinsics,
        task,
        color_filter_min_fraction=color_filter_min_fraction,
    )
    return scene, detections_with_state, rgb_np, depth_np


def _observe(
    robot,
    camera_key: str,
    detector,
    compute_object_state,
    intrinsics: dict,
    task: str,
    *,
    color_filter_min_fraction: float | None = None,
) -> tuple[SceneObservation, list[tuple[object, dict]], np.ndarray, np.ndarray]:
    """Get RGB+depth from robot, run VLM detection, compute 3D state. Returns (scene, detections_with_state, rgb, depth)."""
    obs = robot.get_observation()
    rgb = obs.get(camera_key)
    depth = obs.get(f"{camera_key}_depth")
    if rgb is None or depth is None:
        return (
            SceneObservation(objects=[], task=task),
            [],
            np.asarray(rgb) if rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            np.asarray(depth) if depth is not None else np.zeros((480, 640), dtype=np.uint16),
        )
    rgb_np = np.asarray(rgb)
    depth_np = np.asarray(depth)
    return _observe_from_images(
        rgb_np,
        depth_np,
        detector,
        compute_object_state,
        intrinsics,
        task,
        color_filter_min_fraction=color_filter_min_fraction,
    )


def _xyxy_to_xywh(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))


def _xywh_to_xyxy(bbox_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    return int(x), int(y), int(x + w), int(y + h)


def _clamp_xyxy(
    bbox_xyxy: tuple[int, int, int, int], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w - 1, int(x2)))
    y2 = max(0, min(img_h - 1, int(y2)))
    if x2 <= x1:
        x2 = min(img_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2


def _make_tracker():
    # Prefer CSRT (robust, slower) then KCF (faster, less robust).
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        try:
            return cv2.TrackerKCF_create()
        except Exception:
            return None


def _log_agentic_sim3d_once(
    cfg: AgenticManipulateConfig,
    *,
    dry_run: bool,
    obs_loop: dict | None,
    kinematics,
    planner,
    detections_with_state: list,
    rerun_frame_idx: int,
    last_plan_ee_positions: np.ndarray | None,
    last_plan_summary: str | None,
    active_object_index: int | None,
) -> None:
    """Log SO101 chain, objects (base frame), plan, and EE marker for the current loop tick."""
    if not cfg.display_sim3d:
        return
    try:
        if dry_run or obs_loop is None:
            joint_deg_viz = np.zeros(len(SO100_MOTOR_NAMES), dtype=np.float64)
        else:
            joint_deg_viz = np.array(
                [float(obs_loop[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
                dtype=np.float64,
            )
        obj_centers: list = []
        obj_halves: list = []
        obj_lbls: list[str] = []
        for det, st in detections_with_state:
            c = st["obj_center_xyz"]
            s = st["obj_size_xyz"]
            obj_centers.append(planner.point_camera_to_base(c))
            obj_halves.append(planner.half_extents_base_from_size_cam(s))
            obj_lbls.append(str(getattr(det, "label", "?")))
        Oc = np.stack(obj_centers, axis=0) if obj_centers else None
        Oh = np.stack(obj_halves, axis=0) if obj_halves else None
        n_obj = len(detections_with_state)
        focus_idx: int | None = None
        if n_obj > 0:
            ai = active_object_index
            if ai is not None and 0 <= int(ai) < n_obj:
                focus_idx = int(ai)
            else:
                focus_idx = 0
        log_manipulation_sim3d(
            frame_sequence=rerun_frame_idx,
            kinematics=kinematics,
            joint_deg=joint_deg_viz,
            object_centers_base=Oc,
            object_half_sizes_base=Oh,
            object_labels=obj_lbls if obj_lbls else None,
            focus_object_index=focus_idx,
            planned_ee_positions_base=last_plan_ee_positions,
            plan_summary=last_plan_summary or None,
        )
    except Exception as e:
        logger.warning("display_sim3d logging failed: %s", e)


@parser.wrap()
def agentic_manipulate(cfg: AgenticManipulateConfig):
    """Run the agentic loop: observe → reason → act until done/fail or max steps."""
    init_logging(console_level=cfg.log_console_level or "INFO")
    _agentic_reduce_third_party_log_noise()

    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.grasp_planner import GraspPlanner
    from lerobot.perception.vlm_detector import VLMDetector
    from lerobot.processor.depth_perception_processor import (
        compute_object_state,
        fuse_depth_temporal_median,
    )

    mount = (cfg.camera_mount or "fixed").lower().strip()
    if mount not in ("fixed", "gripper"):
        raise ValueError(f"camera_mount must be 'fixed' or 'gripper', got {cfg.camera_mount!r}")

    T_ee_cam = parse_tf_string(cfg.gripper_camera_tf)

    cam_to_robot = parse_tf_string(cfg.camera_to_robot_tf)
    if mount == "fixed" and (cfg.camera_frame_convention or "").lower() == "opencv":
        # Apply OpenCV->robot rotation; keep user's translation (camera position in robot frame).
        t = cam_to_robot[:3, 3].copy()
        R = camera_opencv_to_robot_rotation(flip_lateral=cfg.camera_flip_lateral)
        cam_to_robot = np.eye(4, dtype=np.float64)
        cam_to_robot[:3, :3] = R
        cam_to_robot[:3, 3] = t
    elif mount == "gripper" and (cfg.camera_frame_convention or "").lower() == "opencv":
        logger.debug(
            "camera_mount=gripper: gripper_camera_tf = camera(optical)→%s (opencv conv.; calib in optical frame)",
            cfg.ee_frame,
        )

    logger.debug("Initializing VLM detector...")
    detector = VLMDetector(
        backend=cfg.vlm.backend,
        model_id=cfg.vlm.model_id,
        device=cfg.vlm.device or None,
        api_key=cfg.vlm.api_key or None,
        cloud_model=cfg.vlm.cloud_model,
        cloud_base_url=cfg.vlm.cloud_base_url or None,
    )

    logger.debug("Initializing reasoning agent...")
    reasoning_agent = ReasoningAgent(
        api_key=cfg.agent.api_key or None,
        base_url=cfg.agent.base_url or None,
        model=cfg.agent.model,
        is_gemini=cfg.agent.use_gemini,
    )

    logger.debug("Initializing grasp planner...")
    planner_init_tf = cam_to_robot if mount == "fixed" else np.eye(4, dtype=np.float64)
    planner = GraspPlanner(
        camera_to_robot_tf=planner_init_tf,
        pre_grasp_height_m=cfg.pre_grasp_height,
        lift_height_m=cfg.lift_height,
        arm_base_xyz_m=parse_xyz_string(cfg.arm_base_xyz),
        min_reach_m=cfg.min_reach_m,
        max_reach_m=cfg.max_reach_m,
        min_grasp_z_m=cfg.min_grasp_z_m,
        max_grasp_z_m=cfg.max_grasp_z_m,
        approach_xy_retract_m=cfg.approach_xy_retract_m,
    )
    # Apply interaction heuristics from config (clearance, gentle "seat", and place clearance).
    planner.interaction.grasp_clearance_m = float(cfg.grasp_clearance_m)
    planner.interaction.grasp_seat_m = float(cfg.grasp_seat_m)
    planner.interaction.place_clearance_m = float(cfg.place_clearance_m)

    motion_exec = MotionExecutionConfig(
        use_cartesian_interp=cfg.motion_use_cartesian,
        cartesian_step_m=cfg.motion_cartesian_step_m,
        min_steps_per_segment=cfg.motion_min_steps_per_segment,
        inter_step_sleep_s=cfg.motion_inter_step_sleep_s,
        settle_threshold_deg=cfg.motion_settle_threshold_deg,
        settle_timeout_s=cfg.motion_settle_timeout_s,
    )

    reason_strategy = (cfg.agent_reasoning_strategy or "summary").lower().strip()
    if reason_strategy not in ("summary", "json_state"):
        raise ValueError(
            f"agent_reasoning_strategy must be 'summary' or 'json_state', got {cfg.agent_reasoning_strategy!r}"
        )
    if reason_strategy == "json_state":
        logger.info(
            "agent_reasoning_strategy=json_state: LLM receives/returns JSON (VLM seed → model refines poses → motion)."
        )

    logger.debug("Loading kinematics from URDF: %s", cfg.urdf)
    kinematics = RobotKinematics(
        urdf_path=cfg.urdf,
        target_frame_name=cfg.ee_frame,
        joint_names=SO100_MOTOR_NAMES,
    )
    if mount == "gripper":
        logger.debug(
            "Eye-in-hand: T_base_cam = FK(q) @ T_cam→%s each frame",
            cfg.ee_frame,
        )

    robot = None
    camera_key = cfg.camera_key
    intrinsics: dict[str, float] = {}

    if cfg.dry_run:
        if not cfg.dry_run_image or not cfg.dry_run_image.strip():
            raise ValueError("dry_run=True requires --dry-run-image=<path> to an RGB image.")
        logger.info("Dry run: skipping robot connection, using image from %s", cfg.dry_run_image.strip())
        rgb_np = cv2.imread(cfg.dry_run_image.strip())
        if rgb_np is None:
            raise FileNotFoundError(f"Could not load dry-run image: {cfg.dry_run_image}")
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        if cfg.dry_run_depth and cfg.dry_run_depth.strip():
            depth_np = np.load(cfg.dry_run_depth.strip())
            if depth_np.ndim == 3:
                depth_np = depth_np.squeeze()
        else:
            h, w = rgb_np.shape[:2]
            depth_np = np.full((h, w), 500, dtype=np.uint16)  # 500 mm synthetic
        # Default intrinsics (e.g. 640x480)
        h, w = rgb_np.shape[:2]
        intrinsics = {
            "fx": 525.0,
            "fy": 525.0,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "depth_scale": 0.001,
        }
    else:
        logger.debug("Connecting robot...")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

        depth_camera = None
        for cam_name, cam in robot.cameras.items():
            if cam_name == camera_key:
                depth_camera = cam
                break
        if depth_camera is None:
            raise ValueError(
                f"Camera '{camera_key}' not found. Available: {list(robot.cameras.keys())}"
            )

        if hasattr(depth_camera, "get_depth_intrinsics"):
            try:
                intrinsics = depth_camera.get_depth_intrinsics()
            except Exception as e:
                logger.warning("Could not get depth intrinsics: %s", e)

    intrinsics = dict(intrinsics)
    intrinsics["min_depth_m"] = float(cfg.perception_min_depth_m)
    intrinsics["max_depth_m"] = float(cfg.perception_max_depth_m)
    if float(cfg.perception_bbox_fallback_if_mask_far_m) > 0:
        intrinsics["bbox_fallback_if_mask_far_m"] = float(cfg.perception_bbox_fallback_if_mask_far_m)

    display_compressed = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )
    if cfg.display_data or cfg.display_sim3d:
        init_rerun(session_name="agentic_manipulation", ip=cfg.display_ip, port=cfg.display_port)
        send_agentic_rerun_blueprint(
            show_camera_stream=bool(cfg.display_data),
            show_sim3d=bool(cfg.display_sim3d),
        )
        if cfg.display_data or cfg.display_sim3d:
            logger.info(
                "[agentic] Rerun │ stream=%s │ sim3d=%s",
                bool(cfg.display_data),
                bool(cfg.display_sim3d),
            )

    try:
        logger.info(
            "[agentic] run │ task=%r │ step_mode=%s │ vlm=%s%s",
            cfg.task,
            (cfg.agent_step_mode or "macro").lower().strip(),
            cfg.vlm.backend,
            " [dry run]" if cfg.dry_run else "",
        )

        color_fn = cfg.color_filter_min_fraction if cfg.color_filter_min_fraction > 0 else None
        depth_history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.depth_fusion_frames)))
        rerun_frame_idx = 0
        last_plan_ee_positions: np.ndarray | None = None
        last_plan_summary: str = ""
        jsonl_path = (cfg.log_jsonl_path or "").strip()
        if not jsonl_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = os.path.abspath(f"agentic_manipulate_{ts}.jsonl")
        jsonl_f = open(jsonl_path, "a", encoding="utf-8")
        logger.info("[agentic] trace JSONL → %s", jsonl_path)

        for attempt in range(cfg.max_attempts):
            logger.info("[agentic] attempt %d/%d", attempt + 1, cfg.max_attempts)

            depth_history.clear()
            last_plan_ee_positions = None
            last_plan_summary = ""
            tracker = None
            tracked_label = ""
            tracked_bbox_xyxy: tuple[int, int, int, int] | None = None
            last_detect_t = 0.0
            last_reason_t = 0.0
            last_action: AgentAction | None = None
            step_mode = (cfg.agent_step_mode or "macro").lower().strip()
            if step_mode not in ("macro", "micro"):
                raise ValueError(f"agent_step_mode must be 'macro' or 'micro', got {cfg.agent_step_mode!r}")

            # Micro-step execution state: keep an active plan and execute 1 waypoint per agent step.
            active_action: str | None = None  # "pick" | "place"
            active_object_label: str = ""
            active_object_index: int | None = None
            active_waypoints: list | None = None
            active_wp_i: int = 0
            active_wp_micro_i: int = 0
            force_reason_next: bool = True
            last_loop_status_log_t: float = 0.0
            refined_json_state: dict | None = None
            lost_sight_retreat_count: int = 0

            # Main loop: track at camera FPS, detect+reason on timers.
            step = 0
            while step < cfg.max_agent_steps:
                step += 1

                # Acquire fresh frame each step (keeps loop responsive).
                obs_loop = None
                if cfg.dry_run:
                    # In dry-run we keep using the same image unless user changes it.
                    rgb_curr = rgb_np
                    depth_curr = depth_np
                else:
                    obs_loop = robot.get_observation()
                    rgb_curr = (
                        np.asarray(obs_loop.get(camera_key))
                        if obs_loop.get(camera_key) is not None
                        else np.zeros((480, 640, 3), dtype=np.uint8)
                    )
                    depth_curr = (
                        np.asarray(obs_loop.get(f"{camera_key}_depth"))
                        if obs_loop.get(f"{camera_key}_depth") is not None
                        else np.zeros((480, 640), dtype=np.uint16)
                    )
                    if mount == "gripper":
                        joints_deg = np.array(
                            [float(obs_loop[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
                            dtype=np.float64,
                        )
                        T_base_ee = kinematics.forward_kinematics(joints_deg)
                        planner.set_camera_to_robot_tf(T_base_ee @ T_ee_cam)

                depth_history.append(np.asarray(depth_curr, dtype=np.uint16).copy())
                if cfg.depth_fusion_frames > 1:
                    depth_fused = fuse_depth_temporal_median(list(depth_history))
                else:
                    depth_fused = depth_history[-1]

                now = time.time()
                img_h, img_w = rgb_curr.shape[:2]

                if (cfg.display_data and rgb_curr.size > 0) or cfg.display_sim3d:
                    rerun_frame_idx += 1
                if cfg.display_data and rgb_curr.size > 0:
                    log_stereo_pair_to_rerun(
                        camera_key=camera_key,
                        rgb_hwc=rgb_curr,
                        depth_u16_hw=depth_fused,
                        frame_sequence=rerun_frame_idx,
                        compress_images=display_compressed,
                    )

                # Tracking update (fast).
                if cfg.enable_tracking and tracker is not None and tracked_bbox_xyxy is not None:
                    try:
                        ok, bbox_xywh = tracker.update(cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2BGR))
                    except Exception:
                        ok, bbox_xywh = False, None
                    if ok and bbox_xywh is not None:
                        tracked_bbox_xyxy = _clamp_xyxy(_xywh_to_xyxy(tuple(map(int, bbox_xywh))), img_w, img_h)
                    else:
                        tracker = None
                        tracked_bbox_xyxy = None

                pause_vlm = (
                    not bool(cfg.vlm_every_agent_step)
                    and bool(cfg.micro_pause_vlm_during_motion)
                    and step_mode == "micro"
                    and active_waypoints is not None
                    and active_wp_i < len(active_waypoints)
                )
                vlm_due = bool(cfg.vlm_every_agent_step) or (
                    (now - last_detect_t) >= float(cfg.detect_period_s) or tracked_bbox_xyxy is None
                )
                # Periodic detection (slow), or every step if vlm_every_agent_step.
                if (not pause_vlm) and vlm_due:
                    scene, detections_with_state, _, _ = _observe_from_images(
                        rgb_curr,
                        depth_fused,
                        detector,
                        compute_object_state,
                        intrinsics,
                        cfg.task,
                        color_filter_min_fraction=color_fn,
                    )

                    # Initialize tracker from first usable detection (or keep existing if we already track).
                    if cfg.enable_tracking and tracker is None and detections_with_state:
                        det0, _state0 = detections_with_state[0]
                        tracked_bbox_xyxy = _clamp_xyxy(det0.bbox_xyxy, img_w, img_h)
                        tracked_label = det0.label
                        tr = _make_tracker()
                        if tr is not None:
                            tracker = tr
                            try:
                                tracker.init(
                                    cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2BGR),
                                    _xyxy_to_xywh(tracked_bbox_xyxy),
                                )
                            except Exception:
                                tracker = None
                        else:
                            tracker = None

                    if (
                        scene.is_empty()
                        and cfg.enable_tracking
                        and tracked_bbox_xyxy is not None
                    ):
                        fb_scene, fb_dets = _scene_from_tracker_bbox(
                            rgb_curr,
                            depth_fused,
                            tracked_bbox_xyxy,
                            tracked_label,
                            compute_object_state,
                            intrinsics,
                            cfg.task,
                        )
                        if not fb_scene.is_empty():
                            logger.info(
                                "[perceive] tracker fallback (VLM had no valid 3D) bbox=%s",
                                tracked_bbox_xyxy,
                            )
                            scene, detections_with_state = fb_scene, fb_dets

                    last_detect_t = now
                else:
                    # Build a minimal scene from tracker bbox + depth if we didn't run detection this tick.
                    if tracked_bbox_xyxy is not None:
                        scene, detections_with_state = _scene_from_tracker_bbox(
                            rgb_curr,
                            depth_fused,
                            tracked_bbox_xyxy,
                            tracked_label,
                            compute_object_state,
                            intrinsics,
                            cfg.task,
                        )
                    else:
                        scene = SceneObservation(objects=[], task=cfg.task)
                        detections_with_state = []

                detections_with_state, scene = _filter_detections_by_cam_distance(
                    detections_with_state, cfg.task, float(cfg.max_object_cam_distance_m)
                )
                raw_manual = (cfg.manual_object_xyz or "").strip()
                if raw_manual and not detections_with_state:
                    class _ManualDet:
                        def __init__(self, w: int, h: int):
                            self.label = "manual_object"
                            self.bbox_xyxy = (
                                max(0, w // 4),
                                max(0, h // 4),
                                min(w, 3 * w // 4),
                                min(h, 3 * h // 4),
                            )

                    sz_in = (cfg.manual_object_size_xyz or "").strip()
                    if sz_in:
                        sz = parse_xyz_string(sz_in).astype(np.float32)
                    else:
                        sz = np.array([0.04, 0.04, 0.04], dtype=np.float32)
                    ph = np.array([0.1, 0.1, 0.4], dtype=np.float32)
                    st0 = {
                        "obj_center_xyz": ph,
                        "obj_size_xyz": sz,
                        "obj_distance": np.array([float(np.linalg.norm(ph))], dtype=np.float32),
                    }
                    detections_with_state = [(_ManualDet(img_w, img_h), st0)]
                    scene = _scene_from_det_pairs(detections_with_state, cfg.task)
                    logger.info("[perceive] using --manual-object-* placeholder (distance filter empty)")

                if not scene.is_empty():
                    lost_sight_retreat_count = 0

                # Visualization.
                if cfg.show_viz and rgb_curr.size > 0:
                    try:
                        vis_rgb = cv2.cvtColor(rgb_curr.copy(), cv2.COLOR_RGB2BGR)
                        for i, (det, state) in enumerate(detections_with_state):
                            x1, y1, x2, y2 = det.bbox_xyxy
                            x1, y1, x2, y2 = _clamp_xyxy((x1, y1, x2, y2), img_w, img_h)
                            color = (0, 255, 0) if i == 0 else (0, 200, 255)
                            cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)
                            d_m = float(state["obj_distance"][0]) if "obj_distance" in state else 0.0
                            cv2.putText(
                                vis_rgb,
                                f"{det.label} d={d_m:.2f}m",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )
                        if tracked_bbox_xyxy is not None and not detections_with_state:
                            x1, y1, x2, y2 = tracked_bbox_xyxy
                            cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(
                                vis_rgb,
                                "TRACK",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2,
                            )

                        depth_vis = colorize_depth(depth_fused)
                        depth_resized = cv2.resize(depth_vis, (img_w, img_h))
                        combined = np.hstack([vis_rgb, depth_resized])
                        cv2.imshow("Agentic - RGB | Depth", combined)
                        cv2.waitKey(1)
                    except Exception:
                        pass

                # Periodic reasoning (slow), or every step if llm_every_agent_step.
                should_reason = (
                    bool(cfg.llm_every_agent_step)
                    or force_reason_next
                    or ((now - last_reason_t) >= float(cfg.reason_period_s))
                )
                if cfg.log_loop_tick_debug:
                    dt_llm_s = (now - last_reason_t) if last_reason_t > 0 else None
                    micro_tail = ""
                    if step_mode == "micro" and active_waypoints is not None:
                        micro_tail = f" micro={active_action} wp={active_wp_i}/{len(active_waypoints)}"
                    logger.debug(
                        "loop_tick step=%d mode=%s should_reason=%s force_reason_next=%s "
                        "dt_since_llm=%s reason_period_s=%.2f detect_due_in=%.2fs n_objects=%d%s",
                        step,
                        step_mode,
                        should_reason,
                        force_reason_next,
                        f"{dt_llm_s:.2f}s" if dt_llm_s is not None else "never",
                        float(cfg.reason_period_s),
                        max(0.0, float(cfg.detect_period_s) - (now - last_detect_t)),
                        len(scene.objects),
                        micro_tail,
                    )

                if should_reason:
                    if (
                        cfg.recover_on_lost_sight
                        and not cfg.dry_run
                        and step_mode == "micro"
                        and scene.is_empty()
                        and active_waypoints is not None
                        and active_action == "pick"
                        and active_wp_i > 0
                        and lost_sight_retreat_count < int(cfg.lost_sight_max_retreats_per_pick)
                    ):
                        lost_sight_retreat_count += 1
                        try:
                            active_wp_i, active_wp_micro_i = _micro_retreat_previous_waypoint(
                                robot,
                                kinematics,
                                active_waypoints,
                                SO100_MOTOR_NAMES,
                                motion_exec,
                                active_wp_i,
                            )
                            logger.warning(
                                "[agentic] lost sight (empty scene) → retreated to wp %d/%d %s (recovery %d/%d)",
                                active_wp_i + 1,
                                len(active_waypoints),
                                getattr(active_waypoints[active_wp_i], "label", "?"),
                                lost_sight_retreat_count,
                                int(cfg.lost_sight_max_retreats_per_pick),
                            )
                        except Exception as e:
                            logger.warning("[motion] lost-sight retreat failed: %s", e)
                        force_reason_next = True
                        continue

                    llm_trigger = "forced" if force_reason_next else "timer"
                    force_reason_next = False
                    dt_since_llm = (now - last_reason_t) if last_reason_t > 0 else None
                    n_scene_before = len(scene.objects)
                    if reason_strategy == "json_state":
                        if not detections_with_state:
                            logger.debug("json_state: empty detections; LLM still called")
                        payload = build_vlm_scene_json_payload(cfg.task, detections_with_state)
                        if refined_json_state is not None:
                            payload["previous_model_state"] = refined_json_state
                        action, parsed_state = reasoning_agent.reason_json_state(payload)
                        if parsed_state and isinstance(parsed_state.get("objects"), list):
                            refined_json_state = {
                                "task": cfg.task,
                                "objects": list(parsed_state["objects"]),
                            }
                    else:
                        action = reasoning_agent.reason(scene, scene_text_override=None)
                    last_action = action
                    last_reason_t = now

                    # Structured logging for obs + LLM.
                    try:
                        rec: dict = {
                            "ts": time.time(),
                            "attempt": attempt + 1,
                            "step": step,
                            "scene": {
                                "task": scene.task,
                                "objects": [
                                    {
                                        "index": o.index,
                                        "label": o.label,
                                        "center_xyz": list(o.center_xyz),
                                        "size_xyz": list(o.size_xyz),
                                        "distance_m": o.distance_m,
                                    }
                                    for o in scene.objects
                                ],
                            },
                            "agent": {
                                "model": cfg.agent.model,
                                "action": {
                                    "action": action.action,
                                    "object_index": action.object_index,
                                    "place_xyz": list(action.place_xyz) if action.place_xyz is not None else None,
                                    "success": action.success,
                                    "reason": action.reason,
                                },
                            },
                        }
                        if reason_strategy == "json_state":
                            rec["agent"]["reasoning_strategy"] = "json_state"
                            if refined_json_state is not None:
                                rec["agent"]["refined_json_state"] = refined_json_state
                        if cfg.log_scene_text:
                            rec["agent"]["scene_text"] = reasoning_agent.last_scene_text
                        if cfg.log_llm_raw:
                            rec["agent"]["raw_response_text"] = reasoning_agent.last_raw_response_text
                        jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                    except Exception as e:
                        logger.debug("Failed to write JSONL log: %s", e)

                    dt_part = "first" if dt_since_llm is None else f"{dt_since_llm:.1f}s"
                    act_tail = action.action
                    if action.action == "done" and action.success is not None:
                        act_tail += f" success={action.success}"
                    if action.object_index is not None:
                        act_tail += f" #{action.object_index}"
                    if action.place_xyz is not None:
                        px, py, pz = action.place_xyz
                        act_tail += f" place=({px:.3f},{py:.3f},{pz:.3f})"
                    if action.reason:
                        act_tail += f" — {action.reason}"
                    log_line = (
                        f"step {step}/{cfg.max_agent_steps} {step_mode} │ "
                        f"LLM {llm_trigger} Δt={dt_part} │ scene n={n_scene_before} │ → {act_tail}"
                    )
                    _lvl = logging.INFO
                    if action.action == "fail" or (n_scene_before == 0 and action.action in ("pick", "place")):
                        _lvl = logging.WARNING
                    logger.log(_lvl, "[agentic] %s", log_line)

                    if action.action == "done":
                        _log_agentic_sim3d_once(
                            cfg,
                            dry_run=cfg.dry_run,
                            obs_loop=obs_loop,
                            kinematics=kinematics,
                            planner=planner,
                            detections_with_state=detections_with_state,
                            rerun_frame_idx=rerun_frame_idx,
                            last_plan_ee_positions=last_plan_ee_positions,
                            last_plan_summary=last_plan_summary,
                            active_object_index=active_object_index,
                        )
                        break
                    if action.action == "fail":
                        if (
                            cfg.recover_on_lost_sight
                            and not cfg.dry_run
                            and step_mode == "micro"
                            and n_scene_before == 0
                            and active_waypoints is not None
                            and active_action == "pick"
                            and active_wp_i > 0
                            and lost_sight_retreat_count < int(cfg.lost_sight_max_retreats_per_pick)
                        ):
                            lost_sight_retreat_count += 1
                            try:
                                active_wp_i, active_wp_micro_i = _micro_retreat_previous_waypoint(
                                    robot,
                                    kinematics,
                                    active_waypoints,
                                    SO100_MOTOR_NAMES,
                                    motion_exec,
                                    active_wp_i,
                                )
                                logger.warning(
                                    "[agentic] LLM fail on empty scene → retreated to wp %d/%d %s (%d/%d)",
                                    active_wp_i + 1,
                                    len(active_waypoints),
                                    getattr(active_waypoints[active_wp_i], "label", "?"),
                                    lost_sight_retreat_count,
                                    int(cfg.lost_sight_max_retreats_per_pick),
                                )
                            except Exception as e:
                                logger.warning("[motion] retreat after LLM fail failed: %s", e)
                            force_reason_next = True
                            continue
                        _log_agentic_sim3d_once(
                            cfg,
                            dry_run=cfg.dry_run,
                            obs_loop=obs_loop,
                            kinematics=kinematics,
                            planner=planner,
                            detections_with_state=detections_with_state,
                            rerun_frame_idx=rerun_frame_idx,
                            last_plan_ee_positions=last_plan_ee_positions,
                            last_plan_summary=last_plan_summary,
                            active_object_index=active_object_index,
                        )
                        break
                    if action.action == "retry":
                        refined_json_state = None
                        tracker = None
                        tracked_bbox_xyxy = None
                        _log_agentic_sim3d_once(
                            cfg,
                            dry_run=cfg.dry_run,
                            obs_loop=obs_loop,
                            kinematics=kinematics,
                            planner=planner,
                            detections_with_state=detections_with_state,
                            rerun_frame_idx=rerun_frame_idx,
                            last_plan_ee_positions=last_plan_ee_positions,
                            last_plan_summary=last_plan_summary,
                            active_object_index=active_object_index,
                        )
                        continue

                    if action.is_pick():
                        idx = int(action.object_index or 0)
                        if idx < 0 or idx >= len(detections_with_state):
                            logger.warning(
                                "Invalid object_index %s (have %d objects).",
                                idx,
                                len(detections_with_state),
                            )
                            _log_agentic_sim3d_once(
                                cfg,
                                dry_run=cfg.dry_run,
                                obs_loop=obs_loop,
                                kinematics=kinematics,
                                planner=planner,
                                detections_with_state=detections_with_state,
                                rerun_frame_idx=rerun_frame_idx,
                                last_plan_ee_positions=last_plan_ee_positions,
                                last_plan_summary=last_plan_summary,
                                active_object_index=active_object_index,
                            )
                            continue
                        det, state = detections_with_state[idx]

                        center = np.asarray(state["obj_center_xyz"], dtype=np.float32).reshape(3)
                        size = np.asarray(state["obj_size_xyz"], dtype=np.float32).reshape(3)
                        if reason_strategy == "json_state":
                            rp = _refined_object_pose_from_json_state(refined_json_state, idx)
                            if rp is not None:
                                center, size = rp
                                logger.info(
                                    "Pick geometry from json_state refinement (object %d): center=%s size=%s",
                                    idx,
                                    center,
                                    size,
                                )

                        # Optionally re-init tracker to the agent-chosen object.
                        if cfg.enable_tracking and cfg.track_agent_choice and hasattr(det, "bbox_xyxy"):
                            tracked_bbox_xyxy = _clamp_xyxy(det.bbox_xyxy, img_w, img_h)
                            tracked_label = det.label
                            tr = _make_tracker()
                            if tr is not None:
                                tracker = tr
                                try:
                                    tracker.init(
                                        cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2BGR),
                                        _xyxy_to_xywh(tracked_bbox_xyxy),
                                    )
                                except Exception:
                                    tracker = None

                        center, size = _apply_manual_object_geometry(cfg, planner, idx, center, size)
                        waypoints = planner.plan_pick(center, size, det.label)
                        last_plan_ee_positions = np.stack(
                            [np.asarray(w.pose_4x4, dtype=np.float64)[:3, 3] for w in waypoints],
                            axis=0,
                        )
                        last_plan_summary = (
                            f"Pick '{det.label}' → " + " → ".join(w.label for w in waypoints)
                        )
                        if cfg.dry_run:
                            _log_agentic_sim3d_once(
                                cfg,
                                dry_run=True,
                                obs_loop=obs_loop,
                                kinematics=kinematics,
                                planner=planner,
                                detections_with_state=detections_with_state,
                                rerun_frame_idx=rerun_frame_idx,
                                last_plan_ee_positions=last_plan_ee_positions,
                                last_plan_summary=last_plan_summary,
                                active_object_index=idx,
                            )
                            logger.info(
                                "[dry run] Would execute pick: object %d '%s' at %s",
                                idx,
                                det.label,
                                center,
                            )
                            break

                        if step_mode == "macro":
                            cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
                            logger.info(
                                "[motion] pick %r │ base (%.3f,%.3f,%.3f) │ macro ×%d wps",
                                det.label,
                                cx,
                                cy,
                                cz,
                                len(waypoints),
                            )
                            execute_waypoints(robot, kinematics, waypoints, SO100_MOTOR_NAMES, motion_exec)
                            logger.debug("[motion] macro pick sequence done")
                            if cfg.place_target:
                                place_xyz = np.array(
                                    [float(v) for v in cfg.place_target.split(",")],
                                    dtype=np.float64,
                                )
                                place_wps = planner.plan_place(place_xyz)
                                last_plan_ee_positions = np.stack(
                                    [np.asarray(w.pose_4x4, dtype=np.float64)[:3, 3] for w in place_wps],
                                    axis=0,
                                )
                                last_plan_summary = "Place after pick: " + " → ".join(w.label for w in place_wps)
                                execute_waypoints(robot, kinematics, place_wps, SO100_MOTOR_NAMES, motion_exec)
                            _log_agentic_sim3d_once(
                                cfg,
                                dry_run=cfg.dry_run,
                                obs_loop=obs_loop,
                                kinematics=kinematics,
                                planner=planner,
                                detections_with_state=detections_with_state,
                                rerun_frame_idx=rerun_frame_idx,
                                last_plan_ee_positions=last_plan_ee_positions,
                                last_plan_summary=last_plan_summary,
                                active_object_index=idx,
                            )
                            break

                        # micro: execute exactly one waypoint, then return to agent.
                        if (
                            active_action != "pick"
                            or active_object_index != idx
                            or active_waypoints is None
                            or active_wp_i >= len(active_waypoints)
                        ):
                            active_action = "pick"
                            active_object_index = idx
                            active_object_label = det.label
                            active_waypoints = waypoints
                            active_wp_i = 0
                            active_wp_micro_i = 0
                        wp = active_waypoints[active_wp_i]
                        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
                        logger.info(
                            "[motion] pick %r │ base (%.3f,%.3f,%.3f) │ wp %d/%d %s",
                            det.label,
                            cx,
                            cy,
                            cz,
                            active_wp_i + 1,
                            len(active_waypoints),
                            getattr(wp, "label", "?"),
                        )
                        max_ms = int(max(1, cfg.micro_max_microsteps_per_decision))
                        _j, _mi, _done = execute_waypoint_microsteps(
                            robot,
                            kinematics,
                            wp,
                            SO100_MOTOR_NAMES,
                            motion_exec,
                            current_joints=None,
                            microstep_index=active_wp_micro_i,
                            max_microsteps=max_ms,
                        )
                        active_wp_micro_i = int(_mi)
                        try:
                            rec2 = {
                                "ts": time.time(),
                                "attempt": attempt + 1,
                                "step": step,
                                "execution": {
                                    "mode": "micro",
                                    "action": "pick",
                                    "object_index": active_object_index,
                                    "object_label": active_object_label,
                                    "waypoint_index": active_wp_i,
                                    "waypoint_label": getattr(wp, "label", "waypoint"),
                                    "waypoint_microstep_index": active_wp_micro_i,
                                    "max_microsteps_per_decision": max_ms,
                                },
                            }
                            jsonl_f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                            jsonl_f.flush()
                        except Exception:
                            pass
                        if _done:
                            active_wp_i += 1
                            active_wp_micro_i = 0
                        pick_focus_for_viz = active_object_index
                        if active_waypoints is not None and active_wp_i >= len(active_waypoints):
                            logger.debug("[motion] micro pick waypoint finished")
                            active_action = None
                            active_waypoints = None
                            active_object_index = None
                            active_object_label = ""
                            # Optional automatic place after micro pick is kept as macro behavior only.
                        if cfg.micro_reason_each_waypoint:
                            force_reason_next = True
                        else:
                            force_reason_next = active_waypoints is None
                        _log_agentic_sim3d_once(
                            cfg,
                            dry_run=cfg.dry_run,
                            obs_loop=obs_loop,
                            kinematics=kinematics,
                            planner=planner,
                            detections_with_state=detections_with_state,
                            rerun_frame_idx=rerun_frame_idx,
                            last_plan_ee_positions=last_plan_ee_positions,
                            last_plan_summary=last_plan_summary,
                            active_object_index=pick_focus_for_viz,
                        )
                        continue

                    if action.is_place():
                        xyz = action.place_xyz
                        if xyz is not None:
                            place_xyz_np = np.array(xyz, dtype=np.float64)
                            place_wps = planner.plan_place(place_xyz_np)
                            last_plan_ee_positions = np.stack(
                                [np.asarray(w.pose_4x4, dtype=np.float64)[:3, 3] for w in place_wps],
                                axis=0,
                            )
                            last_plan_summary = "Place: " + " → ".join(w.label for w in place_wps)
                            if cfg.dry_run:
                                _log_agentic_sim3d_once(
                                    cfg,
                                    dry_run=True,
                                    obs_loop=obs_loop,
                                    kinematics=kinematics,
                                    planner=planner,
                                    detections_with_state=detections_with_state,
                                    rerun_frame_idx=rerun_frame_idx,
                                    last_plan_ee_positions=last_plan_ee_positions,
                                    last_plan_summary=last_plan_summary,
                                    active_object_index=active_object_index,
                                )
                                logger.info("[dry run] Would execute place at %s", xyz)
                                break
                            execute_waypoints(robot, kinematics, place_wps, SO100_MOTOR_NAMES, motion_exec)
                            _log_agentic_sim3d_once(
                                cfg,
                                dry_run=cfg.dry_run,
                                obs_loop=obs_loop,
                                kinematics=kinematics,
                                planner=planner,
                                detections_with_state=detections_with_state,
                                rerun_frame_idx=rerun_frame_idx,
                                last_plan_ee_positions=last_plan_ee_positions,
                                last_plan_summary=last_plan_summary,
                                active_object_index=active_object_index,
                            )
                        break

                else:
                    # No LLM call this tick: optional heartbeat so the terminal shows why.
                    iv = float(cfg.log_loop_status_interval_s)
                    if iv > 0 and last_reason_t > 0:
                        if last_loop_status_log_t <= 0.0:
                            last_loop_status_log_t = now
                        elif (now - last_loop_status_log_t) >= iv:
                            last_loop_status_log_t = now
                            dt_llm = now - last_reason_t
                            next_in = max(0.0, float(cfg.reason_period_s) - dt_llm)
                            micro_s = ""
                            if step_mode == "micro" and active_waypoints is not None:
                                micro_s = (
                                    f" active_micro={active_action} wp={active_wp_i}/{len(active_waypoints)}"
                                )
                            logger.info(
                                "Loop status: mode=%s (no LLM this tick) last_llm_ago=%.2fs "
                                "next_llm_in≈%.2fs (reason_period_s=%.2f) n_objects=%d%s",
                                step_mode,
                                dt_llm,
                                next_in,
                                float(cfg.reason_period_s),
                                len(scene.objects),
                                micro_s,
                            )

                _log_agentic_sim3d_once(
                    cfg,
                    dry_run=cfg.dry_run,
                    obs_loop=obs_loop,
                    kinematics=kinematics,
                    planner=planner,
                    detections_with_state=detections_with_state,
                    rerun_frame_idx=rerun_frame_idx,
                    last_plan_ee_positions=last_plan_ee_positions,
                    last_plan_summary=last_plan_summary,
                    active_object_index=active_object_index,
                )

                # If not reasoning this tick, keep loop responsive but don't busy-spin.
                if not cfg.dry_run:
                    time.sleep(0.03)

            else:
                logger.info("[agentic] max steps (%d) reached", cfg.max_agent_steps)
            break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            jsonl_f.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if cfg.display_data or cfg.display_sim3d:
            try:
                rr.rerun_shutdown()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                logger.warning("[agentic] disconnect (bus may already be dead): %s", e)


def main():
    agentic_manipulate()


if __name__ == "__main__":
    main()
