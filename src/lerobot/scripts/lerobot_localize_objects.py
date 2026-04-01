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
Natural-language 3D localization using stereo depth (e.g. OAK-D) + VLM + depth backprojection.

Runs a loop: RGB+depth → VLM(query) → masks → 3D centers/sizes → optional robot-base frame,
temporal smoothing, and JSON log. Does not move the arm.

Examples::

    # OAK-D only (no robot), Gemini + depth; 3D in camera frame only
    export GEMINI_API_KEY=...
    lerobot-localize-objects --camera_only=true --include_base_frame=false \\
        --oakd_fps=30 --oakd_width=640 --oakd_height=480 \\
        --query="the red cube" \\
        --vlm.backend=gemini

    # Same pipeline through SO-100 + front OAK-D (uses robot.cameras)
    lerobot-localize-objects \\
        --robot.type=so100_follower \\
        --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \\
        --query="pen on the desk" \\
        --camera_frame_convention=opencv \\
        --camera_to_robot_tf="0.4,0,0.1,0,0,0"

    # Dry run on a saved image (+ optional depth .npy)
    lerobot-localize-objects --dry_run=true --dry_run_image=scene.png --query="mug"

Tuning::

    --depth_fusion_frames=11 --color_filter_min_fraction=0.12 \\
    --oakd_stereo_preset=HIGH_DETAIL --oakd_stereo_confidence=180
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lerobot.agent import ReasoningAgent, SceneObservation
from lerobot.cameras.oakd.camera_oakd import OAKDCamera
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
from lerobot.configs import parser
from lerobot.perception.object_localization import (
    LOCALIZE_DISAMBIGUATION_SYSTEM_PROMPT,
    LocalizedObjectSnapshot,
    TemporalObjectMap,
    build_scene_summary_renumbered,
    select_focus_detection_index,
    snapshots_from_scene,
)
from lerobot.perception.scene_from_rgb_depth import build_scene_from_rgb_depth
from lerobot.perception.vlm_detector import VLMDetector
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig  # noqa: F401 — registers so100_follower choice
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


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
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    base_url: str = ""
    use_gemini: bool = True


@dataclass
class LocalizeObjectsConfig:
    # Concrete default (``RobotConfig(type=...)`` is invalid on the abstract registry class).
    robot: RobotConfig = field(
        default_factory=lambda: SOFollowerRobotConfig(port="", cameras={}),
    )
    vlm: VLMConfig = field(default_factory=VLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Natural-language object(s) to find (passed to the VLM as the detection prompt).
    query: str = "the red cube"

    camera_key: str = "front"
    camera_to_robot_tf: str = "0,0,0.4,0,0,0"
    camera_frame_convention: str = "opencv"
    camera_flip_lateral: bool = False
    # If false, skip ``center_base_m`` / ``size_base_m`` (camera-frame-only 3D).
    include_base_frame: bool = True

    # ``single``: one object for the map (heuristic or LLM pick). ``all``: every valid detection.
    focus_mode: str = "single"
    # For ``single`` when multiple detections: closest | largest_bbox | central | first | llm
    single_strategy: str = "closest"

    smooth_alpha: float = 0.35
    period_s: float = 0.5
    max_frames: int = 0
    max_duration_s: float = 0.0

    output_jsonl: str = ""
    print_json: bool = True
    show_viz: bool = True

    dry_run: bool = False
    dry_run_image: str = ""
    dry_run_depth: str = ""

    # Use an OAK-D directly (no robot). Ignores ``robot`` except for type registration side effects.
    # Flat fields (not a nested ``OAKDCameraConfig``): draccus/argparse cannot use ``int | None`` types
    # from ``CameraConfig`` as CLI ``type=``.
    camera_only: bool = False
    oakd_fps: int = 30
    oakd_width: int = 640
    oakd_height: int = 480
    oakd_use_depth: bool = True
    oakd_device_id: str = ""
    oakd_warmup_s: int = 2
    oakd_stereo_preset: str = "FAST_ACCURACY"
    # 0--255, higher = stricter; -1 = do not set (SDK default)
    oakd_stereo_confidence: int = 180

    # Median fuse this many consecutive depth frames (>=2 activates fusion).
    depth_fusion_frames: int = 7
    # If > 0 and the query names a color, drop VLM boxes with too few matching HSV pixels.
    color_filter_min_fraction: float = 0.08


def parse_tf_string(tf_str: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    parts = [float(v.strip()) for v in tf_str.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(parts)}: {tf_str}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = parts[:3]
    if any(abs(v) > 1e-8 for v in parts[3:6]):
        tf[:3, :3] = Rotation.from_rotvec(parts[3:6]).as_matrix()
    return tf


def camera_opencv_to_robot_rotation(flip_lateral: bool = False) -> np.ndarray:
    R = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    if flip_lateral:
        R[1, 0] *= -1
    return R


def colorize_depth(depth_mm: np.ndarray, max_range_mm: int = 2000) -> np.ndarray:
    depth = np.asarray(depth_mm)
    clipped = np.clip(depth.astype(np.float32), 0, max_range_mm)
    normalized = (clipped / max_range_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _snapshot_to_dict(s: LocalizedObjectSnapshot) -> dict:
    d = {
        "label": s.label,
        "detection_index": s.detection_index,
        "center_cam_m": list(s.center_cam_m),
        "size_cam_m": list(s.size_cam_m),
        "distance_m": s.distance_m,
        "bbox_xyxy": list(s.bbox_xyxy),
    }
    if s.center_base_m is not None:
        d["center_base_m"] = list(s.center_base_m)
    if s.size_base_m is not None:
        d["size_base_m"] = list(s.size_base_m)
    return d


@parser.wrap()
def localize_objects(cfg: LocalizeObjectsConfig) -> None:
    init_logging()
    from lerobot.processor.depth_perception_processor import (
        compute_object_state,
        fuse_depth_temporal_median,
    )

    cam_to_robot = parse_tf_string(cfg.camera_to_robot_tf)
    if (cfg.camera_frame_convention or "").lower() == "opencv":
        t = cam_to_robot[:3, 3].copy()
        R = camera_opencv_to_robot_rotation(flip_lateral=cfg.camera_flip_lateral)
        cam_to_robot = np.eye(4, dtype=np.float64)
        cam_to_robot[:3, :3] = R
        cam_to_robot[:3, 3] = t

    tf_for_objects: np.ndarray | None = cam_to_robot if cfg.include_base_frame else None

    detector = VLMDetector(
        backend=cfg.vlm.backend,
        model_id=cfg.vlm.model_id,
        device=cfg.vlm.device or None,
        api_key=cfg.vlm.api_key or None,
        cloud_model=cfg.vlm.cloud_model,
        cloud_base_url=cfg.vlm.cloud_base_url or None,
    )

    disambiguation_agent: ReasoningAgent | None = None
    if (cfg.single_strategy or "").lower() == "llm":
        disambiguation_agent = ReasoningAgent(
            api_key=cfg.agent.api_key or None,
            base_url=cfg.agent.base_url or None,
            model=cfg.agent.model,
            is_gemini=cfg.agent.use_gemini,
            system_prompt=LOCALIZE_DISAMBIGUATION_SYSTEM_PROMPT,
        )

    robot = None
    oak_cam: OAKDCamera | None = None
    intrinsics: dict[str, float] = {}
    camera_key = cfg.camera_key

    if cfg.dry_run:
        if not cfg.dry_run_image or not cfg.dry_run_image.strip():
            raise ValueError("dry_run=True requires --dry_run_image=<path> to an RGB image.")
        rgb_np = cv2.imread(cfg.dry_run_image.strip())
        if rgb_np is None:
            raise FileNotFoundError(f"Could not load image: {cfg.dry_run_image}")
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        if cfg.dry_run_depth and cfg.dry_run_depth.strip():
            depth_np = np.load(cfg.dry_run_depth.strip())
            if depth_np.ndim == 3:
                depth_np = depth_np.squeeze()
        else:
            h, w = rgb_np.shape[:2]
            depth_np = np.full((h, w), 500, dtype=np.uint16)
        h, w = rgb_np.shape[:2]
        intrinsics = {
            "fx": 525.0,
            "fy": 525.0,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "depth_scale": 0.001,
        }

        def read_frames():
            return rgb_np, depth_np

    elif cfg.camera_only:
        oak_cfg = OAKDCameraConfig(
            fps=cfg.oakd_fps,
            width=cfg.oakd_width,
            height=cfg.oakd_height,
            use_depth=cfg.oakd_use_depth,
            device_id=cfg.oakd_device_id,
            warmup_s=cfg.oakd_warmup_s,
            stereo_preset=cfg.oakd_stereo_preset,
            stereo_confidence_threshold=cfg.oakd_stereo_confidence,
        )
        oak_cam = OAKDCamera(oak_cfg)
        oak_cam.connect()
        if hasattr(oak_cam, "get_depth_intrinsics"):
            try:
                intrinsics = oak_cam.get_depth_intrinsics()
            except Exception as e:
                logger.warning("Could not read OAK-D intrinsics: %s", e)
        if not intrinsics:
            raise RuntimeError("OAK-D intrinsics unavailable; check depth is enabled on config.")

        def read_frames():
            assert oak_cam is not None
            c = np.asarray(oak_cam.read())
            d = np.asarray(oak_cam.read_depth())
            return c, d

    else:
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        depth_camera = None
        for cam_name, cam in robot.cameras.items():
            if cam_name == camera_key:
                depth_camera = cam
                break
        if depth_camera is None:
            raise ValueError(f"Camera '{camera_key}' not found. Available: {list(robot.cameras.keys())}")
        if hasattr(depth_camera, "get_depth_intrinsics"):
            try:
                intrinsics = depth_camera.get_depth_intrinsics()
            except Exception as e:
                logger.warning("Could not get depth intrinsics: %s", e)
        if not intrinsics:
            raise RuntimeError("Depth intrinsics missing; use a depth camera (e.g. OAK-D with use_depth=true).")

        def read_frames():
            assert robot is not None
            obs = robot.get_observation()
            rgb = obs.get(camera_key)
            depth = obs.get(f"{camera_key}_depth")
            if rgb is None or depth is None:
                raise RuntimeError("Missing RGB or depth in robot observation.")
            return np.asarray(rgb), np.asarray(depth)

    smoother = TemporalObjectMap(alpha=cfg.smooth_alpha)
    jsonl_path = Path(cfg.output_jsonl) if cfg.output_jsonl else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    depth_history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.depth_fusion_frames)))
    color_fn = cfg.color_filter_min_fraction if cfg.color_filter_min_fraction > 0 else None

    start = time.time()
    frame_i = 0
    logger.info("Localize loop | query=%r | focus_mode=%s | strategy=%s", cfg.query, cfg.focus_mode, cfg.single_strategy)

    try:
        while True:
            if cfg.max_frames > 0 and frame_i >= cfg.max_frames:
                break
            if cfg.max_duration_s > 0 and (time.time() - start) >= cfg.max_duration_s:
                break

            rgb_curr, depth_curr = read_frames()
            depth_history.append(np.asarray(depth_curr, dtype=np.uint16).copy())
            if cfg.depth_fusion_frames > 1:
                depth_fused = fuse_depth_temporal_median(list(depth_history))
            else:
                depth_fused = depth_history[-1]

            scene, dws = build_scene_from_rgb_depth(
                rgb_curr,
                depth_fused,
                detector,
                compute_object_state,
                intrinsics,
                cfg.query,
                color_filter_min_fraction=color_fn,
            )

            if scene.is_empty():
                payload = {
                    "frame": frame_i,
                    "time_s": round(time.time() - start, 3),
                    "query": cfg.query,
                    "objects": [],
                    "focus": None,
                }
                if cfg.print_json:
                    print(json.dumps(payload))
                if jsonl_path:
                    with jsonl_path.open("a") as jf:
                        jf.write(json.dumps(payload) + "\n")
                frame_i += 1
                time.sleep(cfg.period_s)
                continue

            focus_idx = 0
            if cfg.focus_mode == "single" and len(dws) > 1:
                strat = (cfg.single_strategy or "closest").lower()
                if strat == "llm" and disambiguation_agent is not None:
                    obs_llm = SceneObservation(objects=list(scene.objects), task=cfg.query)
                    user_text = (
                        f"OBJECT_FOCUS: {cfg.query}\n\n{build_scene_summary_renumbered(obs_llm)}"
                    )
                    action = disambiguation_agent.reason(obs_llm, scene_text_override=user_text)
                    if action.action == "pick" and action.object_index is not None:
                        focus_idx = int(action.object_index)
                        focus_idx = max(0, min(focus_idx, len(dws) - 1))
                    else:
                        focus_idx = select_focus_detection_index(
                            dws, "closest", (rgb_curr.shape[0], rgb_curr.shape[1])
                        )
                        if focus_idx is None:
                            focus_idx = 0
                else:
                    picked = select_focus_detection_index(
                        dws, strat, (rgb_curr.shape[0], rgb_curr.shape[1])
                    )
                    focus_idx = picked if picked is not None else 0

            if cfg.focus_mode == "single":
                dws_focus = [dws[focus_idx]]
                scene_focus = SceneObservation(
                    objects=[scene.objects[focus_idx]],
                    task=cfg.query,
                )
            else:
                dws_focus = dws
                scene_focus = scene

            snaps = snapshots_from_scene(scene_focus, dws_focus, tf_for_objects)
            snaps = smoother.smooth_snapshots(snaps)

            objects_out = [_snapshot_to_dict(s) for s in snaps]
            focus_out = objects_out[0] if objects_out else None

            payload = {
                "frame": frame_i,
                "time_s": round(time.time() - start, 3),
                "query": cfg.query,
                "objects": objects_out,
                "focus": focus_out,
            }
            if cfg.print_json:
                print(json.dumps(payload))
            if jsonl_path:
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(payload) + "\n")

            if cfg.show_viz and rgb_curr.size > 0:
                try:
                    vis = cv2.cvtColor(rgb_curr.copy(), cv2.COLOR_RGB2BGR)
                    img_h, img_w = vis.shape[:2]
                    for si, snap in enumerate(snaps):
                        x1, y1, x2, y2 = snap.bbox_xyxy
                        x1 = max(0, min(img_w - 1, x1))
                        y1 = max(0, min(img_h - 1, y1))
                        x2 = max(0, min(img_w - 1, x2))
                        y2 = max(0, min(img_h - 1, y2))
                        col = (0, 255, 0) if cfg.focus_mode == "all" or si == 0 else (0, 180, 255)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                        line = f"{snap.label} d={snap.distance_m:.2f}m"
                        cv2.putText(
                            vis,
                            line,
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            col,
                            1,
                        )
                    depth_vis = colorize_depth(depth_fused)
                    depth_vis = cv2.resize(depth_vis, (img_w, img_h))
                    cv2.imshow("Localize — RGB | Depth", np.hstack([vis, depth_vis]))
                    cv2.waitKey(1)
                except Exception:
                    pass

            frame_i += 1
            time.sleep(cfg.period_s)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if robot is not None:
            robot.disconnect()
        if oak_cam is not None:
            oak_cam.disconnect()


def main() -> None:
    localize_objects()


if __name__ == "__main__":
    main()
