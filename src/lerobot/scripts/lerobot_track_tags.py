#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    so_follower,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class TagTrackingConfig:
    robot: RobotConfig = field(default_factory=lambda: RobotConfig(type="so100_follower"))
    camera_key: str = "front"

    dictionary: str = "apriltag_36h11"
    # Comma-separated tag IDs to report (empty = report all)
    tag_ids: str = ""
    # Physical edge length of the black square (meters). If > 0, estimate 6D pose via PnP (robot-arm tag).
    tag_size_m: float = 0.0
    # Axis length for pose visualization (meters). Only used when tag_size_m > 0 and show_viz.
    pose_axes_length_m: float = 0.04
    # Depth median window size (pixels)
    depth_window: int = 7
    # UI
    show_viz: bool = True
    # Throttle JSON output (seconds). Set 0 to print every frame.
    print_period_s: float = 0.2


def _parse_id_list(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


@parser.wrap()
def track_tags(cfg: TagTrackingConfig):
    init_logging()

    from lerobot.perception.robot_arm_pose_tracker import (
        default_distortion_coeffs,
        draw_pose_axes,
        estimate_tag_pose6d,
        intrinsics_dict_to_camera_matrix,
    )
    from lerobot.perception.tag_tracker import TagTracker, depth_at_pixel_m, pixel_to_camera_xyz

    tag_filter = _parse_id_list(cfg.tag_ids)

    logger.info("Connecting robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    cam = robot.cameras.get(cfg.camera_key)
    if cam is None:
        raise ValueError(f"Camera '{cfg.camera_key}' not found. Available: {list(robot.cameras.keys())}")

    intrinsics: dict[str, float] = {}
    if hasattr(cam, "get_depth_intrinsics"):
        intrinsics = cam.get_depth_intrinsics()
        logger.info("Depth intrinsics: %s", intrinsics)
    else:
        raise RuntimeError("Camera does not expose get_depth_intrinsics(); cannot backproject to 3D.")

    K = intrinsics_dict_to_camera_matrix(intrinsics) if cfg.tag_size_m > 0 else None
    dist_pose = default_distortion_coeffs()

    tracker = TagTracker(dictionary=cfg.dictionary)
    last_print = 0.0
    if cfg.tag_size_m > 0:
        logger.info(
            "6D pose (PnP) enabled: tag_size_m=%s m — use same dictionary as printed tag; "
            "measure the black square edge length.",
            cfg.tag_size_m,
        )

    try:
        logger.info("Starting tag tracking. Press Ctrl+C to stop.")
        while True:
            obs = robot.get_observation()
            rgb = obs.get(cfg.camera_key)
            depth = obs.get(f"{cfg.camera_key}_depth")
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            rgb_np = np.asarray(rgb)
            depth_np = np.asarray(depth)
            detections = tracker.detect(rgb_np)

            if detections:
                logger.info("Detected tag IDs (pre-filter): %s", [d.tag_id for d in detections])

            # Compute depth-based XYZ and optional PnP 6D pose (fiducial on robot arm).
            out = []
            for d in detections:
                if tag_filter and d.tag_id not in tag_filter:
                    continue
                u, v = d.center_px
                z_m = depth_at_pixel_m(depth_np, u, v, intrinsics, window=int(cfg.depth_window))
                xyz = pixel_to_camera_xyz(u, v, z_m, intrinsics) if z_m > 0 else np.zeros(3, dtype=np.float64)
                entry: dict = {
                    "tag_id": d.tag_id,
                    "center_px": [u, v],
                    "depth_m": z_m,
                    "xyz_m": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "bbox_xyxy": list(map(int, d.bbox_xyxy)),
                }
                if K is not None and cfg.tag_size_m > 0:
                    pose = estimate_tag_pose6d(d, float(cfg.tag_size_m), K, dist_pose)
                    if pose is not None:
                        entry["pose6d"] = pose.to_jsonable()
                out.append(entry)

            now = time.time()
            if cfg.print_period_s <= 0 or (now - last_print) >= float(cfg.print_period_s):
                payload = {
                    "t": now,
                    "dictionary": cfg.dictionary,
                    "camera_key": cfg.camera_key,
                    "tags": out,
                }
                if tag_filter and detections and not out:
                    payload["note"] = f"Tags detected but filtered out by tag_ids={sorted(tag_filter)}"
                logger.info("TAG_JSON %s", json.dumps(payload))
                last_print = now

            if cfg.show_viz:
                try:
                    vis = cv2.cvtColor(rgb_np.copy(), cv2.COLOR_RGB2BGR)
                    for t in out:
                        x1, y1, x2, y2 = t["bbox_xyxy"]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        err = ""
                        p6 = t.get("pose6d")
                        if p6 is not None:
                            err = f" reproj={p6['reproj_error_px']:.2f}px"
                        txt = f"id={t['tag_id']} z={t['depth_m']:.2f}m{err}"
                        cv2.putText(
                            vis,
                            txt,
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        if K is not None and p6 is not None:
                            rvec = np.asarray(p6["rvec"], dtype=np.float64).reshape(3, 1)
                            tvec = np.asarray(p6["tvec_m"], dtype=np.float64).reshape(3, 1)
                            draw_pose_axes(vis, K, dist_pose, rvec, tvec, float(cfg.pose_axes_length_m))
                    cv2.imshow("Tag tracking (RGB)", vis)
                    cv2.waitKey(1)
                except Exception:
                    pass

            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        robot.disconnect()


def main():
    track_tags()


if __name__ == "__main__":
    main()

