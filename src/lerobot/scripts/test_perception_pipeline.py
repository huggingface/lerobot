#!/usr/bin/env python

"""
Test the vision/perception pipeline on an OAK-D camera — no robot arm needed.

Connects to OAK-D, captures RGB + depth, runs VLM detection, segments objects,
backprojects to 3D, and displays annotated results in an OpenCV window.

Usage:
    # Gemini (default, needs GEMINI_API_KEY)
    python -m lerobot.scripts.test_perception_pipeline --query "a red cube and a blue cup"

    # Local Florence-2 (needs GPU, downloads ~1.5GB model on first run)
    python -m lerobot.scripts.test_perception_pipeline --backend local --query "a red cube and a blue cup"

    # OpenAI cloud VLM (needs OPENAI_API_KEY)
    python -m lerobot.scripts.test_perception_pipeline --backend cloud --query "a red cube"

    # With a specific OAK-D device
    python -m lerobot.scripts.test_perception_pipeline --device-id 18443010A1B2C3D4E5

Keys:
    SPACE  - run detection on current frame
    S      - save current annotated frame to disk
    Q/ESC  - quit
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def colorize_depth(depth_mm: np.ndarray, max_range_mm: int = 1000) -> np.ndarray:
    """Convert uint16 depth (mm) to a colorized BGR image for display."""
    clipped = np.clip(depth_mm.astype(np.float32), 0, max_range_mm)
    normalized = (clipped / max_range_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def draw_detections(rgb: np.ndarray, detections, obj_states: list[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, masks, and 3D info on the image."""
    vis = rgb.copy()

    for det, state in zip(detections, obj_states, strict=False):
        x1, y1, x2, y2 = det.bbox_xyxy
        color = (0, 255, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        center = state["obj_center_xyz"]
        size = state["obj_size_xyz"]
        dist = state["obj_distance"][0]

        label_text = f"{det.label}"
        info_text = f"d={dist:.2f}m  sz={size[0]*100:.1f}x{size[1]*100:.1f}x{size[2]*100:.1f}cm"
        coord_text = f"xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"

        cv2.putText(vis, label_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis, coord_text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if det.mask is not None:
            mask_overlay = vis.copy()
            mask_bool = det.mask > 0
            mask_overlay[mask_bool] = (
                mask_overlay[mask_bool] * 0.5 + np.array([0, 200, 0]) * 0.5
            ).astype(np.uint8)
            vis = mask_overlay

    return vis


def main():
    parser = argparse.ArgumentParser(description="Test perception pipeline on OAK-D")
    parser.add_argument("--query", type=str, default="objects on the table", help="What to detect")
    parser.add_argument("--backend", type=str, default="gemini", choices=["local", "cloud", "gemini", "claude"])
    parser.add_argument("--model-id", type=str, default="microsoft/Florence-2-base")
    parser.add_argument("--device-id", type=str, default="", help="OAK-D MX ID (empty=auto)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    from lerobot.cameras.oakd.camera_oakd import OAKDCamera
    from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig

    logger.info("Connecting OAK-D camera...")
    cam_cfg = OAKDCameraConfig(
        device_id=args.device_id,
        fps=args.fps,
        width=args.width,
        height=args.height,
        use_depth=True,
    )
    camera = OAKDCamera(cam_cfg)
    camera.connect()

    intrinsics = {}
    try:
        intrinsics = camera.get_depth_intrinsics()
        logger.info(f"Depth intrinsics: {intrinsics}")
    except Exception as e:
        logger.warning(f"Could not read intrinsics: {e}")

    logger.info(f"Loading VLM detector (backend={args.backend})...")
    from lerobot.perception.vlm_detector import VLMDetector

    detector = VLMDetector(backend=args.backend, model_id=args.model_id)

    from lerobot.processor.depth_perception_processor import compute_object_state

    logger.info("Ready. Press SPACE to detect, Q to quit.\n")

    detections = []
    obj_states = []
    annotated = None

    try:
        while True:
            rgb = camera.read_latest()
            depth = camera.read_depth()

            display = rgb.copy()
            if annotated is not None:
                display = annotated

            depth_vis = colorize_depth(depth)

            h, w = display.shape[:2]
            depth_resized = cv2.resize(depth_vis, (w, h))
            combined = np.hstack([
                cv2.cvtColor(display, cv2.COLOR_RGB2BGR),
                depth_resized,
            ])

            status = f"Query: '{args.query}' | Detections: {len(detections)} | SPACE=detect Q=quit S=save"
            cv2.putText(combined, status, (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Perception Pipeline Test", combined)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord(" "):
                logger.info(f"Running detection: '{args.query}'...")
                t0 = time.time()
                detections = detector.detect(rgb, args.query)
                dt_detect = time.time() - t0
                logger.info(f"VLM detection took {dt_detect:.2f}s, found {len(detections)} object(s)")

                obj_states = []
                for det in detections:
                    logger.info(f"  - {det.label} bbox={det.bbox_xyxy}")
                    mask = det.mask
                    if mask is not None:
                        state = compute_object_state(depth, mask, intrinsics)
                        obj_states.append(state)
                        c = state["obj_center_xyz"]
                        s = state["obj_size_xyz"]
                        d = state["obj_distance"][0]
                        logger.info(f"    3D: center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
                                    f"size=({s[0]*100:.1f},{s[1]*100:.1f},{s[2]*100:.1f})cm "
                                    f"dist={d:.3f}m")
                    else:
                        obj_states.append({
                            "obj_center_xyz": np.zeros(3, dtype=np.float32),
                            "obj_size_xyz": np.zeros(3, dtype=np.float32),
                            "obj_distance": np.zeros(1, dtype=np.float32),
                        })

                annotated = draw_detections(rgb, detections, obj_states)

            elif key == ord("s") and annotated is not None:
                fname = f"perception_test_{int(time.time())}.png"
                cv2.imwrite(fname, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved annotated frame to {fname}")

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()
        logger.info("Camera disconnected.")


if __name__ == "__main__":
    main()
