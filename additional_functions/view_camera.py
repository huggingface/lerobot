"""
Bigger display version of camera viewer.
Use --scale to control how big the camera windows are.
Example:
    python unitree_lerobot/eval_robot/view_camera.py --scale 1.5
"""

import time
import cv2
import argparse
import numpy as np
from unitree_lerobot.eval_robot.make_robot import setup_image_client

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by a scale factor for larger display."""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def view_camera_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.5, help="Resize factor for display windows")
    args = parser.parse_args()

    # Create minimal config (no robot control needed)
    cfg = argparse.Namespace(
        sim=False,      # Real robot (not simulation)
        arm="G1_29",    # Doesn't matter for camera-only
        ee="dex3",      # Doesn't matter for camera-only
        motion=False
    )

    scale_factor = args.scale

    logger_mp.info("Setting up image client...")
    logger_mp.info("Make sure image_server.py is running on the robot!")
    logger_mp.info("Default robot IP: 192.168.123.164:5555")

    image_info = setup_image_client(cfg)
    tv_img_array = image_info["tv_img_array"]
    wrist_img_array = image_info["wrist_img_array"]
    tv_img_shape = image_info["tv_img_shape"]
    wrist_img_shape = image_info["wrist_img_shape"]
    is_binocular = image_info["is_binocular"]
    has_wrist_cam = image_info["has_wrist_cam"]

    logger_mp.info(f"Camera config:")
    logger_mp.info(f"  - Head camera: {tv_img_shape}, binocular: {is_binocular}")
    logger_mp.info(f"  - Wrist camera: {wrist_img_shape if has_wrist_cam else 'Not available'}")
    logger_mp.info("")
    logger_mp.info("Waiting for first frame...")

    time.sleep(2)

    logger_mp.info("=" * 60)
    logger_mp.info("CAMERA VIEWER ACTIVE")
    logger_mp.info("=" * 60)
    logger_mp.info(f"Scale factor: {scale_factor}")
    logger_mp.info("Controls:")
    logger_mp.info("  - Press 'q' to quit")
    logger_mp.info("  - Press 's' to save current frame as image")
    logger_mp.info("=" * 60)

    frame_count = 0

    try:
        while True:
            head_image = tv_img_array.copy()

            # HEAD CAMERA
            if head_image is not None and head_image.size > 0:
                if np.any(head_image):
                    if is_binocular:
                        h, w = head_image.shape[:2]
                        left_head = head_image[:, :w // 2]
                        right_head = head_image[:, w // 2:]

                        left_display = resize_for_display(left_head, scale_factor)
                        right_display = resize_for_display(right_head, scale_factor)

                        cv2.imshow('Left Head Camera', left_display)
                        cv2.imshow('Right Head Camera', right_display)
                    else:
                        display_img = resize_for_display(head_image, scale_factor)
                        cv2.imshow('Head Camera', display_img)
                else:
                    logger_mp.warning("Received empty frame from head camera")

            # WRIST CAMERA
            if has_wrist_cam and wrist_img_array is not None:
                wrist_image = wrist_img_array.copy()
                if wrist_image is not None and wrist_image.size > 0 and np.any(wrist_image):
                    h, w = wrist_image.shape[:2]
                    left_wrist = wrist_image[:, :w // 2]
                    right_wrist = wrist_image[:, w // 2:]

                    left_wrist_display = resize_for_display(left_wrist, scale_factor)
                    right_wrist_display = resize_for_display(right_wrist, scale_factor)

                    cv2.imshow('Left Wrist Camera', left_wrist_display)
                    cv2.imshow('Right Wrist Camera', right_wrist_display)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger_mp.info("Quit command received")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'head_camera_{timestamp}.jpg', head_image)
                logger_mp.info(f"Saved: head_camera_{timestamp}.jpg")
                if has_wrist_cam and wrist_img_array is not None:
                    wrist_image = wrist_img_array.copy()
                    if wrist_image is not None and wrist_image.size > 0:
                        cv2.imwrite(f'wrist_camera_{timestamp}.jpg', wrist_image)
                        logger_mp.info(f"Saved: wrist_camera_{timestamp}.jpg")

            frame_count += 1
            if frame_count % 30 == 0:
                logger_mp.info(f"Frames received: {frame_count}")

            time.sleep(0.033)

    except KeyboardInterrupt:
        logger_mp.info("\nInterrupted by user (Ctrl+C)")

    finally:
        logger_mp.info("Closing camera viewer...")
        cv2.destroyAllWindows()
        #from lerobot.robots.unitree_g1.eval_robot.utils.utils import cleanup_resources
        #cleanup_resources(image_info)
        logger_mp.info("Done!")


if __name__ == "__main__":
    view_camera_main()
