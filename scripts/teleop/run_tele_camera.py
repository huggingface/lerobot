import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# --- 1. Cấu hình ---
RECORD_DIR = Path("teleop_recordings")
if RECORD_DIR.exists():
    shutil.rmtree(RECORD_DIR)  # Xóa cũ để tránh lẫn lộn
os.makedirs(RECORD_DIR / "wrist", exist_ok=True)
os.makedirs(RECORD_DIR / "portal", exist_ok=True)

camera_config = {
    "wrist": OpenCVCameraConfig(index_or_path=0, width=800, height=600, fps=30),
    "portal": OpenCVCameraConfig(index_or_path=4, width=1280, height=720, fps=30),
}

robot_config = SO101FollowerConfig(port="COM6", id="DI_VLA_FOLLOWER", cameras=camera_config, use_degrees=True)
teleop_config = SO101LeaderConfig(port="COM5", id="DI_VLA_LEADER", use_degrees=True)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)


def process_for_save(img_data):
    """Chuyển đổi sang BGR chuẩn để lưu ảnh"""
    if torch.is_tensor(img_data):
        img_np = img_data.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img_data.transpose(1, 2, 0) if img_data.shape[0] == 3 else img_data

    if img_np.max() <= 1.1:
        img_np = (img_np * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


# --- 2. Execution ---
try:
    robot.connect()
    teleop_device.connect()
    print("🚀 BẮT ĐẦU RECORD (Lưu ảnh thô)... Nhấn Ctrl+C để dừng.")

    frame_count = 0
    start_time = time.time()

    while True:
        loop_start = time.time()
        observation = robot.get_observation()

        # Lưu ảnh Wrist
        if "wrist" in observation:
            frame_w = process_for_save(observation["wrist"])
            cv2.imwrite(
                str(RECORD_DIR / "wrist" / f"frame_{frame_count:06d}.jpg"),
                frame_w,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

        # Lưu ảnh Portal (Phone)
        if "portal" in observation:
            frame_p = process_for_save(observation["portal"])
            cv2.imwrite(
                str(RECORD_DIR / "portal" / f"frame_{frame_count:06d}.jpg"),
                frame_p,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

        # Điều khiển
        action = teleop_device.get_action()
        robot.send_action(action)

        frame_count += 1

        # In FPS thực tế mỗi 30 frames
        if frame_count % 30 == 0:
            current_fps = frame_count / (time.time() - start_time)
            print(f"📊 Recording... Frame: {frame_count} | Real FPS: {current_fps:.1f}")

except KeyboardInterrupt:
    print("\n🛑 Đã dừng.")
finally:
    robot.disconnect()
    teleop_device.disconnect()
