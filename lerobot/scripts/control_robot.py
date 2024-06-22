

import argparse
from pathlib import Path
import shutil
import time

from PIL import Image
import cv2
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
from lerobot.common.robot_devices.robots.aloha import AlohaRobot, AlohaRobotConfig
from lerobot.scripts.robot_controls.record_dataset import record_dataset
import concurrent.futures

CONTROL_MODES = [
    "teleoperate",
    "record_dataset",
    "replay_episode",
    "run_policy",
    "disable_torque",
]



CONFIG = AlohaRobotConfig(
    activated_leaders=["left"],
    activated_followers=["left"],
    activated_cameras=["cam_high", "cam_low", "cam_left_wrist"],
    camera_devices={
        "cam_high": IntelRealSenseCamera(128422271609, width=640, height=480, color="rgb", fps=30),
        "cam_low": IntelRealSenseCamera(128422271393, width=640, height=480, color="rgb", fps=30),
        "cam_left_wrist": IntelRealSenseCamera(128422271614, width=640, height=480, color="rgb", fps=30),
    }
)

def teleoperate():
    robot = AlohaRobot(CONFIG, activated_cameras=None)
    robot.init_teleop()

    while True:
        now = time.time()
        robot.teleop_step()

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def save_image(observation, key, frame_index, episode_index, videos_dir):
    img = observation[key]
    img = Image.fromarray(img.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def record_dataset(out_dir="tmp/data/lerobot/debug"):
    robot = AlohaRobot(CONFIG)
    robot.init_teleop()

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    episode_index = 0
    timestamps = []
    observations = []
    actions = []

    start_time = time.time()

    # We need to save images with threads to reach high fps (30 and more)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            # START
            now = time.time()

            observation, action = robot.teleop_step(record_data=True)
            timestamp = time.time() - start_time

            def save_image_threaded_(key):
                save_image(observation, key, frame_index, episode_index, videos_dir)

            image_keys = [key for key in observation if "image" in key]
            not_image_keys = [key for key in observation if "image" not in key]

            executor.map(save_image_threaded_, image_keys)
            observations.append({key: observation[key] for key in not_image_keys})
            actions.append(action)
            timestamps.append(timestamp)

            frame_index += 1

            # END
            dt_s = (time.time() - now)
            print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")

            if timestamp > 1:
                print("END")
                break
    
    ep_dict = {}
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=CONTROL_MODES, default="teleoperate")
    args = parser.parse_args()

    if args.mode == "teleoperate":
        teleoperate()
    elif args.mode == "record_dataset":
        record_dataset()
    elif args.mode == "replay_episode":
        replay_episode()
    # elif args.mode == "find_camera_ids":
    #     find_camera_ids()
    elif args.mode == "disable_torque":
        disable_torque()
