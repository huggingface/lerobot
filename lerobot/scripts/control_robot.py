

import argparse
import os
from pathlib import Path
import shutil
import time

from PIL import Image
import cv2
import torch
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index, load_hf_dataset
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
from lerobot.common.robot_devices.robots.aloha import AlohaRobot, AlohaRobotConfig
from lerobot.scripts.push_dataset_to_hub import save_meta_data
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
    #activated_leaders=["left"],
    activated_leaders=["right"],
    #activated_followers=["left"],
    activated_followers=["right"],
    #activated_cameras=["cam_high", "cam_low", "cam_left_wrist"],
    #activated_cameras=["cam_high", "cam_left_wrist"],
    #activated_cameras=["cam_left_wrist"],
    activated_cameras=[],
    camera_devices={
        # "cam_high": IntelRealSenseCamera(128422271609, width=640, height=480, color="rgb", fps=30),
        # "cam_low": IntelRealSenseCamera(128422271393, width=640, height=480, color="rgb", fps=30),
        "cam_left_wrist": IntelRealSenseCamera(128422271614, width=640, height=480, color="rgb", fps=30),
    }
)

def teleoperate():
    robot = AlohaRobot(CONFIG, activated_cameras=None)
    robot.init_teleop()

    while True:
        now = time.perf_counter()
        robot.teleop_step()

        dt_s = (time.perf_counter() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_tensor.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def record_dataset(root="tmp/data", repo_id="lerobot/debug", fps=30, video=True, warmup_time_s=2, record_time_s=60):
    if not video:
        raise NotImplementedError()

    robot = AlohaRobot(CONFIG)
    robot.init_teleop()

    local_dir = Path(root) / repo_id
    if local_dir.exists():
        shutil.rmtree(local_dir)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    episode_index = 0
    ep_dict = {}

    start_time = time.perf_counter()

    is_warmup_print = False
    is_record_print = False

    # Save images using threads to reach high fps (30 and more)
    # Using `with` ensures the program exists smoothly if an execption is raised.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            if not is_warmup_print:
                print("Warming up by skipping frames")
                os.system('spd-say "Warmup"')
                is_warmup_print = True
            now = time.perf_counter()

            observation, action = robot.teleop_step(record_data=True)
            timestamp = time.perf_counter() - start_time

            if timestamp < warmup_time_s:
                dt_s = (time.perf_counter() - now)
                print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f} (Warmup)")
                continue

            if not is_record_print:
                print("Recording")
                os.system('spd-say "Recording"')
                is_record_print = True

            image_keys = [key for key in observation if "image" in key]
            not_image_keys = [key for key in observation if "image" not in key]

            for key in image_keys:
                executor.submit(save_image, observation[key], key, frame_index, episode_index, videos_dir)

            for key in not_image_keys:
                if key not in ep_dict:
                    ep_dict[key] = []
                ep_dict[key].append(observation[key])

            for key in action:
                if key not in ep_dict:
                    ep_dict[key] = []
                ep_dict[key].append(action[key])

            frame_index += 1

            dt_s = (time.perf_counter() - now)
            print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")

            if timestamp > record_time_s - warmup_time_s:
                break

            # try:
            #     if keyboard.is_pressed('q'):
            #         break
            # except ImportError:
            #     break

    print("Encoding to `LeRobotDataset` format")
    os.system('spd-say "Encoding"')

    num_frames = frame_index

    for key in image_keys:
        tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
        fname = f"{key}_episode_{episode_index:06d}.mp4"
        video_path = local_dir / "videos" / fname
        encode_video_frames(tmp_imgs_dir, video_path, fps)

        # clean temporary images directory
        # shutil.rmtree(tmp_imgs_dir)

        # store the reference to the video frame
        ep_dict[key] = []
        for i in range(num_frames):
            ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / fps})

    for key in not_image_keys:
        ep_dict[key] = torch.stack(ep_dict[key])
    
    for key in action:
        ep_dict[key] = torch.stack(ep_dict[key])

    ep_dict["episode_index"] = torch.tensor([episode_index] * num_frames)
    ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
    ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

    done = torch.zeros(num_frames, dtype=torch.bool)
    done[-1] = True
    ep_dict["next.done"] = done

    ep_dicts = [ep_dict]

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"

    stats = {}
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    # for key in image_keys:
    #     time.sleep(10)
    #     tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
    #     shutil.rmtree(tmp_imgs_dir)

    # lerobot_dataset = LeRobotDataset.from_preloaded(
    #     repo_id=repo_id,
    #     hf_dataset=hf_dataset,
    #     episode_data_index=episode_data_index,
    #     info=info,
    #     videos_dir=videos_dir,
    # )
    # stats = compute_stats(lerobot_dataset, batch_size, num_workers)


def replay_episode(root="tmp/data", repo_id="lerobot/debug", fps=30):
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    hf_dataset = load_hf_dataset(repo_id, CODEBASE_VERSION, root, "train")

    robot = AlohaRobot(CONFIG)
    robot.init_teleop()
    
    print("Replaying episode")
    os.system('spd-say "Replaying episode"')

    items = hf_dataset.select_columns("action")
    for item in items:
        now = time.perf_counter()

        action = item["action"]
        robot.send_action(action)

        dt_s = (time.perf_counter() - now)
        time.sleep(1 / fps - dt_s)

        dt_s = (time.perf_counter() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


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
