"""
Examples of usage:

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py record_dataset \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay_episode \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

- Record a full dataset in order to train a policy:
```bash
python lerobot/scripts/control_robot.py record_dataset \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --run-compute-stats 1
```

- Train on this dataset with the ACT policy:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py run_policy \
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import concurrent.futures
import logging
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import DictConfig
from PIL import Image

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import save_meta_data

########################################################################################
# Utilities
########################################################################################


def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_tensor.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)

def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

def log_control_info(robot, dt_s, episode_index=None, frame_index=None):
    log_items = []
    if episode_index is not None:
        log_items += [f"ep:{episode_index}"]
    if frame_index is not None:
        log_items += [f"frame:{frame_index}"]
        
    def log_dt(shortname, dt_val_s):
        nonlocal log_items
        log_items += [f"{shortname}:{dt_val_s * 1000:5.2f}={1/ dt_val_s:3.1f}hz"]

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    for name in robot.leader_arms:
        key = f'read_leader_{name}_pos_dt_s'
        if key in robot.logs:
            log_dt("dtRlead", robot.logs[key])

    for name in robot.follower_arms:
        key = f'write_follower_{name}_goal_pos_dt_s'
        if key in robot.logs:
            log_dt("dtRfoll", robot.logs[key])

        key = f'read_follower_{name}_pos_dt_s'
        if key in robot.logs:
            log_dt("dtWfoll", robot.logs[key])

    for name in robot.cameras:
        key = f"read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt("dtRcam", robot.logs[key])

        key = f"async_read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt("dtARcam", robot.logs[key])

    logging.info(" ".join(log_items))

########################################################################################
# Control modes
########################################################################################


def teleoperate(robot: Robot, fps: int | None = None, teleop_time_s: float | None = None):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    start_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - now
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s)

        if teleop_time_s is not None and time.perf_counter() - start_time > teleop_time_s:
            break


def record_dataset(
    robot: Robot,
    fps: int | None = None,
    root="data",
    repo_id="lerobot/debug",
    warmup_time_s=2,
    episode_time_s=10,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
):
    # TODO(rcadene): Add option to record logs

    if not video:
        raise NotImplementedError()

    if not robot.is_connected:
        robot.connect()

    local_dir = Path(root) / repo_id
    if local_dir.exists():
        shutil.rmtree(local_dir)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    # Using only 4 worker threads to avoid blocking the main thread.

    futures = []

    # Execute a few seconds without recording data, to give times
    # to the robot devices to connect and start synchronizing.
    timestamp = 0
    start_time = time.perf_counter()
    is_warmup_print = False
    while timestamp < warmup_time_s:
        if not is_warmup_print:
            logging.info("Warming up (no data recording)")
            os.system('say "Warmup" &')
            is_warmup_print = True

        now = time.perf_counter()
        observation, action = robot.teleop_step(record_data=True)

        dt_s = time.perf_counter() - now
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s)

        timestamp = time.perf_counter() - start_time

    # Start recording all episodes
    ep_dicts = []
    for episode_index in range(num_episodes):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            ep_dict = {}
            frame_index = 0
            timestamp = 0
            start_time = time.perf_counter()
            is_record_print = False
            while timestamp < episode_time_s:
                if not is_record_print:
                    logging.info(f"Recording episode {episode_index}")
                    os.system(f'say "Recording episode {episode_index}" &')
                    is_record_print = True

                now = time.perf_counter()
                observation, action = robot.teleop_step(record_data=True)

                image_keys = [key for key in observation if "image" in key]
                not_image_keys = [key for key in observation if "image" not in key]

                for key in image_keys:
                    future = executor.submit(save_image, observation[key], key, frame_index, episode_index, videos_dir)
                    futures.append(future)

                for key in not_image_keys:
                    if key not in ep_dict:
                        ep_dict[key] = []
                    ep_dict[key].append(observation[key])

                for key in action:
                    if key not in ep_dict:
                        ep_dict[key] = []
                    ep_dict[key].append(action[key])

                frame_index += 1

                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

                dt_s = time.perf_counter() - now
                log_control_info(robot, dt_s)

                timestamp = time.perf_counter() - start_time

        logging.info("Encoding images to videos")

        num_frames = frame_index

        for key in image_keys:
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps)

            # TODO(rcadene): uncomment?
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

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }

    meta_data_dir = local_dir / "meta_data"

    for key in image_keys:
        time.sleep(10)
        tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
        shutil.rmtree(tmp_imgs_dir)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset) if run_compute_stats else {}
    lerobot_dataset.stats = stats

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    # TODO(rcadene): push to hub

    return lerobot_dataset


def replay_episode(robot: Robot, episode: int, fps: int | None = None, root="data", repo_id="lerobot/debug"):
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()

    if not robot.is_connected:
        robot.connect()

    logging.info("Replaying episode")
    os.system('say "Replaying episode"')

    for idx in range(from_idx, to_idx):
        now = time.perf_counter()

        action = items[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - now
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s)


def run_policy(robot: Robot, policy: torch.nn.Module, hydra_cfg: DictConfig, run_time_s: float | None = None):
    # TODO(rcadene): Add option to record eval dataset and logs
    policy.eval()

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    fps = hydra_cfg.env.fps

    if not robot.is_connected:
        robot.connect()

    start_time = time.perf_counter()
    while True:
        now = time.perf_counter()

        observation = robot.capture_observation()

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext(),
        ):
            action = policy.select_action(observation)

        robot.send_action(action)

        dt_s = time.perf_counter() - now
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s)

        if run_time_s is not None and time.perf_counter() - start_time > run_time_s:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot",
        type=str,
        default="koch",
        help="Name of the robot provided to the `make_robot(name)` factory function.",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )

    parser_record = subparsers.add_parser("record_dataset", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument("--root", type=Path, default="data", help="")
    parser_record.add_argument("--repo-id", type=str, default="lerobot/test", help="")
    parser_record.add_argument("--warmup-time-s", type=int, default=2, help="")
    parser_record.add_argument("--episode-time-s", type=int, default=10, help="")
    parser_record.add_argument("--num-episodes", type=int, default=50, help="")
    parser_record.add_argument("--run-compute-stats", type=int, default=1, help="")

    parser_replay = subparsers.add_parser("replay_episode", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument("--root", type=Path, default="data", help="")
    parser_replay.add_argument("--repo-id", type=str, default="lerobot/test", help="")
    parser_replay.add_argument("--episode", type=int, default=0, help="")

    parser_policy = subparsers.add_parser("run_policy", parents=[base_parser])
    parser_policy.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_policy.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_name = args.robot
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot"]

    robot = make_robot(robot_name)
    if control_mode == "teleoperate":
        teleoperate(robot, **kwargs)
    elif control_mode == "record_dataset":
        record_dataset(robot, **kwargs)
    elif control_mode == "replay_episode":
        replay_episode(robot, **kwargs)

    elif control_mode == "run_policy":
        pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)
        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", args.overrides)
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
        run_policy(robot, policy, hydra_cfg)
