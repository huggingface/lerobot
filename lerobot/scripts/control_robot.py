"""
Example of usage:

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

- Train on this dataset (TODO(rcadene)):
```bash
python lerobot/scripts/train.py
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py run_policy \
    -p TODO(rcadene)
```
"""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import time

from PIL import Image
from omegaconf import DictConfig
import torch
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index, load_hf_dataset
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.scripts.robot_controls.record_dataset import record_dataset
import concurrent.futures


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
    # TODO(rcadene): find an alternative
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

########################################################################################
# Control modes
########################################################################################

def teleoperate(robot: Robot, fps: int | None = None):
    robot.init_teleop()

    while True:
        now = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = (time.perf_counter() - now)
            busy_wait(1 / fps - dt_s)

        dt_s = (time.perf_counter() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def record_dataset(robot: Robot, fps: int | None = None, root="data", repo_id="lerobot/debug", warmup_time_s=2, episode_time_s=10, num_episodes=50, video=True, run_compute_stats=True):
    if not video:
        raise NotImplementedError()

    robot.init_teleop()

    local_dir = Path(root) / repo_id
    if local_dir.exists():
        shutil.rmtree(local_dir)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)


    start_time = time.perf_counter()

    is_warmup_print = False
    is_record_print = False
    ep_dicts = []

    # Save images using threads to reach high fps (30 and more)
    # Using `with` ensures the program exists smoothly if an execption is raised.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for episode_index in range(num_episodes):

            ep_dict = {}
            frame_index = 0

            while True:
                if not is_warmup_print:
                    print("Warming up by skipping frames")
                    os.system('say "Warmup"')
                    is_warmup_print = True
                now = time.perf_counter()

                observation, action = robot.teleop_step(record_data=True)
                timestamp = time.perf_counter() - start_time

                if timestamp < warmup_time_s:
                    dt_s = (time.perf_counter() - now)
                    busy_wait(1 / fps - dt_s)

                    dt_s = (time.perf_counter() - now)
                    print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f} (Warmup)")
                    continue

                if not is_record_print:
                    print("Recording")
                    os.system(f'say "Recording episode {episode_index}"')
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
                busy_wait(1 / fps - dt_s)

                dt_s = (time.perf_counter() - now)
                print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")

                if timestamp > episode_time_s - warmup_time_s:
                    break

            print("Encoding to `LeRobotDataset` format")
            os.system('say "Encoding"')

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
        # shutil.rmtree(tmp_imgs_dir)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    if run_compute_stats:
        stats = compute_stats(lerobot_dataset)
    else:
        stats = {}

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    # TODO(rcadene): push to hub


def replay_episode(robot: Robot, episode: int, fps: int | None = None, root="data", repo_id="lerobot/debug"):
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()

    robot.init_teleop()
    
    print("Replaying episode")
    os.system('say "Replaying episode"')

    for idx in range(from_idx, to_idx):
        now = time.perf_counter()

        action = items[idx]["action"]
        robot.send_action(action)

        dt_s = (time.perf_counter() - now)
        busy_wait(1 / fps - dt_s)

        dt_s = (time.perf_counter() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def run_policy(robot: Robot, policy: torch.nn.Module, hydra_cfg: DictConfig):
    policy.eval()

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    fps = hydra_cfg.env.fps

    while True:
        now = time.perf_counter()

        observation = robot.capture_observation()

        with torch.inference_mode(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
            action = policy.select_action(observation)

        robot.send_action(action)

        dt_s = (time.perf_counter() - now)
        busy_wait(1 / fps - dt_s)

        dt_s = (time.perf_counter() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--robot", type=str, default="koch", help="Name of the robot provided to the `make_robot(name)` factory function.")

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument('--fps', type=none_or_int, default=None, help='Frames per second (set to None to disable)')

    parser_record = subparsers.add_parser("record_dataset", parents=[base_parser])
    parser_record.add_argument('--fps', type=none_or_int, default=None, help='Frames per second (set to None to disable)')
    parser_record.add_argument('--root', type=Path, default="data", help='')
    parser_record.add_argument('--repo-id', type=str, default="lerobot/test", help='')
    parser_record.add_argument('--warmup-time-s', type=int, default=2, help='')
    parser_record.add_argument('--episode-time-s', type=int, default=10, help='')
    parser_record.add_argument('--num-episodes', type=int, default=50, help='')
    parser_record.add_argument('--run-compute-stats', type=int, default=1, help='')

    parser_replay = subparsers.add_parser("replay_episode", parents=[base_parser])
    parser_replay.add_argument('--fps', type=none_or_int, default=None, help='Frames per second (set to None to disable)')
    parser_replay.add_argument('--root', type=Path, default="data", help='')
    parser_replay.add_argument('--repo-id', type=str, default="lerobot/test", help='')
    parser_replay.add_argument('--episode', type=int, default=0, help='')

    parser_policy = subparsers.add_parser("run_policy", parents=[base_parser])
    parser_policy.add_argument('-p', '--pretrained-policy-name-or-path', type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        )
    )
    parser_policy.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

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
