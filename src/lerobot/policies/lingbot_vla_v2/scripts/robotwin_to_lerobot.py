# RoboTwin raw HDF5 episodes -> lerobot v3 dataset converter
#
# This is step 1 of turning raw data collected by RoboTwin into a format trainable by
# lingbot-vla-v2 (the lerobot framework).
# The output schema matches the local data/lerobot/rotate_qrcode_joint_v30 exactly:
#   observation.state  (14,) float32   [left_arm6, left_ee, right_arm6, right_ee]
#   action             (14,) float32   state[t+1] (teacher-shift, aligned with the pkl2hdf5 split)
#   observation.images.cam_high        video (3,H,W)
#   observation.images.cam_left_wrist  video (3,H,W)
#   observation.images.cam_right_wrist video (3,H,W)
#   task               str
#
# The source HDF5 is the output of pkl2hdf5.create_xpolicylab_hdf5, structured as:
#   state/{left,right}_{arm,ee}_joint_states   (T-1, d)   per-frame state
#   action/{left,right}_{arm,ee}_joint_states  (T-1, d)   per-frame action
#   vision/cam_head|cam_left_wrist|cam_right_wrist/colors  (T-1,) JPEG-encoded bytes
#   instructions                                 JSON-encoded list of strings
#   additional_info/frequency                    collection frequency
#
# The camera mapping matches pkl2hdf5 (cam_head -> cam_high), and the target keys align with
# the origin_keys of lingbot robotwin.yaml, so the converted output can be fed directly to
# lerobot-train / the converter --profile robotwin.
#
# Usage:
#   python -m lerobot.policies.lingbot_vla_v2.scripts.robotwin_to_lerobot \
#       --input-dir /path/to/robwin_task_episodes \
#       --repo-id my_robotwin_task \
#       --fps 15 \
#       [--robot-type aloha_agilex] [--mode video|image] [--out-dir ...] [--max-episodes N]
#
# Requires lerobot (this repo's lingbot_vla_v2 environment) + h5py + numpy + opencv (JPEG decoding).

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# HDF5 source camera keys -> lerobot target camera keys (aligned with pkl2hdf5.CAMERA_MAP + robotwin.yaml)
VISION_TO_LEROBOT_CAM = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

# Joint fields (left arm + left gripper first, then right arm + right gripper, matching the dual-arm definition in _robot_info.json)
JOINT_FIELDS = [
    ("left_arm_joint_states", "arm"),
    ("left_ee_joint_states", "ee"),
    ("right_arm_joint_states", "arm"),
    ("right_ee_joint_states", "ee"),
]


def _decode_jpeg(buf) -> np.ndarray:
    """Decode the JPEG bytes written by pkl2hdf5 into HWC uint8 RGB."""
    import cv2

    if isinstance(buf, np.ndarray) and buf.ndim == 3:
        return buf  # already a decoded image
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed: the byte stream is not a valid JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_episode(ep_path: Path) -> dict:
    """Load one RoboTwin HDF5 episode and return state/action/images/instructions."""
    with h5py.File(ep_path, "r") as f:
        state_parts, action_parts = [], []
        for field, _kind in JOINT_FIELDS:
            if f"state/{field}" in f:
                state_parts.append(np.asarray(f[f"state/{field}"], dtype=np.float32))
                action_parts.append(np.asarray(f[f"action/{field}"], dtype=np.float32))
        if not state_parts:
            raise KeyError(f"{ep_path} is missing state/*_joint_states")

        state = np.concatenate(state_parts, axis=1)  # (T-1, 14)
        action = np.concatenate(action_parts, axis=1)  # (T-1, 14)

        images = {}
        for src_cam, dst_cam in VISION_TO_LEROBOT_CAM.items():
            key = f"vision/{src_cam}/colors"
            if key not in f:
                continue
            images[dst_cam] = np.stack(
                [_decode_jpeg(b) for b in f[key][()]], axis=0
            )  # (T-1, H, W, 3) uint8

        instructions = None
        if "instructions" in f:
            raw = f["instructions"][()]
            raw = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
            try:
                instructions = json.loads(raw)
            except json.JSONDecodeError:
                instructions = [raw]

        fps = None
        if "additional_info/frequency" in f:
            fps = int(np.asarray(f["additional_info/frequency"]).item())

    return {
        "state": state,
        "action": action,
        "images": images,
        "instructions": instructions,
        "fps": fps,
    }


def _create_dataset(
    repo_id: str,
    root: Path,
    robot_type: str,
    fps: int,
    action_dim: int,
    img_shape: tuple,
    mode: str,
) -> LeRobotDataset:
    """Create an empty lerobot v3 dataset with a schema aligned to the local rotate_qrcode_joint_v30."""
    h, w, _c = img_shape
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
    }
    for cam in VISION_TO_LEROBOT_CAM.values():
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }

    # The pyav backend defaults to vcodec=libsvtav1, which is unavailable on this machine
    # -> explicitly use h264 (the most widely supported option).
    # Note: DatasetWriter constructs a default RGBEncoderConfig even in image mode and crashes
    # on av<15, so we always pass h264; actual encoding only happens in video mode.
    from lerobot.configs.video import RGBEncoderConfig

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=(mode == "video"),
        image_writer_processes=0,
        image_writer_threads=4,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboTwin HDF5 -> lerobot v3 dataset converter")
    ap.add_argument("--input-dir", required=True, help="Directory containing episode_*.hdf5 files (searched recursively)")
    ap.add_argument("--repo-id", required=True, help="lerobot repo_id (e.g. my_robotwin_task)")
    ap.add_argument("--out-dir", default=None, help="Output root directory (defaults to HF_LEROBOT_HOME/repo_id)")
    ap.add_argument("--fps", type=int, default=15, help="Collection frequency (the HDF5 frequency takes precedence when present)")
    ap.add_argument("--robot-type", default="aloha_agilex", help="robot_type tag")
    ap.add_argument("--mode", choices=["video", "image"], default="video", help="Store frames as video or as individual images")
    ap.add_argument("--max-episodes", type=int, default=None, help="Only convert the first N episodes (for debugging)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    ep_files = sorted(input_dir.rglob("*.hdf5"))
    if not ep_files:
        raise FileNotFoundError(f"No *.hdf5 episode files found under {input_dir}")
    if args.max_episodes is not None:
        ep_files = ep_files[: args.max_episodes]
    print(f"Found {len(ep_files)} episodes, starting conversion -> repo_id={args.repo_id}")

    # Probe dimensions/image shape/fps from the first episode, then create the dataset
    first = _load_episode(ep_files[0])
    action_dim = int(first["state"].shape[1])
    if not first["images"]:
        raise ValueError(f"{ep_files[0]} has no camera images; cannot determine the image shape")
    any_cam = sorted(first["images"])[0]
    img_shape = first["images"][any_cam].shape[1:]  # (H, W, 3)
    fps = first["fps"] or args.fps
    print(f"Probe: action_dim={action_dim}, img={img_shape}, fps={fps}, cams={list(first['images'])}")

    root = Path(args.out_dir) / args.repo_id if args.out_dir else None
    dataset = _create_dataset(
        repo_id=args.repo_id,
        root=root,
        robot_type=args.robot_type,
        fps=fps,
        action_dim=action_dim,
        img_shape=img_shape,
        mode=args.mode,
    )

    n_frames = 0
    for ep_path in ep_files:
        try:
            data = _load_episode(ep_path)
            state, action, images = data["state"], data["action"], data["images"]
            instrs = data["instructions"] or [""]
            task = instrs[0]
            T = state.shape[0]

            for i in range(T):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                    "task": task,
                }
                for cam, imgs in images.items():
                    if i < imgs.shape[0]:
                        # add_frame expects HWC uint8 (lerobot internally converts to CHW for
                        # encoding); do not permute beforehand.
                        frame[f"observation.images.{cam}"] = torch.from_numpy(imgs[i])
                dataset.add_frame(frame)

            dataset.save_episode()
            n_frames += T
            print(f"  {ep_path.name}: {T} frames")
        except Exception as e:  # noqa: BLE001 - a single episode failure must not abort the whole batch
            print(f"  {ep_path.name}: failed ({e}), skipping", file=sys.stderr)

    print(f"Done: {len(ep_files)} episodes / {n_frames} frames -> {dataset.root}")
    print("Next steps: generate norm_stats (--quantiles) + converter --profile robotwin, then it is ready for lerobot-train")


if __name__ == "__main__":
    main()
