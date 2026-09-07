# RoboTwin 原始 HDF5 回合 -> lerobot v3 数据集 转换器
#
# 这是把 RoboTwin 采集的原始数据转成 lingbot-vla-v2 (lerobot 框架) 可训练格式的第 1 步。
# 产物 schema 与本机 data/lerobot/rotate_qrcode_joint_v30 完全一致:
#   observation.state  (14,) float32   [left_arm6, left_ee, right_arm6, right_ee]
#   action             (14,) float32   state[t+1](teacher-shift,与 pkl2hdf5 的 split 对齐)
#   observation.images.cam_high        video (3,H,W)
#   observation.images.cam_left_wrist  video (3,H,W)
#   observation.images.cam_right_wrist video (3,H,W)
#   task               str
#
# 源 HDF5 是 pkl2hdf5.create_xpolicylab_hdf5 的产出,结构:
#   state/{left,right}_{arm,ee}_joint_states   (T-1, d)   每帧状态
#   action/{left,right}_{arm,ee}_joint_states  (T-1, d)   每帧动作
#   vision/cam_head|cam_left_wrist|cam_right_wrist/colors  (T-1,) JPEG 编码字节
#   instructions                                 JSON 字符串列表
#   additional_info/frequency                    采集频率
#
# 相机映射与 pkl2hdf5 一致(cam_head -> cam_high),目标键与 lingbot robotwin.yaml 的
# origin_keys 对齐,转完即可直接喂给 lerobot-train / 转换器 --profile robotwin。
#
# 用法:
#   python scripts/robotwin_to_lerobot.py \
#       --input-dir /path/to/robwin_task_episodes \
#       --repo-id my_robotwin_task \
#       --fps 15 \
#       [--robot-type aloha_agilex] [--mode video|image] [--out-dir ...] [--max-episodes N]
#
# 依赖 lerobot(本仓 lingbot_vla_v2 环境) + h5py + numpy + opencv(解 JPEG)。

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# HDF5 源相机键 -> lerobot 目标相机键(与 pkl2hdf5.CAMERA_MAP + robotwin.yaml 对齐)
VISION_TO_LEROBOT_CAM = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

# 关节字段(左臂+左夹爪在前,右臂+右夹爪在后,与 _robot_info.json 的双臂定义一致)
JOINT_FIELDS = [
    ("left_arm_joint_states", "arm"),
    ("left_ee_joint_states", "ee"),
    ("right_arm_joint_states", "arm"),
    ("right_ee_joint_states", "ee"),
]


def _decode_jpeg(buf) -> np.ndarray:
    """把 pkl2hdf5 写入的 JPEG 字节解成 HWC uint8 RGB。"""
    import cv2

    if isinstance(buf, np.ndarray) and buf.ndim == 3:
        return buf  # 已是解码后的图像
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode 失败:字节流不是有效 JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_episode(ep_path: Path) -> dict:
    """读一个 RoboTwin HDF5 回合,返回 state/action/images/instructions。"""
    with h5py.File(ep_path, "r") as f:
        state_parts, action_parts = [], []
        for field, _kind in JOINT_FIELDS:
            if f"state/{field}" in f:
                state_parts.append(np.asarray(f[f"state/{field}"], dtype=np.float32))
                action_parts.append(np.asarray(f[f"action/{field}"], dtype=np.float32))
        if not state_parts:
            raise KeyError(f"{ep_path} 缺 state/*_joint_states")

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
    """建空 lerobot v3 数据集,schema 对齐本机 rotate_qrcode_joint_v30。"""
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

    # pyav 后端默认 vcodec=libsvtav1 在本机不可用 -> 显式用 h264(最通用)。
    # 注:DatasetWriter 即使 image 模式也会构造默认 RGBEncoderConfig 并在 av<15 上炸,
    # 所以无条件传 h264;真正编码只在 video 模式发生。
    from lerobot.configs.video import RGBEncoderConfig

    create_kwargs = {"rgb_encoder": RGBEncoderConfig(vcodec="h264")}

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=(mode == "video"),
        image_writer_processes=0,
        image_writer_threads=4,
        **create_kwargs,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboTwin HDF5 -> lerobot v3 数据集转换器")
    ap.add_argument("--input-dir", required=True, help="含 episode_*.hdf5 的目录(递归找)")
    ap.add_argument("--repo-id", required=True, help="lerobot repo_id(如 my_robotwin_task)")
    ap.add_argument("--out-dir", default=None, help="输出根目录(默认 HF_LEROBOT_HOME/repo_id)")
    ap.add_argument("--fps", type=int, default=15, help="采集频率(HDF5 里有 frequency 时以其为准)")
    ap.add_argument("--robot-type", default="aloha_agilex", help="robot_type 标签")
    ap.add_argument("--mode", choices=["video", "image"], default="video", help="图像存视频还是散图")
    ap.add_argument("--max-episodes", type=int, default=None, help="只转前 N 个回合(调试用)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    ep_files = sorted(input_dir.rglob("*.hdf5"))
    if not ep_files:
        raise FileNotFoundError(f"{input_dir} 下没有 *.hdf5 回合文件")
    if args.max_episodes is not None:
        ep_files = ep_files[: args.max_episodes]
    print(f"发现 {len(ep_files)} 个回合,开始转换 -> repo_id={args.repo_id}")

    # 先用第一个回合探测维度/图像形状/fps,再建数据集
    first = _load_episode(ep_files[0])
    action_dim = int(first["state"].shape[1])
    if not first["images"]:
        raise ValueError(f"{ep_files[0]} 没有任何相机图像,无法确定图像形状")
    any_cam = sorted(first["images"])[0]
    img_shape = first["images"][any_cam].shape[1:]  # (H, W, 3)
    fps = first["fps"] or args.fps
    print(f"探测: action_dim={action_dim}, img={img_shape}, fps={fps}, cams={list(first['images'])}")

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
                        # add_frame 收 HWC uint8(lerobot 内部自转 CHW 编码),别先 permute。
                        frame[f"observation.images.{cam}"] = torch.from_numpy(imgs[i])
                dataset.add_frame(frame)

            dataset.save_episode()
            n_frames += T
            print(f"  {ep_path.name}: {T} 帧")
        except Exception as e:  # noqa: BLE001 - 单回合失败不拖垮整批
            print(f"  {ep_path.name}: 失败({e}),跳过", file=sys.stderr)

    print(f"完成:{len(ep_files)} 回合 / {n_frames} 帧 -> {dataset.root}")
    print("下一步: 生成 norm_stats(--quantiles) + 转换器 --profile robotwin,即可 lerobot-train")


if __name__ == "__main__":
    main()
