#!/usr/bin/env python
"""Generate an end-effector trajectory video for the trained space-bar ACT policy."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

DEFAULT_POLICY_PATH = "outputs/train/act_spacebar_progress/checkpoints/last/pretrained_model"
DEFAULT_DATASET_REPO_ID = "local/so101_blind_task2_progress_v1"
DEFAULT_DATASET_ROOT = "outputs/datasets/so101_blind_task2_progress_v1"
DEFAULT_URDF_REPO_ID = "lehome/asset_challenge"
DEFAULT_URDF_FILE = "robots/so101_new_calib.urdf"
MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@dataclass
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default="press the space bar")
    parser.add_argument("--urdf-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/project4_ee_video"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def resolve_device(requested: str | None, policy_device: str | None) -> str:
    if requested and is_torch_device_available(requested):
        return requested
    if policy_device and is_torch_device_available(policy_device):
        return policy_device
    return auto_select_torch_device().type


def resolve_urdf(output_dir: Path, urdf_path: Path | None) -> Path:
    if urdf_path is not None:
        return urdf_path
    return Path(
        hf_hub_download(
            repo_id=DEFAULT_URDF_REPO_ID,
            filename=DEFAULT_URDF_FILE,
            repo_type="dataset",
            local_dir=output_dir / "assets",
        )
    )


def parse_float_triplet(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in value.split()], dtype=float)


def load_joint_specs(urdf_path: Path) -> dict[str, JointSpec]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        spec = JointSpec(
            name=str(joint.attrib["name"]),
            parent=str(parent.attrib["link"]),
            child=str(child.attrib["link"]),
            joint_type=str(joint.attrib.get("type", "fixed")),
            xyz=parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            rpy=parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
            axis=parse_float_triplet(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
        )
        joints[spec.child] = spec
    return joints


def chain_to_target(joint_specs_by_child: dict[str, JointSpec], base_link: str, target_link: str) -> list[JointSpec]:
    chain = []
    link = target_link
    while link != base_link:
        if link not in joint_specs_by_child:
            raise ValueError(f"Could not trace URDF chain from {target_link} back to {base_link}; stopped at {link}.")
        joint = joint_specs_by_child[link]
        chain.append(joint)
        link = joint.parent
    return list(reversed(chain))


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def transform_from_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def fk(chain: list[JointSpec], joint_degrees: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4)
    link_points = [transform[:3, 3].copy()]
    for joint in chain:
        transform = transform @ transform_from_xyz_rpy(joint.xyz, joint.rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            angle = np.deg2rad(joint_degrees.get(f"{joint.name}.pos", joint_degrees.get(joint.name, 0.0)))
            transform = transform @ transform_from_rotation(axis_angle_matrix(joint.axis, angle))
        link_points.append(transform[:3, 3].copy())
    return transform[:3, 3].copy(), np.stack(link_points)


def load_policy(policy_path: str, device: str):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def reset_inference(policy, preprocessor, postprocessor) -> None:
    for obj in (policy, preprocessor, postprocessor):
        reset = getattr(obj, "reset", None)
        if callable(reset):
            reset()


def select_action(policy, preprocessor, postprocessor, observation: dict, device: str, task: str) -> np.ndarray:
    obs = {key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()}
    obs = prepare_observation_for_inference(obs, torch.device(device), task=task)
    with torch.inference_mode():
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def episode_indices(dataset: LeRobotDataset, episode: int) -> list[int]:
    meta = dataset.meta.episodes[episode]
    return list(range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"])))


def named_joints(values: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(names)}


def compute_trajectories(args: argparse.Namespace, chain: list[JointSpec]):
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    device = resolve_device(args.device, policy_cfg.device)
    policy, preprocessor, postprocessor = load_policy(args.policy_path, device)
    reset_inference(policy, preprocessor, postprocessor)

    rows = dataset.hf_dataset.with_format(None)
    names = dataset.meta.features[ACTION]["names"]
    indices = episode_indices(dataset, args.episode)

    recorded_positions = []
    predicted_positions = []
    observed_positions = []
    recorded_skeletons = []
    predicted_skeletons = []
    predicted_actions = []
    recorded_actions = []

    for idx in indices:
        row = rows[int(idx)]
        observation = {OBS_STATE: row[OBS_STATE], OBS_ENV_STATE: row[OBS_ENV_STATE]}
        predicted_action = select_action(policy, preprocessor, postprocessor, observation, device, args.task)
        recorded_action = np.asarray(row[ACTION], dtype=np.float32)
        observed_state = np.asarray(row[OBS_STATE], dtype=np.float32)

        pred_pos, pred_skel = fk(chain, named_joints(predicted_action, names))
        rec_pos, rec_skel = fk(chain, named_joints(recorded_action, names))
        obs_pos, _ = fk(chain, named_joints(observed_state, dataset.meta.features[OBS_STATE]["names"]))

        predicted_positions.append(pred_pos)
        recorded_positions.append(rec_pos)
        observed_positions.append(obs_pos)
        predicted_skeletons.append(pred_skel)
        recorded_skeletons.append(rec_skel)
        predicted_actions.append(predicted_action)
        recorded_actions.append(recorded_action)

    return {
        "device": device,
        "names": names,
        "recorded_positions": np.stack(recorded_positions),
        "predicted_positions": np.stack(predicted_positions),
        "observed_positions": np.stack(observed_positions),
        "recorded_skeletons": np.stack(recorded_skeletons),
        "predicted_skeletons": np.stack(predicted_skeletons),
        "predicted_actions": np.stack(predicted_actions),
        "recorded_actions": np.stack(recorded_actions),
    }


def equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.6, 0.08)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_csv(output_path: Path, recorded: np.ndarray, predicted: np.ndarray, observed: np.ndarray, fps: int) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "time_s",
                "recorded_x",
                "recorded_y",
                "recorded_z",
                "predicted_x",
                "predicted_y",
                "predicted_z",
                "observed_x",
                "observed_y",
                "observed_z",
            ]
        )
        for frame in range(len(predicted)):
            writer.writerow(
                [
                    frame,
                    frame / fps,
                    *recorded[frame].tolist(),
                    *predicted[frame].tolist(),
                    *observed[frame].tolist(),
                ]
            )


def make_animation(args: argparse.Namespace, trajectories: dict, output_path: Path) -> None:
    predicted = trajectories["predicted_positions"]
    recorded = trajectories["recorded_positions"]
    observed = trajectories["observed_positions"]
    pred_skeletons = trajectories["predicted_skeletons"]
    rec_skeletons = trajectories["recorded_skeletons"]
    frames = len(predicted)
    time_axis = np.arange(frames) / args.fps

    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xyz = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    all_points = np.concatenate([predicted, recorded, observed], axis=0)
    equal_axes(ax3d, all_points)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title(f"SO101 End-Effector Path - Episode {args.episode}")
    ax3d.view_init(elev=28, azim=-58)

    rec_path, = ax3d.plot([], [], [], color="#7a7a7a", linewidth=2.0, label="recorded action EE")
    pred_path, = ax3d.plot([], [], [], color="#f28e2b", linewidth=2.4, label="ACT predicted EE")
    obs_path, = ax3d.plot([], [], [], color="#4e79a7", linewidth=1.6, alpha=0.5, label="observed state EE")
    rec_skel, = ax3d.plot([], [], [], "o-", color="#b0b0b0", linewidth=1.0, markersize=3, alpha=0.55)
    pred_skel, = ax3d.plot([], [], [], "o-", color="#f28e2b", linewidth=1.2, markersize=3)
    current = ax3d.scatter([], [], [], color="#d62728", s=36, depthshade=True)
    ax3d.legend(loc="upper left")

    colors = {"x": "#4e79a7", "y": "#59a14f", "z": "#e15759"}
    xyz_lines = {}
    for dim, key in enumerate(("x", "y", "z")):
        ax_xyz.plot(time_axis, recorded[:, dim], color=colors[key], alpha=0.35, linestyle="--", linewidth=1.0)
        line, = ax_xyz.plot([], [], color=colors[key], linewidth=1.7, label=f"pred {key}")
        xyz_lines[key] = (dim, line)
    cursor_xyz = ax_xyz.axvline(0, color="black", alpha=0.25, linewidth=1)
    ax_xyz.set_title("Predicted EE position; dashed = recorded")
    ax_xyz.set_xlabel("time (s)")
    ax_xyz.set_ylabel("meters")
    ax_xyz.legend(loc="upper right", ncol=3)
    ax_xyz.grid(alpha=0.25)

    error = np.linalg.norm(predicted - recorded, axis=1)
    err_line, = ax_err.plot([], [], color="#d62728", linewidth=1.8)
    cursor_err = ax_err.axvline(0, color="black", alpha=0.25, linewidth=1)
    ax_err.set_xlim(0, time_axis[-1])
    ax_err.set_ylim(0, max(float(error.max()) * 1.15, 0.02))
    ax_err.set_title("EE distance: predicted vs recorded")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("meters")
    ax_err.grid(alpha=0.25)

    subtitle = fig.text(0.02, 0.02, "", fontsize=10)

    def update(frame: int):
        upto = frame + 1
        rec_path.set_data(recorded[:upto, 0], recorded[:upto, 1])
        rec_path.set_3d_properties(recorded[:upto, 2])
        pred_path.set_data(predicted[:upto, 0], predicted[:upto, 1])
        pred_path.set_3d_properties(predicted[:upto, 2])
        obs_path.set_data(observed[:upto, 0], observed[:upto, 1])
        obs_path.set_3d_properties(observed[:upto, 2])

        rec = rec_skeletons[frame]
        pred = pred_skeletons[frame]
        rec_skel.set_data(rec[:, 0], rec[:, 1])
        rec_skel.set_3d_properties(rec[:, 2])
        pred_skel.set_data(pred[:, 0], pred[:, 1])
        pred_skel.set_3d_properties(pred[:, 2])
        current._offsets3d = ([predicted[frame, 0]], [predicted[frame, 1]], [predicted[frame, 2]])

        for _, (dim, line) in xyz_lines.items():
            line.set_data(time_axis[:upto], predicted[:upto, dim])
        ax_xyz.set_xlim(0, time_axis[-1])
        xyz_min = min(float(predicted.min()), float(recorded.min()))
        xyz_max = max(float(predicted.max()), float(recorded.max()))
        pad = max((xyz_max - xyz_min) * 0.08, 0.01)
        ax_xyz.set_ylim(xyz_min - pad, xyz_max + pad)
        cursor_xyz.set_xdata([time_axis[frame], time_axis[frame]])

        err_line.set_data(time_axis[:upto], error[:upto])
        cursor_err.set_xdata([time_axis[frame], time_axis[frame]])
        subtitle.set_text(
            f"frame {frame:03d}/{frames - 1}   t={time_axis[frame]:.2f}s   "
            f"EE error={error[frame]:.4f} m"
        )
        return [
            rec_path,
            pred_path,
            obs_path,
            rec_skel,
            pred_skel,
            current,
            *[line for _, line in xyz_lines.values()],
            err_line,
            cursor_xyz,
            cursor_err,
            subtitle,
        ]

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=2400)
    animation.save(output_path, writer=writer, dpi=args.dpi)
    update(frames - 1)
    fig.savefig(output_path.with_suffix(".png"), dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = resolve_urdf(args.output_dir, args.urdf_path)
    chain = chain_to_target(load_joint_specs(urdf_path), "base_link", "gripper_frame_link")
    trajectories = compute_trajectories(args, chain)

    output_path = args.output_dir / f"act_spacebar_episode_{args.episode:03d}_ee_position.mp4"
    make_animation(args, trajectories, output_path)
    csv_path = output_path.with_suffix(".csv")
    save_csv(
        csv_path,
        trajectories["recorded_positions"],
        trajectories["predicted_positions"],
        trajectories["observed_positions"],
        args.fps,
    )
    print(f"video={output_path}")
    print(f"csv={csv_path}")
    print(f"device={trajectories['device']}")
    print(f"urdf={urdf_path}")


if __name__ == "__main__":
    main()
