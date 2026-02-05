# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr
import torch

from .constants import OBS_PREFIX, OBS_STR
import xml.etree.ElementTree as ET


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return (
        isinstance(x, float)
        or isinstance(x, numbers.Real)
        or isinstance(x, (np.integer, np.floating))
        or (isinstance(x, np.ndarray) and x.ndim == 0)
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalar values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


def transform_from_pose(p):
    t = p[:3]
    w = p[3:6]

    ang = float(np.linalg.norm(w))
    if ang < 1e-12:  # no rotation
        axis = np.array([1.0, 0.0, 0.0])  # arbitrary
        R = np.eye(3)
    else:
        axis = (w / ang).astype(float)
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_rerun_action_chunk(action_chunk: torch.Tensor, name="action_chunk_", prefix="leader"):
    """Log action chunk to rerun for visualization.

    Supports both single-arm and bimanual setups:
    - Single-arm: 7 elements (x, y, z, wx, wy, wz, gripper)
    - Bimanual: 14 elements (left_x, left_y, left_z, left_wx, left_wy, left_wz, left_gripper,
                              right_x, right_y, right_z, right_wx, right_wy, right_wz, right_gripper)

    Args:
        action_chunk: Tensor of shape (batch, action_dim) where action_dim is 7 or 14
        name: Name prefix for the rerun log entries
        prefix: Prefix for the rerun entity path (e.g., "leader" or "follower")
    """
    action_chunk = action_chunk.cpu().numpy()
    for i, action in enumerate(action_chunk):
        action_dim = len(action)

        if action_dim == 14:
            # Bimanual setup: split into left and right
            left_action = action[:6]  # x, y, z, wx, wy, wz (skip gripper at index 6)
            right_action = action[7:13]  # x, y, z, wx, wy, wz (skip gripper at index 13)

            T_left = transform_from_pose(left_action)
            T_right = transform_from_pose(right_action)

            offset_matrix = np.eye(4)
            offset_matrix[1, 3] = 0.2 # TODO: this is bad should not be hardcoded?
            T_right = apply_offset(T_right, offset_matrix)

            rr.log(f"{prefix}_left/robot/base_link/{name}{i}",
                   rr.Transform3D(translation=T_left[:3, 3], mat3x3=T_left[:3, :3], axis_length=0.1))
            rr.log(f"{prefix}_right/robot/base_link/{name}{i}",
                   rr.Transform3D(translation=T_right[:3, 3], mat3x3=T_right[:3, :3], axis_length=0.1))

        elif action_dim == 7:
            # Single-arm setup: only one EE pose
            single_action = action[:6]  # x, y, z, wx, wy, wz (skip gripper at index 6)
            T = transform_from_pose(single_action)

            rr.log(f"{prefix}/robot/base_link/{name}{i}",
                   rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3], axis_length=0.1))

        else:
            # Unsupported action dimension, skip logging with a warning
            import warnings
            warnings.warn(f"Unsupported action dimension {action_dim} for log_rerun_action_chunk. Expected 7 or 14.")
            break


def parse_urdf_graph(urdf_path: str):
    """
    Returns a dict with:
      - links: set[str]
      - joints: list[str]
      - child_to_parent_link: {child_link: parent_link}
      - child_to_joint: {child_link: joint_name}
      - joint_to_parent_link: {joint_name: parent_link}
      - joint_to_child_link: {joint_name: child_link}
      - root_links: [link_name, ...]
    """
    tree = ET.parse(urdf_path)
    r = tree.getroot()

    links = {e.attrib["name"] for e in r.findall("link")}
    joints = []
    child_to_parent_link = {}
    child_to_joint = {}
    joint_to_parent_link = {}
    joint_to_child_link = {}

    for j in r.findall("joint"):
        jname = j.attrib["name"]
        parent_link = j.find("parent").attrib["link"]
        child_link = j.find("child").attrib["link"]
        joints.append(jname)
        child_to_parent_link[child_link] = parent_link
        child_to_joint[child_link] = jname
        joint_to_parent_link[jname] = parent_link
        joint_to_child_link[jname] = child_link

    root_links = [L for L in links if L not in child_to_parent_link]
    return {
        "links": links,
        "joints": joints,
        "child_to_parent_link": child_to_parent_link,
        "child_to_joint": child_to_joint,
        "joint_to_parent_link": joint_to_parent_link,
        "joint_to_child_link": joint_to_child_link,
        "root_links": root_links,
    }


# --- Path builders (match rerun-urdf-loader layout) ----------------------
def link_entity_path(link: str, G, prefix: str = "robot") -> str:
    """
    <prefix>/<root_link>/<joint_to_child>/<child_link>/.../<joint_to_target>/<target_link>
    """
    chain = []
    cur = link
    while cur in G["child_to_parent_link"]:
        parent = G["child_to_parent_link"][cur]
        joint = G["child_to_joint"][cur]
        chain.append((parent, joint, cur))
        cur = parent  # climb to root

    parts = [prefix, cur]  # cur is now a root link
    for _, joint, child in reversed(chain):
        parts.extend([joint, child])
    return "/".join(parts)


def joint_entity_path(joint: str, G, prefix: str = "robot") -> str:
    """
    Path to the joint node itself:
    link_entity_path(parent_link) + '/' + joint_name
    """
    parent_link = G["joint_to_parent_link"][joint]
    return f"{link_entity_path(parent_link, G, prefix)}/{joint}"


def _inv_SE3(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    RT = R.T
    Ti[:3, :3] = RT
    Ti[:3, 3] = -RT @ t
    return Ti


def _rel_from_world(Tw_parent, Tw_child):
    return _inv_SE3(Tw_parent) @ Tw_child


def _log_transform(path: str, pos: np.ndarray, rot: np.ndarray, axis_len: float = 0.02):
    # New Rerun API (mat3x3 / axis_length), with backward-compat fallback
    try:
        rr.log(path, rr.Transform3D(translation=pos, mat3x3=rot, axis_length=axis_len))
    except TypeError:
        rr.log(path, rr.Transform3D(translation=pos, rotation=rr.Rotation3D.from_matrix(rot)))


def apply_offset(T: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Applies an SE(3) offset: T' = offset @ T"""
    return offset @ T


# --- 3) Drive the loaded URDF hierarchy (updates the rig the loader created) ----
def drive_urdf_with_world_poses(
    placo_robot, G, step: int, prefix: str = "robot", offset: np.ndarray = np.eye(4)
):
    """
    - Logs root link world transforms at their link path.
    - Logs parent-relative transforms for each JOINT at the joint path:
        path: <prefix>/<parent_link>/<joint_name>
        value: T_parent_link^child_link
    This matches the loader's joint-in-the-middle hierarchy.
    """
    rr.set_time_sequence("frame", step)

    # Cache Placo world transforms
    T_world = {}
    for name in placo_robot.frame_names():
        try:
            T_world[name] = apply_offset(placo_robot.get_T_world_frame(name), offset)
        except Exception:
            pass

    # 1) Roots: write world pose to the root link path(s)
    for root_link in G["root_links"]:
        if root_link in T_world:
            Tw = T_world[root_link]
            _log_transform(link_entity_path(root_link, G, prefix), Tw[:3, 3], Tw[:3, :3])

    # 2) Joints: write parent-relative pose to the JOINT entity
    for j in G["joints"]:
        parent = G["joint_to_parent_link"][j]
        child = G["joint_to_child_link"][j]
        if parent not in T_world or child not in T_world:
            continue
        Tpc = _rel_from_world(T_world[parent], T_world[child])
        _log_transform(joint_entity_path(j, G, prefix), Tpc[:3, 3], Tpc[:3, :3])


def visualize_robot(
    robot,
    step: int = 0,
    urdf_prefix: str = "follower/robot",
    urdf_graph: dict = None,
    offset: np.ndarray = np.eye(4),
):
    """
    - update_urdf=False (default): plot flat, world-aligned poses under `flat_prefix/<frame>`.
    - update_urdf=True: drive the URDF rig (requires parent_map and correct urdf_prefix).
    """
    drive_urdf_with_world_poses(robot, urdf_graph, step, prefix=urdf_prefix, offset=offset)
