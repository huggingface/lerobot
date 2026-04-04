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

"""
Heuristic grasp planner that converts 3D object state into a sequence of
end-effector waypoints consumable by the Placo IK solver.

The planner supports several grasp strategies selected by object label and
produces 4x4 homogeneous transforms in the **robot base frame**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Keyword sets for strategy selection
_TOP_GRASP_KEYWORDS = {"cube", "box", "block", "brick", "dice", "square", "lego"}
_SIDE_GRASP_KEYWORDS = {"cup", "mug", "cylinder", "bottle", "can", "glass", "jar", "container"}
_PINCH_GRASP_KEYWORDS = {"ball", "sphere", "marble", "bead", "round", "small", "coin", "pen", "pencil"}


@dataclass
class Waypoint:
    """A single end-effector target in a manipulation plan."""

    pose_4x4: np.ndarray
    gripper_open: bool
    label: str
    gripper_width_pct: float = 100.0


@dataclass
class InteractionStrategy:
    """Tunable heuristics for physical interactions (no force sensing assumed)."""

    # Approach / grasp
    grasp_clearance_m: float = 0.008
    """How far above the object top to stop before closing (meters)."""
    grasp_seat_m: float = 0.004
    """Extra downward motion after closing to "seat" the grasp (meters)."""
    min_top_extent_m: float = 0.008
    """Minimum assumed half-height when size is noisy (meters)."""

    # Place / drop
    place_clearance_m: float = 0.010
    """How far above target Z to open the gripper (meters)."""
    place_settle_pause_s: float = 0.12
    """Pause after opening so object can detach (seconds)."""


def _make_pose(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position (3,) and rotation (3,3)."""
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rotation_matrix
    tf[:3, 3] = position
    return tf


def _top_down_rotation() -> np.ndarray:
    """Rotation matrix for a top-down approach (gripper pointing straight down).

    Convention: Z-axis of gripper points down (-Z_world), X-axis points forward.
    """
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    return rot


def _side_approach_rotation(approach_dir: np.ndarray | None = None) -> np.ndarray:
    """Rotation matrix for a side approach (gripper horizontal).

    Default approach direction is along +X (from the front).
    """
    if approach_dir is None:
        approach_dir = np.array([1.0, 0.0, 0.0])
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-8)
    z_axis = -approach_dir
    y_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return rot


def _classify_object(label: str) -> str:
    """Map an object label to a grasp strategy name."""
    label_lower = label.lower()
    for kw in _TOP_GRASP_KEYWORDS:
        if kw in label_lower:
            return "top_down"
    for kw in _SIDE_GRASP_KEYWORDS:
        if kw in label_lower:
            return "side"
    for kw in _PINCH_GRASP_KEYWORDS:
        if kw in label_lower:
            return "pinch"
    return "top_down"


def _gripper_pct_for_width(width_m: float, max_aperture_m: float = 0.08) -> float:
    """Map object width (m) to closed-loop gripper % (0=closed, 100=open).

    Depth/VLM noise can inflate width; clamp so the gripper does not stay barely cracked
    or slam fully open.
    """
    w = float(np.clip(width_m, 0.012, 0.09))
    grip_target = max(0.0, w - 0.009)
    pct = (grip_target / max_aperture_m) * 100.0
    return float(np.clip(pct, 10.0, 92.0))


def _pregrasp_gripper_pct(close_pct: float) -> float:
    """Opening before grasp: enough clearance over the final close width, not full 100%."""
    return float(np.clip(close_pct + 18.0, 36.0, 78.0))


class GraspPlanner:
    """Heuristic grasp planner for tabletop manipulation.

    Converts object 3D state (position, size, label) into a sequence of
    end-effector waypoints that the IK solver can execute.

    Args:
        camera_to_robot_tf: 4x4 transform from camera frame to robot base frame.
            If ``None``, identity is used (camera frame = robot frame).
        pre_grasp_height_m: How far above the object to place the pre-grasp waypoint.
        lift_height_m: How far to lift after grasping.
        max_gripper_aperture_m: Physical max opening of the gripper in meters.
        default_orientation: Default grasp approach rotation matrix (3x3).
            If ``None``, top-down is used.
    """

    def __init__(
        self,
        camera_to_robot_tf: np.ndarray | None = None,
        pre_grasp_height_m: float = 0.06,
        lift_height_m: float = 0.08,
        max_gripper_aperture_m: float = 0.08,
        arm_base_xyz_m: np.ndarray | None = None,
        min_reach_m: float = 0.10,
        max_reach_m: float = 0.42,
        min_grasp_z_m: float = 0.01,
        max_grasp_z_m: float = 0.35,
        approach_xy_retract_m: float = 0.055,
        interaction: InteractionStrategy | None = None,
        default_orientation: np.ndarray | None = None,
    ):
        self.cam_to_robot_tf = camera_to_robot_tf if camera_to_robot_tf is not None else np.eye(4)
        self.pre_grasp_height = pre_grasp_height_m
        self.lift_height = lift_height_m
        self.max_aperture = max_gripper_aperture_m
        self.approach_xy_retract_m = float(approach_xy_retract_m)
        self.arm_base_xyz = (
            np.asarray(arm_base_xyz_m, dtype=np.float64)
            if arm_base_xyz_m is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.min_reach_m = float(min_reach_m)
        self.max_reach_m = float(max_reach_m)
        self.min_grasp_z_m = float(min_grasp_z_m)
        self.max_grasp_z_m = float(max_grasp_z_m)
        self.interaction = interaction or InteractionStrategy()
        self.default_orientation = default_orientation if default_orientation is not None else _top_down_rotation()

    def set_camera_to_robot_tf(self, tf: np.ndarray) -> None:
        """Update camera→base transform (e.g. eye-in-hand: FK(q) @ T_ee_cam)."""
        self.cam_to_robot_tf = np.asarray(tf, dtype=np.float64)

    def _to_robot_frame(self, point_cam: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to robot base frame."""
        p_hom = np.array([*point_cam, 1.0], dtype=np.float64)
        p_robot = (self.cam_to_robot_tf @ p_hom)[:3]
        return p_robot

    def point_camera_to_base(self, xyz_cam: np.ndarray) -> np.ndarray:
        """Map a point from the camera optical frame to the robot base frame (meters)."""
        v = np.asarray(xyz_cam, dtype=np.float64).reshape(3)
        return self._to_robot_frame(v)

    def half_extents_base_from_size_cam(self, size_xyz_cam: np.ndarray) -> np.ndarray:
        """Approximate axis-aligned half-extents in base frame from size in camera frame."""
        R = self.cam_to_robot_tf[:3, :3]
        s = np.asarray(size_xyz_cam, dtype=np.float64).reshape(3)
        return 0.5 * np.abs(R @ s)

    def _project_to_reachable_workspace(self, center_robot: np.ndarray) -> np.ndarray:
        """Project a target point into a simple annular XY workspace and Z bounds.

        This basic geometry keeps targets within a practical grasp envelope:
        - XY radius around arm base is clamped to [min_reach_m, max_reach_m].
        - Z is clamped to [min_grasp_z_m, max_grasp_z_m].
        """
        target = np.asarray(center_robot, dtype=np.float64).copy()
        rel_xy = target[:2] - self.arm_base_xyz[:2]
        radius = float(np.linalg.norm(rel_xy))
        if radius < 1e-8:
            rel_xy = np.array([1.0, 0.0], dtype=np.float64)
            radius = 1.0
        unit_xy = rel_xy / radius
        clamped_radius = float(np.clip(radius, self.min_reach_m, self.max_reach_m))
        target[:2] = self.arm_base_xyz[:2] + unit_xy * clamped_radius
        target[2] = float(np.clip(target[2], self.min_grasp_z_m, self.max_grasp_z_m))
        return target

    def plan_pick(
        self,
        obj_center_xyz: np.ndarray,
        obj_size_xyz: np.ndarray,
        obj_label: str,
    ) -> list[Waypoint]:
        """Plan a pick sequence for a single object.

        Args:
            obj_center_xyz: (3,) object center in camera frame (meters).
            obj_size_xyz: (3,) object bounding-box extents in camera frame (meters).
            obj_label: Object label string from VLM.

        Returns:
            Ordered list of Waypoints. The nominal sequence is:
                approach_far -> pre_grasp -> mid_grasp -> grasp -> post_grasp -> lift_far
            Some of these may collapse together depending on configured heights.
        """
        center_robot_raw = self._to_robot_frame(np.asarray(obj_center_xyz, dtype=np.float64))
        center_robot = self._project_to_reachable_workspace(center_robot_raw)
        size_cam = np.asarray(obj_size_xyz, dtype=np.float64).reshape(3)
        strategy = _classify_object(obj_label)

        logger.debug(
            "Planning pick %r raw=%s projected=%s strategy=%s",
            obj_label,
            center_robot_raw,
            center_robot,
            strategy,
        )
        logger.info(
            "[plan] pick %r %s │ base xyz=(%.3f, %.3f, %.3f) m │ |raw|=%.3f m",
            obj_label,
            strategy,
            float(center_robot[0]),
            float(center_robot[1]),
            float(center_robot[2]),
            float(np.linalg.norm(center_robot_raw)),
        )

        if strategy == "top_down":
            orientation = _top_down_rotation()
            grasp_width = float(max(size_cam[0], size_cam[1]))
        elif strategy == "side":
            approach = center_robot.copy()
            approach[2] = 0.0
            if np.linalg.norm(approach) > 1e-3:
                approach = approach / np.linalg.norm(approach)
            else:
                approach = np.array([1.0, 0.0, 0.0])
            orientation = _side_approach_rotation(approach)
            grasp_width = (
                float(size_cam[2]) if size_cam[2] > 0.01 else float(max(size_cam[0], size_cam[1]))
            )
        elif strategy == "pinch":
            orientation = _top_down_rotation()
            grasp_width = float(min(size_cam[0], size_cam[1]))
        else:
            orientation = self.default_orientation
            grasp_width = float(max(size_cam[0], size_cam[1]))

        grip_close_pct = _gripper_pct_for_width(grasp_width, self.max_aperture)
        pre_open_pct = _pregrasp_gripper_pct(grip_close_pct)

        # Compute a safe "interaction surface" in base Z.
        # For top-down grasps, the prior implementation targeted the *object center*, which tends to slam
        # the tool into the object. Instead, target the object's top surface + clearance.
        R = self.cam_to_robot_tf[:3, :3]
        size_base = np.abs(R @ size_cam.reshape(3))
        half_z = float(0.5 * size_base[2])
        half_z = max(half_z, float(self.interaction.min_top_extent_m))
        top_z = float(center_robot[2] + half_z)

        grasp_pos = center_robot.copy()
        if strategy in ("top_down", "pinch"):
            grasp_pos[2] = float(
                np.clip(
                    top_z + float(self.interaction.grasp_clearance_m),
                    self.min_grasp_z_m,
                    self.max_grasp_z_m,
                )
            )
        else:
            # Side grasps keep the original center target (still projected into workspace).
            grasp_pos = center_robot.copy()

        # High approach, optionally retracted in XY toward the arm base, then align XY, then descend.
        approach_far_pos = center_robot.copy()
        approach_far_pos[2] += max(self.pre_grasp_height * 2.0, self.pre_grasp_height + 0.04)
        retract_m = self.approach_xy_retract_m
        d_xy = self.arm_base_xyz[:2] - center_robot[:2]
        nd_xy = float(np.linalg.norm(d_xy))
        if retract_m > 1e-6 and nd_xy > 1e-6:
            u_xy = d_xy / nd_xy
            approach_far_pos[:2] = center_robot[:2] + u_xy * retract_m

        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += float(self.pre_grasp_height)

        mid_grasp_pos = grasp_pos.copy()
        mid_grasp_pos[2] += float(max(self.pre_grasp_height * 0.35, 0.018))

        waypoints: list[Waypoint] = []

        wp_approach_far = Waypoint(
            pose_4x4=_make_pose(approach_far_pos, orientation),
            gripper_open=True,
            gripper_width_pct=pre_open_pct,
            label="approach_far",
        )
        waypoints.append(wp_approach_far)

        if retract_m > 1e-6 and nd_xy > 1e-6:
            approach_align_pos = center_robot.copy()
            approach_align_pos[2] = float(approach_far_pos[2])
            wp_align = Waypoint(
                pose_4x4=_make_pose(approach_align_pos, orientation),
                gripper_open=True,
                gripper_width_pct=pre_open_pct,
                label="approach_align_xy",
            )
            waypoints.append(wp_align)

        wp_pre = Waypoint(
            pose_4x4=_make_pose(pre_grasp_pos, orientation),
            gripper_open=True,
            gripper_width_pct=pre_open_pct,
            label="pre_grasp",
        )
        waypoints.append(wp_pre)

        wp_mid = Waypoint(
            pose_4x4=_make_pose(mid_grasp_pos, orientation),
            gripper_open=True,
            gripper_width_pct=pre_open_pct,
            label="mid_grasp",
        )
        waypoints.append(wp_mid)

        # Interaction: do NOT close while descending into the object. Instead:
        # 1) stop at a hover grasp pose (top surface + clearance) while open
        # 2) close in place (no translation) to actually use the gripper
        # 3) optionally "seat" downward a few mm after closing
        wp_grasp_hover = Waypoint(
            pose_4x4=_make_pose(grasp_pos.copy(), orientation),
            gripper_open=True,
            gripper_width_pct=pre_open_pct,
            label="grasp_hover",
        )
        waypoints.append(wp_grasp_hover)

        wp_close = Waypoint(
            pose_4x4=_make_pose(grasp_pos.copy(), orientation),
            gripper_open=False,
            gripper_width_pct=grip_close_pct,
            label="close_gripper",
        )
        waypoints.append(wp_close)

        if strategy in ("top_down", "pinch") and float(self.interaction.grasp_seat_m) > 0:
            seat_pos = grasp_pos.copy()
            seat_pos[2] = float(
                np.clip(
                    seat_pos[2] - float(self.interaction.grasp_seat_m),
                    self.min_grasp_z_m,
                    self.max_grasp_z_m,
                )
            )
            waypoints.append(
                Waypoint(
                    pose_4x4=_make_pose(seat_pos, orientation),
                    gripper_open=False,
                    gripper_width_pct=grip_close_pct,
                    label="seat_grasp",
                )
            )

        post_grasp_pos = center_robot.copy()
        post_grasp_pos[2] += self.lift_height
        wp_post = Waypoint(
            pose_4x4=_make_pose(post_grasp_pos, orientation),
            gripper_open=False,
            gripper_width_pct=grip_close_pct,
            label="post_grasp",
        )
        waypoints.append(wp_post)

        lift_far_pos = center_robot.copy()
        lift_far_pos[2] += max(self.lift_height * 1.8, self.lift_height + 0.04)
        wp_lift_far = Waypoint(
            pose_4x4=_make_pose(lift_far_pos, orientation),
            gripper_open=False,
            gripper_width_pct=grip_close_pct,
            label="lift_far",
        )
        waypoints.append(wp_lift_far)

        return waypoints

    def plan_place(
        self,
        target_xyz: np.ndarray,
        approach_height_m: float | None = None,
    ) -> list[Waypoint]:
        """Plan a place sequence at a target location.

        Args:
            target_xyz: (3,) target position in camera frame (meters).
            approach_height_m: Height above target for approach. Defaults to pre_grasp_height.

        Returns:
            Ordered list of Waypoints: pre_place -> place -> retreat.
        """
        target_robot_raw = self._to_robot_frame(np.asarray(target_xyz, dtype=np.float64))
        target_robot = self._project_to_reachable_workspace(target_robot_raw)
        approach_h = approach_height_m if approach_height_m is not None else self.pre_grasp_height
        orientation = _top_down_rotation()

        pre_place_pos = target_robot.copy()
        pre_place_pos[2] += approach_h
        wp_pre = Waypoint(
            pose_4x4=_make_pose(pre_place_pos, orientation),
            gripper_open=False,
            gripper_width_pct=0.0,
            label="pre_place",
        )

        place_pos = target_robot.copy()
        place_pos[2] = float(
            np.clip(
                place_pos[2] + float(self.interaction.place_clearance_m),
                self.min_grasp_z_m,
                self.max_grasp_z_m,
            )
        )
        wp_place = Waypoint(
            pose_4x4=_make_pose(place_pos, orientation),
            gripper_open=True,
            gripper_width_pct=88.0,
            label="place",
        )

        retreat_pos = target_robot.copy()
        retreat_pos[2] += self.lift_height
        wp_retreat = Waypoint(
            pose_4x4=_make_pose(retreat_pos, orientation),
            gripper_open=True,
            gripper_width_pct=88.0,
            label="retreat",
        )

        return [wp_pre, wp_place, wp_retreat]

    def plan_pick_and_place(
        self,
        pick_center_xyz: np.ndarray,
        pick_size_xyz: np.ndarray,
        pick_label: str,
        place_target_xyz: np.ndarray,
    ) -> list[Waypoint]:
        """Plan a full pick-and-place sequence.

        Returns:
            Combined list: pick waypoints + place waypoints.
        """
        pick_wps = self.plan_pick(pick_center_xyz, pick_size_xyz, pick_label)
        place_wps = self.plan_place(place_target_xyz)
        return pick_wps + place_wps
