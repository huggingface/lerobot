#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Scripted, human-like joint trajectories: minimum-jerk segments with randomized timing and paths.

A ``Planner`` appends one joint command per control step. Tasks compose its moves (``transit``,
``linear_move``, ``gripper``, ``dwell``, and the composite ``grasp``/``place``) and name the subtask
each command belongs to.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import MotionConfig
from .randomization import uniform
from .scene import ObjectParams
from .so101 import ARM_JOINTS, GRIPPER, WRIST_ROLL, Arm

DOWN = np.array([0.0, 0.0, -1.0])
# A command that misses its Cartesian target by more than this is unreachable.
IK_TOLERANCE = 0.004


class PlanningError(Exception):
    """The scripted motion can't be executed in this scene: draw another one."""


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk time scaling: zero velocity and acceleration at both ends."""
    return s**3 * (10 - 15 * s + 6 * s**2)


@dataclass
class Grasp:
    """Where the gripper site goes to hold an object, and the jaws' direction and the object's width."""

    pos: np.ndarray
    approach: np.ndarray
    jaw: np.ndarray
    width: float
    # Extra room each jaw keeps from the object on the way down, for a tilted approach.
    clearance: float
    # Height of the gripper site above the object's bottom once it holds it.
    height: float
    # Approach tilt from vertical and the expected wrist roll, in degrees.
    tilt: float
    roll: float


@dataclass
class Plan:
    commands: np.ndarray
    subtask_of_frame: np.ndarray
    subtasks: list[str]
    # Named frame indices, such as "lifted", for checks during execution.
    marks: dict[str, int] = field(default_factory=dict)


class Planner:
    def __init__(
        self,
        arm: Arm,
        start: np.ndarray,
        fps: int,
        cfg: MotionConfig,
        rng: np.random.Generator,
        workspace: tuple[float, float] = (0.17, 0.30),
    ):
        self.arm = arm
        # Distances from the base where objects may be put down.
        self.workspace = workspace
        self.q = np.array(start, dtype=np.float64)
        self.fps = fps
        self.cfg = cfg
        self.rng = rng
        self.speed = uniform(rng, cfg.speed)
        self.frames: list[np.ndarray] = []
        self.subtask_ids: list[int] = []
        self.subtasks: list[str] = []
        self.marks: dict[str, int] = {}

    def subtask(self, label: str) -> None:
        """Commands from now on belong to a new subtask."""
        self.subtasks.append(label)

    def mark(self, name: str) -> None:
        self.marks[name] = len(self.frames)

    def plan(self) -> Plan:
        return Plan(
            commands=np.stack(self.frames),
            subtask_of_frame=np.asarray(self.subtask_ids),
            subtasks=list(self.subtasks),
            marks=dict(self.marks),
        )

    def _emit(self, q: np.ndarray) -> None:
        self.q = np.array(q, dtype=np.float64)
        self.frames.append(self.q.copy())
        self.subtask_ids.append(max(len(self.subtasks) - 1, 0))

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def _segment_speed(self) -> float:
        return self.speed * uniform(self.rng, self.cfg.segment_speed)

    # Primitive moves.

    def dwell(self, seconds: float | None = None) -> None:
        seconds = uniform(self.rng, self.cfg.dwell) if seconds is None else seconds
        for _ in range(int(round(seconds * self.fps))):
            self._emit(self.q)

    def gripper(self, value: float) -> None:
        """Open or close the gripper, the arm held still."""
        start = self.q[GRIPPER]
        n = self._steps(self.rng.uniform(0.3, 0.7) / self.speed)
        for s in min_jerk(np.arange(1, n + 1) / n):
            q = self.q.copy()
            q[GRIPPER] = start + s * (value - start)
            self._emit(q)

    def transit(self, target: np.ndarray, gripper: float | None = None) -> None:
        """Joint-space minimum-jerk move to ``target``, its path bent upwards through a random via point."""
        start = self.q.copy()
        target = np.array(target, dtype=np.float64)
        target[GRIPPER] = start[GRIPPER] if gripper is None else gripper
        distance = np.max(np.abs(target[:ARM_JOINTS] - start[:ARM_JOINTS]))
        seconds = max(0.5, distance / (uniform(self.rng, self.cfg.joint_speed) * self._segment_speed()))
        n = self._steps(seconds)
        # A quadratic Bezier passes through (start + 2 control + target) / 4 halfway: aim that at the via.
        middle = (start + target) / 2
        control = 2 * self._via(middle) - middle
        for s in min_jerk(np.arange(1, n + 1) / n):
            q = (1 - s) ** 2 * start + 2 * s * (1 - s) * control + s**2 * target
            q[GRIPPER] = start[GRIPPER] + s * (target[GRIPPER] - start[GRIPPER])
            self._emit(q)

    def _via(self, middle: np.ndarray) -> np.ndarray:
        """``middle`` with the gripper raised by a random height, or unchanged when that is unreachable."""
        pos, rot = self.arm.tcp_pose(middle)
        raised = pos + np.array([0.0, 0.0, uniform(self.rng, self.cfg.via_height)])
        q, err = self.arm.ik(raised, rot[:, 0], None, middle)
        return q if err < 0.01 else middle

    def linear_move(self, pos: np.ndarray, approach: np.ndarray, jaw: np.ndarray | None) -> None:
        """Move the gripper site along a straight line, solving IK at every step."""
        start, _ = self.arm.tcp_pose(self.q)
        distance = float(np.linalg.norm(pos - start))
        seconds = max(0.3, distance / (uniform(self.rng, self.cfg.linear_speed) * self._segment_speed()))
        n = self._steps(seconds)
        q = self.q.copy()
        for s in min_jerk(np.arange(1, n + 1) / n):
            q, err = self.arm.ik(start + s * (pos - start), approach, jaw, q)
            if err > IK_TOLERANCE:
                raise PlanningError(f"Linear move out of reach by {err * 1000:.1f} mm")
            self._emit(q)

    def solve(
        self,
        pos: np.ndarray,
        approach: np.ndarray,
        jaw: np.ndarray | None,
        init: np.ndarray | None = None,
        roll: float | None = None,
    ) -> np.ndarray:
        """Joint values (degrees) that put the gripper site at ``pos``, or raise ``PlanningError``.

        Tries from ``init`` (the current pose by default), then from a pose turned towards ``pos`` with
        the wrist pre-rolled, since damped least squares only finds the nearest solution.
        """
        turned = np.array([np.rad2deg(np.arctan2(pos[1], pos[0])), -20.0, 40.0, 70.0, 0.0, self.q[GRIPPER]])
        if roll is not None:
            turned[WRIST_ROLL] = roll
        best_q, best_err = None, np.inf
        for start in (self.q if init is None else init, turned):
            q, err = self.arm.ik(pos, approach, jaw, start, iters=150)
            if err < best_err:
                best_q, best_err = q, err
        if best_err > IK_TOLERANCE:
            raise PlanningError(f"Pose out of reach by {best_err * 1000:.1f} mm")
        return best_q

    # Composite moves shared by the tasks.

    def pick(self, obj: ObjectParams) -> Grasp:
        """From anywhere: move above the object, descend, close the jaws and lift it.

        The approach tilt is sampled, then the nearest reachable one is used: the SO-101's wrist can't
        point straight down everywhere.
        """
        cfg, rng = self.cfg, self.rng
        jitter = np.deg2rad(rng.uniform(-1, 1) * cfg.jaw_yaw_jitter)
        offset = rng.uniform(0.0, cfg.grasp_offset)
        sampled_height = uniform(rng, cfg.approach_height)
        sampled_lift = uniform(rng, cfg.lift_height)
        for tilt in candidate_tilts(uniform(rng, cfg.tilt)):
            grasp = grasp_for(self.arm, obj, tilt, jitter, offset)
            try:
                q_grasp = self.solve(grasp.pos, grasp.approach, grasp.jaw, roll=grasp.roll)
                # Close to the base the arm can't reach high: come in and leave lower there.
                height, q_pre = self._highest(grasp.pos, grasp.approach, grasp.jaw, q_grasp, sampled_height)
                lift_height, _ = self._highest(grasp.pos, DOWN, None, q_grasp, sampled_lift, grasp)
                break
            except PlanningError:
                continue
        else:
            raise PlanningError(f"No reachable grasp of {obj.label}")
        lift = grasp.pos + np.array([0.0, 0.0, lift_height])
        opening = min(
            100.0,
            self.arm.opening_for_width(grasp.width + 2 * grasp.clearance + uniform(rng, cfg.open_margin)),
        )
        self.transit(q_pre, gripper=opening)
        self.dwell()
        self.linear_move(grasp.pos, grasp.approach, grasp.jaw)
        # Let the servos catch up with the last commands before closing.
        self.dwell(rng.uniform(0.2, 0.45))
        # Squeeze: command past the object's width so the servo keeps holding it.
        self.gripper(max(0.0, self.arm.opening_for_width(grasp.width) - rng.uniform(15, 25)))
        self.dwell(rng.uniform(0.1, 0.3))
        self.mark("grasped")
        self.linear_move(lift, grasp.approach, grasp.jaw)
        self.mark("lifted")
        return grasp

    def place(self, pos: np.ndarray, tilt: float) -> None:
        """Holding an object: move above ``pos``, lower the gripper site there, open, and back off."""
        rng = self.rng
        sampled_height = uniform(rng, self.cfg.approach_height)
        for candidate in candidate_tilts(tilt):
            approach = tilted_approach(pos, candidate)
            try:
                q_release = self.solve(pos, approach, None)
                height, q_above = self._highest(pos, approach, None, q_release, sampled_height)
                break
            except PlanningError:
                continue
        else:
            raise PlanningError("No reachable release pose")
        self.transit(q_above)
        self.dwell(rng.uniform(0.0, 0.2))
        self.linear_move(pos, approach, None)
        self.dwell(rng.uniform(0.0, 0.2))
        self.mark("release")
        self.gripper(min(100.0, self.q[GRIPPER] + rng.uniform(25, 45)))
        self.dwell(rng.uniform(0.1, 0.3))
        self.linear_move(pos - approach * height, approach, None)
        self.mark("released")

    def reach(self, pos: np.ndarray, tilt: float) -> None:
        """Move the gripper site to ``pos`` through joint space, at the reachable tilt nearest ``tilt``."""
        for candidate in candidate_tilts(tilt):
            try:
                q = self.solve(pos, tilted_approach(pos, candidate), None)
                break
            except PlanningError:
                continue
        else:
            raise PlanningError("Target out of reach")
        self.transit(q)

    def _highest(
        self,
        pos: np.ndarray,
        approach: np.ndarray,
        jaw: np.ndarray | None,
        q_at: np.ndarray,
        sampled: float,
        grasp: Grasp | None = None,
    ) -> tuple[float, np.ndarray]:
        """The largest reachable distance up to ``sampled`` back along ``approach`` from ``pos``.

        With ``grasp``, the offset is a vertical lift keeping the grasp's orientation.
        """
        for height in (sampled, 0.75 * sampled, 0.5 * sampled, 0.03):
            if grasp is None:
                target, a, j = pos - approach * height, approach, jaw
            else:
                target, a, j = pos + np.array([0.0, 0.0, height]), grasp.approach, grasp.jaw
            try:
                return height, self.solve(target, a, j, init=q_at)
            except PlanningError:
                continue
        raise PlanningError("Nothing reachable above the target")


def candidate_tilts(sampled: float) -> list[float]:
    """The sampled approach tilt (degrees), then others by closeness to it."""
    grid = np.arange(0.0, 65.0, 7.5)
    return [sampled, *sorted(grid, key=lambda t: abs(t - sampled))]


def tilted_approach(pos: np.ndarray, tilt: float) -> np.ndarray:
    """Pointing down, tilted by ``tilt`` degrees away from the robot."""
    radial = np.array([pos[0], pos[1], 0.0]) / max(float(np.hypot(pos[0], pos[1])), 1e-6)
    t = np.deg2rad(tilt)
    return DOWN * np.cos(t) + radial * np.sin(t)


def grasp_for(arm: Arm, obj: ObjectParams, tilt: float, jaw_jitter: float, offset: float) -> Grasp:
    """A grasp of a resting object from above, tilted outwards, closing across its narrow side.

    The jaw direction's sign picks which side the fixed jaw takes; the one needing the smaller wrist
    roll wins, as a person would turn their wrist the short way.
    """
    center = np.array([obj.pos[0], obj.pos[1], obj.half_size[-1]])
    approach = tilted_approach(center, tilt)
    reach = np.arctan2(center[1], center[0])
    if obj.shape == "cylinder":
        # Round: any direction, preferably along the reach.
        yaw, width = reach + jaw_jitter, 2 * obj.half_size[0]
    elif obj.half_size[0] <= obj.half_size[1]:
        yaw, width = obj.yaw, 2 * obj.half_size[0]
    else:
        yaw, width = obj.yaw + np.pi / 2, 2 * obj.half_size[1]
    # At zero roll the jaws close along the reach: roll by the angle between the two, the short way.
    roll = float(np.angle(np.exp(1j * (yaw - reach))))
    if abs(roll) > np.pi / 2:
        yaw += np.pi
        roll = float(np.angle(np.exp(1j * (yaw - reach))))
    horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    jaw = horizontal - approach * np.dot(horizontal, approach)
    jaw /= np.linalg.norm(jaw)
    # Grip around mid-height, or higher on tall objects, the fingertips clear of the table.
    top = 2 * obj.half_size[-1]
    height = max(obj.half_size[-1], top - 0.03, 0.013)
    # A tilted gripper leans its jaws over the object: above the gripper site they shift towards the
    # robot by tan(tilt) per meter, as far as the jaw axis points along the reach.
    radial = np.array([np.cos(reach), np.sin(reach), 0.0])
    clearance = (top - height) * np.tan(np.deg2rad(tilt)) * abs(np.dot(horizontal, radial))
    # The fixed jaw sits next to the object's side; the moving jaw closes onto it.
    side = width / 2 + arm.fixed_jaw_offset + offset + clearance
    pos = np.array([center[0], center[1], height]) - jaw * side
    return Grasp(
        pos=pos,
        approach=approach,
        jaw=jaw,
        width=width,
        clearance=clearance,
        height=height,
        tilt=tilt,
        roll=float(np.rad2deg(roll)),
    )
