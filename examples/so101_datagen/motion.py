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

Paths are near the ideal one rather than on it: transits arc through random control points, approaches
come in off-axis and curve onto it, and each move's speed profile is skewed a little. Consecutive moves
overlap, so the arm flows from one into the next; only pauses and the gripper bring it to rest. Every
move is slowed down until no joint exceeds its peak speed.

A ``Planner`` appends one joint command per control step. Tasks compose its moves (``transit``,
``linear_move``, ``gripper``, ``dwell``, and the composite ``grasp``/``place``) and name the subtask
each command belongs to.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import MotionConfig
from .randomization import uniform
from .scene import ObjectParams
from .so101 import ARM_JOINTS, GRIPPER, WRIST_ROLL, Arm

DOWN = np.array([0.0, 0.0, -1.0])
UP = -DOWN
# A command that misses its Cartesian target by more than this is unreachable.
IK_TOLERANCE = 0.004
# Shoulder lift, elbow and wrist flex of the poses IK restarts from, turned towards the target.
TURNED_POSES = ((-20.0, 40.0, 70.0), (0.0, 30.0, 30.0))
# How much each joint bends in transit arcs: the ones that move the gripper visibly, never the roll.
BEND_WEIGHTS = np.array([1.0, 0.6, 0.6, 0.4, 0.0, 0.0])


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
        # Frames of the last move that the next one may overlap; zero after a pause or the gripper.
        self._blendable = 0

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

    def _segment(self, seconds: float, path: Callable[[np.ndarray], np.ndarray], blend: bool = True) -> None:
        """Emit ``path`` (joint values at progress values in [0, 1]) over ``seconds``, or longer.

        The progress follows a skewed minimum-jerk profile; the segment is stretched until no joint
        command moves faster than its peak speed. With ``blend``, its start overlaps the end of the
        previous move: there, the new move's displacement is added onto the old one's, which rounds the
        corner between them instead of stopping at it.
        """
        skew = uniform(self.rng, self.cfg.time_skew)
        limit = np.asarray(self.cfg.max_joint_speed)
        n = self._steps(seconds)
        for _ in range(4):
            frames = path(min_jerk((np.arange(1, n + 1) / n) ** skew))
            steps = np.diff(np.vstack([self.q, frames]), axis=0)
            ratio = float(np.max(np.abs(steps) * self.fps / limit))
            if ratio <= 1.0:
                break
            # Far too fast is a jump in the path (an IK branch switch), not a speed to slow down.
            if ratio > 4.0:
                raise PlanningError(f"Path jumps ({ratio:.0f}x the speed limit)")
            n = int(np.ceil(n * ratio * 1.05))
        overlap = 0
        if blend and self._blendable:
            overlap = min(int(min(self._blendable, n) * uniform(self.rng, self.cfg.blend)), n - 1)
        # Overlapping moves add their speeds: shorten the overlap until the sum respects the limits too.
        while overlap:
            end = self.q
            tail = np.array(self.frames[-overlap - 1 :])
            tail[1:] += frames[:overlap] - end
            blended = np.vstack([tail, frames[overlap : overlap + 1]])
            if np.max(np.abs(np.diff(blended, axis=0)) * self.fps / limit) <= 1.0:
                for k in range(overlap):
                    self.frames[k - overlap] = tail[k + 1]
                self.q = self.frames[-1].copy()
                break
            overlap //= 2
        for q in frames[overlap:]:
            self._emit(q)
        self._blendable = len(frames) if blend else 0

    # Primitive moves.

    def dwell(self, seconds: float | None = None) -> None:
        seconds = uniform(self.rng, self.cfg.dwell) if seconds is None else seconds
        steps = int(round(seconds * self.fps))
        for _ in range(steps):
            self._emit(self.q)
        if steps:
            self._blendable = 0

    def gripper(self, value: float, overlap: bool = False) -> None:
        """Open or close the gripper, the arm held still. With ``overlap``, the next move may start
        before the jaws finish, as when backing away from a released object."""
        before = len(self.frames)
        start = self.q.copy()

        def path(s: np.ndarray) -> np.ndarray:
            q = np.repeat(start[None], len(s), axis=0)
            q[:, GRIPPER] = start[GRIPPER] + s * (value - start[GRIPPER])
            return q

        self._segment(self.rng.uniform(0.3, 0.7) / self.speed, path, blend=False)
        if overlap:
            self._blendable = len(self.frames) - before

    def transit(self, target: np.ndarray, gripper: float | None = None, min_z: float | None = None) -> None:
        """Joint-space move to ``target`` along a cubic curve: raised through a random via point and bent
        by random control points, the way a hand never takes the straight path. With ``min_z``, the via
        point keeps the gripper site at least that high, as when carrying something over obstacles."""
        start = self.q.copy()
        target = np.array(target, dtype=np.float64)
        target[GRIPPER] = start[GRIPPER] if gripper is None else gripper
        travel = target - start
        distance = np.max(np.abs(travel[:ARM_JOINTS]))
        seconds = max(0.5, distance / (uniform(self.rng, self.cfg.joint_speed) * self._segment_speed()))
        # A cubic Bezier with controls at the thirds plus ``d`` passes 3/4 d off the line halfway.
        middle = (start + target) / 2
        lift = 4 / 3 * (self._via(middle, min_z) - middle)
        # One arc (both controls bent the same way), or sometimes an S.
        bend = (
            self.rng.uniform(-1, 1, len(start))
            * BEND_WEIGHTS
            * (self.cfg.via_spread * np.abs(travel) + self.cfg.via_jitter)
        )
        second = -1.0 if self.rng.random() < self.cfg.s_curve_prob else 1.0
        p1 = start + travel / 3 + lift + bend
        p2 = start + 2 * travel / 3 + lift + second * bend

        def path(s: np.ndarray) -> np.ndarray:
            s = s[:, None]
            q = (1 - s) ** 3 * start + 3 * s * (1 - s) ** 2 * p1 + 3 * s**2 * (1 - s) * p2 + s**3 * target
            q[:, GRIPPER] = start[GRIPPER] + s[:, 0] * travel[GRIPPER]
            return q

        self._segment(seconds, path)

    def _via(self, middle: np.ndarray, min_z: float | None = None) -> np.ndarray:
        """``middle`` with the gripper raised by a random height, or unchanged when that is unreachable.

        With ``min_z``, raised from that height at least, and the wrist may tilt to get there: carrying
        something up and over, a hand doesn't keep pointing down.
        """
        pos, rot = self.arm.tcp_pose(middle)
        raised = pos.copy()
        raised[2] = max(pos[2], -np.inf if min_z is None else min_z) + uniform(self.rng, self.cfg.via_height)
        q, err = self.arm.ik(raised, rot[:, 0], None, middle)
        if err >= 0.01 and min_z is not None:
            q, err = self.arm.ik(raised, None, None, middle, iters=150, roll=float(middle[WRIST_ROLL]))
        return q if err < 0.01 else middle

    def linear_move(
        self,
        pos: np.ndarray,
        approach: np.ndarray,
        jaw: np.ndarray | None,
        via: np.ndarray | None = None,
        keep_roll: bool = False,
    ) -> None:
        """Move the gripper site to ``pos`` in Cartesian space, solving IK along the way.

        The path is straight, or a quadratic curve through ``via``: aimed at a point on the approach axis,
        it arrives along the axis from wherever it started.
        """
        start, _ = self.arm.tcp_pose(self.q)
        distance = float(np.linalg.norm(pos - start))
        seconds = max(0.3, distance / (uniform(self.rng, self.cfg.linear_speed) * self._segment_speed()))
        control = (start + pos) / 2 if via is None else via
        arm, q0 = self.arm, self.q.copy()
        hold = float(q0[WRIST_ROLL]) if keep_roll else None

        def path(s: np.ndarray) -> np.ndarray:
            q, frames = q0, []
            for t in s:
                point = (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t**2 * pos
                q, err = arm.ik(point, approach, jaw, q, roll=hold)
                if err > IK_TOLERANCE:
                    raise PlanningError(f"Linear move out of reach by {err * 1000:.1f} mm")
                frames.append(q)
            return np.array(frames)

        self._segment(seconds, path)

    def _off_axis(self, approach: np.ndarray) -> np.ndarray:
        """A random offset across ``approach``, so the gripper comes in a little off-axis."""
        side = np.cross(approach, self.rng.normal(size=3))
        return side / np.linalg.norm(side) * self.rng.uniform(0, self.cfg.approach_offset)

    def solve(
        self,
        pos: np.ndarray,
        approach: np.ndarray,
        jaw: np.ndarray | None,
        init: np.ndarray | None = None,
        roll: float | None = None,
        keep_roll: bool = False,
    ) -> np.ndarray:
        """Joint values (degrees) that put the gripper site at ``pos``, or raise ``PlanningError``.

        Tries from ``init`` (the current pose by default) and keeps that solution when it is good enough,
        since it is the nearest one: without a jaw direction the wrist roll is free, and a solution from
        elsewhere can turn the wrist half a turn for nothing. Otherwise, damped least squares only finding
        the nearest solution, it tries from poses turned towards ``pos`` with the wrist pre-rolled.
        """
        here = self.q if init is None else init
        pan = np.rad2deg(np.arctan2(pos[1], pos[0]))
        rolled = here[WRIST_ROLL] if roll is None else roll
        # Turned towards ``pos``: the wrist bent for steep-down approaches, then flatter for tilted ones.
        turned = [
            np.array([pan, lift, elbow, flex, rolled, self.q[GRIPPER]]) for lift, elbow, flex in TURNED_POSES
        ]
        best_q, best_err = None, np.inf
        hold = float(self.q[WRIST_ROLL]) if keep_roll else None
        for start in (here, *turned):
            q, err = self.arm.ik(pos, approach, jaw, start, iters=150, roll=hold)
            if err < best_err:
                best_q, best_err = q, err
            if best_err <= IK_TOLERANCE:
                break
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
        steep = rng.random() < cfg.steep_tilt_prob
        for tilt in candidate_tilts(uniform(rng, cfg.steep_tilt if steep else cfg.tilt)):
            grasp = grasp_for(self.arm, obj, tilt, jitter, offset, current_roll=self.q[WRIST_ROLL])
            try:
                q_grasp = self.solve(grasp.pos, grasp.approach, grasp.jaw, roll=grasp.roll)
                # Close to the base the arm can't reach high: come in and leave lower there.
                entry = self._off_axis(grasp.approach)
                height, q_pre = self._highest(
                    grasp.pos, grasp.approach, grasp.jaw, q_grasp, sampled_height, offset=entry
                )
                lift_height, _ = self._highest(
                    grasp.pos, grasp.approach, grasp.jaw, q_grasp, sampled_lift, away=UP
                )
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
        if rng.random() < cfg.pause_prob:
            self.dwell()
        self.linear_move(grasp.pos, grasp.approach, grasp.jaw, via=grasp.pos - grasp.approach * height / 2)
        # Let the servos catch up with the last commands before closing.
        self.dwell(rng.uniform(0.15, 0.3))
        # Squeeze: command past the object's width so the servo keeps holding it.
        self.gripper(max(0.0, self.arm.opening_for_width(grasp.width) - rng.uniform(15, 25)))
        self.dwell(rng.uniform(0.05, 0.2))
        self.mark("grasped")
        self.linear_move(lift, grasp.approach, grasp.jaw)
        self.mark("lifted")
        return grasp

    def place(self, pos: np.ndarray, tilt: float, hover_z: float = 0.0, min_hover_z: float = 0.0) -> None:
        """Holding an object: move above ``pos``, lower the gripper site there, open, and back off.

        The carry keeps the gripper site at ``hover_z`` or higher until it is above ``pos``, high enough
        for the held object to clear the container's rim or whatever stands on the way, or at
        ``min_hover_z`` where that is out of reach.
        """
        rng = self.rng
        sampled_height = uniform(rng, self.cfg.approach_height)
        rise, min_rise = max(0.0, hover_z - pos[2]), max(0.0, min_hover_z - pos[2])
        found = None
        for candidate in candidate_tilts(tilt):
            approach = tilted_approach(pos, candidate)
            # Carried: the jaw direction may follow, but the wrist keeps its roll.
            try:
                q_release = self.solve(pos, approach, None, keep_roll=True)
            except PlanningError:
                continue
            entry = self._off_axis(approach)
            # Down along the approach, or, when the point that high on its axis is out of reach (close
            # to the base), straight down from above the release, as one lowers something into a box.
            along = max(-approach[2], 0.3)
            for away, target, floor in ((-approach, rise / along, min_rise / along), (UP, rise, min_rise)):
                try:
                    height, q_above = self._highest(
                        pos,
                        approach,
                        None,
                        q_release,
                        max(sampled_height, target),
                        away=away,
                        offset=entry,
                        keep_roll=True,
                        floor=floor,
                    )
                    found = approach, away
                    break
                except PlanningError:
                    continue
            if found:
                break
        if found is None:
            raise PlanningError("No reachable release pose")
        approach, away = found
        self.transit(q_above, min_z=hover_z)
        if rng.random() < self.cfg.pause_prob:
            self.dwell()
        self.linear_move(pos, approach, None, via=pos + away * height / 2, keep_roll=True)
        self.dwell(rng.uniform(0.0, 0.2))
        self.mark("release")
        self.gripper(min(100.0, self.q[GRIPPER] + rng.uniform(25, 45)), overlap=True)
        self.linear_move(pos + away * height, approach, None, keep_roll=True)
        self.mark("released")

    def reach(self, pos: np.ndarray, tilt: float) -> None:
        """Move the gripper site to ``pos`` through joint space, at the reachable tilt nearest ``tilt``."""
        for candidate in candidate_tilts(tilt):
            try:
                q = self.solve(pos, tilted_approach(pos, candidate), None, keep_roll=True)
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
        away: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        keep_roll: bool = False,
        floor: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """The largest reachable distance up to ``sampled``, and not below ``floor``, from ``pos`` towards
        ``away`` (back along ``approach`` by default), keeping the orientation given by ``approach`` and
        ``jaw``. ``offset`` moves the point off that line, scaled down with the distance so low
        approaches stay close to it.
        """
        away = -approach if away is None else away
        heights = [h for h in (sampled, 0.75 * sampled, 0.5 * sampled, 0.03) if h > floor]
        heights += [floor] if floor > 0 else []
        for height in heights:
            side = 0.0 if offset is None else offset * height / sampled
            try:
                return height, self.solve(
                    pos + away * height + side, approach, jaw, init=q_at, keep_roll=keep_roll
                )
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


def grasp_for(
    arm: Arm,
    obj: ObjectParams,
    tilt: float,
    jaw_jitter: float,
    offset: float,
    current_roll: float = 0.0,
) -> Grasp:
    """A grasp of a resting object from above, tilted outwards, closing across its narrow side.

    The jaw direction's sign picks which side the fixed jaw takes; the one nearest the wrist's
    ``current_roll`` (degrees) wins, as a person turns their wrist the short way from where it is.
    """
    center = np.array([obj.pos[0], obj.pos[1], obj.half_size[-1]])
    approach = tilted_approach(center, tilt)
    reach = np.arctan2(center[1], center[0])
    if obj.shape == "cylinder":
        # Round: any direction, preferably as the wrist already is.
        yaw, width = reach + np.deg2rad(np.clip(current_roll, -90, 90)) + jaw_jitter, 2 * obj.half_size[0]
    elif obj.half_size[0] <= obj.half_size[1]:
        yaw, width = obj.yaw, 2 * obj.half_size[0]
    else:
        yaw, width = obj.yaw + np.pi / 2, 2 * obj.half_size[1]
    # At zero roll the jaws close along the reach: roll by the angle between the two. Both jaw sides
    # hold the object, so take the one whose roll is nearest the current one (within the joint's range).
    sides = [yaw, yaw + np.pi]
    rolls = [float(np.angle(np.exp(1j * (side - reach)))) for side in sides]
    best = int(np.argmin([abs(np.rad2deg(r) - current_roll) for r in rolls]))
    yaw, roll = sides[best], rolls[best]
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
