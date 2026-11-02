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

"""Scripted joint paths with rounded corners and continuous timing across connected moves.

Transits arc through control points and approaches curve onto their final axis. Minimum-jerk
primitives build the path; overlapping their displacements rounds its corners. ``plan`` then retimes
connected arm moves with one broad acceleration and deceleration profile, so primitive boundaries
no longer introduce speed dips. Gripper actions and contact pauses retain their own timing. Every
joint stays within its peak command speed.

A ``Planner`` appends one joint command per control step. Tasks compose its moves (``transit``,
``linear_move``, ``gripper``, ``dwell``, and the composite ``pick``/``place``) and name the subtask
each command belongs to. With a live rollout, grasp and release checkpoints freeze and execute
pending commands before planning a recovery from the observed object pose.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .config import MotionConfig
from .errors import EpisodeError
from .randomization import uniform
from .scene import ObjectParams
from .so101 import ARM_JOINTS, GRIPPER, WRIST_ROLL, Arm

if TYPE_CHECKING:
    from .episode import Rollout

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


def rounded_path(path: np.ndarray, limit: np.ndarray, spacing: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Cubic B-spline approximation with repeated end controls to preserve contact poses.

    Controls are spaced by joint travel, not the old timing, so slow IK samples do not bunch them up.
    The returned source coordinates let phase markers follow the curve.
    """
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) / limit, axis=1))]
    if distance[-1] < 1e-8:
        return path, np.arange(len(path), dtype=float)
    count = max(8, int(np.ceil(distance[-1] / spacing)) + 1)
    source = np.interp(np.linspace(0, distance[-1], count), distance, np.arange(len(path)))
    controls = np.column_stack([np.interp(source, np.arange(len(path)), q) for q in path.T])
    controls = np.vstack([controls[0], controls[0], controls, controls[-1], controls[-1]])
    source = np.r_[source[0], source[0], source, source[-1], source[-1]]
    t = np.linspace(0, len(controls) - 3, max(1000, count * 12))
    k = np.minimum(t.astype(int), len(controls) - 4)
    u = t - k
    basis = np.array([(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3]).T / 6
    indices = k[:, None] + np.arange(4)
    return np.sum(basis[:, :, None] * controls[indices], axis=1), np.sum(basis * source[indices], axis=1)


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
    style: str = "object_aligned"
    point: np.ndarray | None = None
    heading_offset: float = 0.0
    offset: float = 0.0
    opening: float = 0.0


class DroppedError(Exception):
    """The object is no longer held before release; recover it within the shared attempt budget."""


@dataclass
class Objective:
    id: str
    label: str
    object_name: str | None = None
    transfer_id: str | None = None


@dataclass
class TransferResult:
    transfer_id: str
    object_name: str
    frame: int


@dataclass
class GraspAttempt:
    grasp: Grasp
    start: int
    looked: int
    approach_end: int
    closing: int
    lifted: int
    miss_mode: str | None = None
    success: bool | None = None
    point_local: np.ndarray | None = None
    object_name: str = ""
    transfer_id: str = "task"
    objective_id: str = ""


@dataclass
class Plan:
    commands: np.ndarray
    subtask_of_frame: np.ndarray
    subtasks: list[str]
    # Named frame indices, such as "lifted", for checks during execution.
    marks: dict[str, int] = field(default_factory=dict)
    grasp: Grasp | None = None
    approach_clearance: float = 0.0
    attempts: list[GraspAttempt] = field(default_factory=list)
    memories: list[str | None] = field(default_factory=list)
    retry_frames: list[int] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    objective_ids: list[str] = field(default_factory=list)
    completed_transfers: list[TransferResult] = field(default_factory=list)

    def attempt_at(self, frame: int) -> GraspAttempt | None:
        """The active objective selects the object, including before its first approach marker."""
        target = None
        transfer = None
        if self.objective_ids:
            identity = self.objective_ids[self.subtask_of_frame[frame]]
            objective = next(o for o in self.objectives if o.id == identity)
            target, transfer = objective.object_name, objective.transfer_id
        attempts = [
            a
            for a in self.attempts
            if (target is None or a.object_name == target) and (transfer is None or a.transfer_id == transfer)
        ]
        return next((a for a in reversed(attempts) if a.start <= frame), attempts[0] if attempts else None)


class Planner:
    def __init__(
        self,
        arm: Arm,
        start: np.ndarray,
        fps: int,
        cfg: MotionConfig,
        rng: np.random.Generator,
        workspace: tuple[float, float] = (0.17, 0.30),
        object_geoms: dict[str, list[int]] | None = None,
        rollout: "Rollout | None" = None,
    ):
        self.arm = arm
        self.rollout = rollout
        self.attempts: list[GraspAttempt] = []
        self.memories: list[str | None] = []
        self.retry_frames: list[int] = []
        self.objectives: list[Objective] = []
        self.objective_ids: list[str] = []
        self.completed_transfers: list[TransferResult] = []
        self.active_object: str | None = None
        self.transfer_id = "task"
        self.pick_objective_id: str | None = None
        self._retry_grasp: Grasp | None = None
        self._straight_entry = False
        self._miss_mode: str | None = None
        self._miss_shift = np.zeros(3)
        self._miss_distance = 0.0
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
        self.object_geoms = object_geoms or {}
        self.grasp: Grasp | None = None
        self.held_geoms: set[int] = set()
        self.check_initial_transit = False
        # Frames of the last move that the next one may overlap; zero after a pause or the gripper.
        self._blendable = 0
        self._last_overlap = 0
        self._flows: list[tuple[int, int]] = []

    def subtask(self, label: str, memory: str | None = None, *, objective_id: str | None = None) -> None:
        """Commands from now on belong to a new subtask."""
        if objective_id is None:
            objective = next((o for o in self.objectives if o.label == label), None)
            if objective is None:
                objective = Objective(f"task:{len(self.objectives)}", label, self.active_object)
                self.objectives.append(objective)
            objective_id = objective.id
        elif not any(o.id == objective_id and o.label == label for o in self.objectives):
            raise ValueError(f"Undeclared objective {objective_id!r}: {label}")
        self.subtasks.append(label)
        self.memories.append(memory)
        self.objective_ids.append(objective_id)

    def begin_transfer(self, transfer_id: str, object_name: str, pick_objective_id: str) -> None:
        """Execute any pending retreat before switching objects; retries stay in this transfer."""
        if self.frames:
            self.flush()
        self.transfer_id = transfer_id
        self.active_object = object_name
        self.pick_objective_id = pick_objective_id
        self._retry_grasp = None
        self.grasp = None
        self.held_geoms.clear()
        if self.rollout is not None:
            self.rollout.attempts_used = 0

    def mark(self, name: str) -> None:
        self.marks[name] = len(self.frames)

    def plan(self) -> Plan:
        # Keep the collision-checked geometric path, but remove the separate stop/start timing of
        # each primitive. One broad speed pulse carries the arm through connected rounded corners.
        commands = np.stack(self.frames)
        ids = np.asarray(self.subtask_ids)
        chunks, labels, source_frames = [], [], []
        cursor = 0
        limit = np.asarray(self.cfg.max_joint_speed)
        for start, end in self._flows:
            chunks.append(commands[cursor:start])
            labels.append(ids[cursor:start])
            source_frames.extend(range(cursor, start))
            original = commands[start:end]
            path, source = self._rounded_flow(original, start, limit)
            # Time the hand in Cartesian distance, with a small rotation component so wrist-only
            # moves remain well-defined. A joint-space speed metric makes Cartesian speed jump as
            # the active joint or the arm configuration changes.
            tcp = np.array([self.arm.tcp_pose(q)[0] for q in path])
            travel = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / 0.1
            turns = np.linalg.norm(np.diff(path, axis=0) / limit, axis=1)
            distance = np.r_[0.0, np.cumsum(np.hypot(travel, 0.1 * turns))]
            if distance[-1] < 1e-8 or len(original) < 4:
                chunks.append(original)
                labels.append(ids[start:end])
                source_frames.extend(range(start, end))
                cursor = end
                continue
            # One broad minimum-jerk pulse over the whole flow: speed varies naturally without
            # dropping at each primitive boundary. Preserve the original path's timing asymmetry,
            # bounded by the configured skew; plan() remains deterministic when called again.
            midpoint = np.interp(distance[-1] / 2, distance, source / max(1, len(original) - 1))
            skew = float(np.clip(np.log(0.5) / np.log(midpoint), *self.cfg.time_skew))
            # Uniformly stretch the whole flow until the sampled commands respect every joint limit.
            n = len(original)
            for _ in range(4):
                progress = min_jerk(np.linspace(0.0, 1.0, n) ** skew)
                at = np.interp(progress * distance[-1], distance, np.arange(len(path)))
                frames = np.column_stack([np.interp(at, np.arange(len(path)), q) for q in path.T])
                ratio = float(np.max(np.abs(np.diff(frames, axis=0)) * self.fps / limit))
                if ratio <= 1.0:
                    break
                n = int(np.ceil(n * ratio * 1.02))
            if ratio > 1.0:
                raise PlanningError("Retimed flow exceeds joint speed limits")
            chunks.append(frames)
            labels.append(ids[start + np.rint(np.interp(at, np.arange(len(path)), source)).astype(int)])
            original_at = np.interp(at, np.arange(len(path)), source)
            source_frames.extend(start + original_at)
            cursor = end
        chunks.append(commands[cursor:])
        labels.append(ids[cursor:])
        source_frames.extend(range(cursor, len(commands)))
        # Markers and subtask labels follow the path through retiming, including the visibility span
        # and the physical grasp/lift/release checks.
        marks = {name: int(np.searchsorted(source_frames, frame)) for name, frame in self.marks.items()}
        attempts = [
            replace(
                a,
                **{
                    name: int(np.searchsorted(source_frames, getattr(a, name)))
                    for name in ("start", "looked", "approach_end", "closing", "lifted")
                },
            )
            for a in self.attempts
        ]
        return Plan(
            commands=np.concatenate(chunks),
            subtask_of_frame=np.concatenate(labels),
            subtasks=list(self.subtasks),
            marks=marks,
            grasp=self.grasp,
            approach_clearance=self.cfg.approach_clearance,
            attempts=attempts,
            memories=list(self.memories),
            retry_frames=[int(np.searchsorted(source_frames, f)) for f in self.retry_frames],
            objectives=list(self.objectives),
            objective_ids=list(self.objective_ids),
            completed_transfers=list(self.completed_transfers),
        )

    def flush(self) -> None:
        """Freeze pending geometry/timing, execute it once, and forbid overlap into that prefix."""
        if self.rollout is None:
            return
        plan = self.plan()
        self.frames = [q.copy() for q in plan.commands]
        self.subtask_ids = plan.subtask_of_frame.tolist()
        self.marks = dict(plan.marks)
        self.attempts = list(plan.attempts)
        self.retry_frames = list(plan.retry_frames)
        self._flows.clear()
        self._blendable = 0
        self.rollout.run_until(len(self.frames))
        self.rollout.sync_scene()

    def _snapshot(self) -> tuple[dict, dict]:
        names = (
            "q",
            "frames",
            "subtask_ids",
            "subtasks",
            "memories",
            "marks",
            "_flows",
            "_blendable",
            "_last_overlap",
            "grasp",
            "attempts",
            "retry_frames",
            "held_geoms",
            "_straight_entry",
            "_retry_grasp",
            "objectives",
            "objective_ids",
            "completed_transfers",
            "active_object",
            "transfer_id",
            "pick_objective_id",
        )
        return {name: deepcopy(getattr(self, name)) for name in names}, deepcopy(self.rng.bit_generator.state)

    def _restore(self, snapshot: tuple[dict, dict]) -> None:
        values, state = snapshot
        for name, value in values.items():
            setattr(self, name, deepcopy(value))
        self.rng.bit_generator.state = deepcopy(state)

    def recover(self, obj: ObjectParams, reason: str) -> None:
        """Open and back off before observing the object again; all executed frames stay frozen."""
        memory = (
            f"My grasp of the {obj.label} missed; trying again."
            if reason == "grasp missed"
            else f"I dropped the {obj.label}; trying again."
        )
        self.subtask(f"pick up the {obj.label}", memory=memory, objective_id=self.pick_objective_id)
        self.retry_frames.append(len(self.frames))
        self.gripper(100.0)
        self.dwell(0.15)
        self.flush()
        pos, rot = self.arm.tcp_pose(self.q)
        checkpoint = self._snapshot()
        before = len(self.frames)
        # A failed lift has often already backed the open hand clear. Keep that useful pose
        # instead of returning to REST and repeating the entire approach.
        geoms = set(self.object_geoms.get(obj.name, []))
        target_body = self.arm.model.body(obj.name).id
        if min(self.arm.jaw_clearances(self.q, geoms)) >= 0.02 and self.arm.wrist_camera_ok(
            self.q, target_body, geoms
        ):
            return
        # Retreat only a few centimetres, retaining the wrist roll and preferably its tilt.
        # If no local route is clear, discard the episode rather than teach a neutral-pose reset.
        away = -rot[:, 0]
        diagonal = away + UP
        diagonal /= np.linalg.norm(diagonal)
        last_error = None
        for distance in (0.02, 0.035):
            for direction in (away, UP, diagonal):
                for keep_tilt in (True, False):
                    self._restore(checkpoint)
                    target = pos + direction * distance
                    try:
                        if keep_tilt:
                            self.linear_move(target, rot[:, 0], None, keep_roll=True)
                        else:
                            q, error = self.arm.ik(
                                target, None, None, self.q, iters=150, roll=float(self.q[WRIST_ROLL])
                            )
                            if (
                                error > IK_TOLERANCE
                                or np.max(np.abs(q[:ARM_JOINTS] - self.q[:ARM_JOINTS])) > 20
                            ):
                                raise PlanningError("Local recovery pose is out of reach")
                            self.transit(q, simple=True)
                        if any(
                            self.arm.collides(q) or not self.arm.wrist_camera_ok(q)
                            for q in self.frames[before:]
                        ):
                            raise PlanningError("Local recovery back-off is obstructed")
                    except PlanningError as error:
                        last_error = error
                        continue
                    self.flush()
                    return
        self._restore(checkpoint)
        raise last_error

    def _rounded_flow(
        self, original: np.ndarray, start: int, limit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Round corners within 2 mm / 1 degree, checking sampled collision and camera constraints.

        Tight spaces use a smaller rounding radius. End poses stay exact. This is a sampled path
        check; the recorded physics trajectory is also checked for camera visibility in simulate().
        """
        target = int(self.arm.model.geom_bodyid[next(iter(self.held_geoms))]) if self.held_geoms else None
        looked = self.marks.get("looked", -1)
        approach_end = self.marks.get("grasp_approach_end", -1)
        for spacing in (0.04, 0.02, 0.01, 0.005):
            path, source = rounded_path(original, limit, spacing)
            reference = np.column_stack([np.interp(source, np.arange(len(original)), q) for q in original.T])
            # Retain the original finger schedule through overlaps with closing/opening.
            path[:, GRIPPER] = reference[:, GRIPPER]
            if np.max(np.abs(path - reference)) > 1.0:
                continue
            valid = True
            for i in np.unique(np.r_[np.arange(0, len(path), 8), len(path) - 1]):
                q, old = path[i], reference[i]
                if np.linalg.norm(self.arm.tcp_pose(q)[0] - self.arm.tcp_pose(old)[0]) > 0.002:
                    valid = False
                    break
                watching = looked <= start + source[i] < approach_end
                if not self.arm.wrist_camera_ok(q, target if watching else None, self.held_geoms):
                    valid = False
                    break
                if (
                    watching
                    and min(self.arm.jaw_clearances(q, self.held_geoms)) < self.cfg.approach_clearance + 0.002
                ):
                    valid = False
                    break
                ignore = set() if watching else self.held_geoms
                if self.arm.collides(q, ignore) and not self.arm.collides(old, ignore):
                    valid = False
                    break
            if valid:
                return path, source
        # Retain a tight path that cannot be rounded within the visibility/clearance constraints.
        return original.copy(), np.arange(len(original), dtype=float)

    def _emit(self, q: np.ndarray) -> None:
        self.q = np.array(q, dtype=np.float64)
        self.frames.append(self.q.copy())
        self.subtask_ids.append(max(len(self.subtasks) - 1, 0))

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def _segment_speed(self) -> float:
        return self.speed * uniform(self.rng, self.cfg.segment_speed)

    def _segment(
        self,
        seconds: float,
        path: Callable[[np.ndarray], np.ndarray],
        blend: bool = True,
        blend_check: Callable[[np.ndarray], bool] | None = None,
        max_overlap: int | None = None,
        continuous: bool = True,
    ) -> None:
        """Emit ``path`` (joint values at progress values in [0, 1]) over ``seconds``, or longer.

        The progress follows a skewed minimum-jerk profile; the segment is stretched until no joint
        command moves faster than its peak speed. With ``blend``, its start overlaps the end of the
        previous move: there, the new move's displacement is added onto the old one's, which rounds the
        corner between them instead of stopping at it.
        """
        before = len(self.frames)
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
            if max_overlap is not None:
                overlap = min(overlap, max_overlap)
        # Overlapping moves add their speeds: shorten the overlap until the sum respects the limits too.
        while overlap:
            end = self.q
            tail = np.array(self.frames[-overlap - 1 :])
            tail[1:] += frames[:overlap] - end
            blended = np.vstack([tail, frames[overlap : overlap + 1]])
            if np.max(np.abs(np.diff(blended, axis=0)) * self.fps / limit) <= 1.0 and (
                blend_check is None or all(blend_check(q) for q in tail[1:])
            ):
                for k in range(overlap):
                    self.frames[k - overlap] = tail[k + 1]
                self.q = self.frames[-1].copy()
                break
            overlap //= 2
        for q in frames[overlap:]:
            self._emit(q)
        if continuous:
            start = before - overlap
            # Gripper overlap is contact-sensitive: preserve its original timing.
            if self._flows and self._flows[-1][1] == before and overlap:
                start = self._flows.pop()[0]
            else:
                start = before
            self._flows.append((start, len(self.frames)))
        self._last_overlap = overlap
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
        if abs(value - self.q[GRIPPER]) < 1e-6:
            return
        before = len(self.frames)
        start = self.q.copy()

        def path(s: np.ndarray) -> np.ndarray:
            q = np.repeat(start[None], len(s), axis=0)
            q[:, GRIPPER] = start[GRIPPER] + s * (value - start[GRIPPER])
            return q

        self._segment(self.rng.uniform(0.3, 0.7) / self.speed, path, blend=False, continuous=False)
        if overlap:
            self._blendable = len(self.frames) - before

    def transit(
        self,
        target: np.ndarray,
        gripper: float | None = None,
        min_z: float | None = None,
        early_roll: bool = False,
        simple: bool = False,
        look_arc: bool = False,
    ) -> None:
        """Joint-space move to ``target`` along a cubic curve: raised through a random via point and bent
        by random control points, the way a hand never takes the straight path. With ``min_z``, the via
        point keeps the gripper site at least that high, as when carrying something over obstacles.
        With ``early_roll``, the wrist turns earlier, without a separate wrist-only movement. ``simple``
        takes a direct joint-space path to a look pose. ``look_arc`` uses a smooth raised arc without
        random sideways bends when the direct path is obstructed."""
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
            if simple:
                q = start + s * travel
            elif look_arc:
                q = (
                    (1 - s) ** 3 * start
                    + 3 * s * (1 - s) ** 2 * (start + travel / 3 + lift)
                    + 3 * s**2 * (1 - s) * (start + 2 * travel / 3 + lift)
                    + s**3 * target
                )
            else:
                q = (1 - s) ** 3 * start + 3 * s * (1 - s) ** 2 * p1 + 3 * s**2 * (1 - s) * p2 + s**3 * target
            q[:, GRIPPER] = start[GRIPPER] + s[:, 0] * travel[GRIPPER]
            if early_roll:
                q[:, WRIST_ROLL] = start[WRIST_ROLL] + (1 - (1 - s[:, 0]) ** 2.5) * travel[WRIST_ROLL]
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
        blend_check: Callable[[np.ndarray], bool] | None = None,
        max_overlap: int | None = None,
        precision: float = 1e-5,
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
                q, err = arm.ik(point, approach, jaw, q, tol=precision, roll=hold)
                if err > IK_TOLERANCE:
                    raise PlanningError(f"Linear move out of reach by {err * 1000:.1f} mm")
                frames.append(q)
            return np.array(frames)

        self._segment(seconds, path, blend_check=blend_check, max_overlap=max_overlap)

    def _off_axis(self, approach: np.ndarray) -> np.ndarray:
        """A random offset across ``approach``, so the gripper comes in a little off-axis."""
        side = np.cross(approach, self.rng.normal(size=3))
        return (
            side
            / np.linalg.norm(side)
            * self.rng.uniform(0, 0.0 if self._straight_entry else self.cfg.approach_offset)
        )

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

    def pick(self, obj: ObjectParams, preflight: Callable[[Grasp], None] | None = None) -> Grasp:
        self.active_object = obj.name
        if self.pick_objective_id is None and self.objective_ids:
            self.pick_objective_id = self.objective_ids[-1]
        if self.rollout is None:
            return self._pick_once(obj)
        recovery = self.rollout.episode.config.recovery

        def plan_attempt(current: ObjectParams) -> Grasp:
            grasp = self._pick_once(current)
            if preflight is not None and self._miss_mode is None:
                after_pick = self._snapshot()
                rollout = self.rollout
                self.rollout = None
                try:
                    preflight(grasp)
                finally:
                    self.rollout = rollout
                    self._restore(after_pick)
            return grasp

        while self.rollout.attempts_used < recovery.max_attempts:
            current = self.rollout.object_now(obj)
            if current is None:
                raise EpisodeError("toppled_or_unreachable")
            self.rollout.sync_scene()
            previous = next(
                (
                    a
                    for a in reversed(self.attempts)
                    if a.transfer_id == self.transfer_id and a.object_name == obj.name
                ),
                None,
            )
            self._retry_grasp = (
                previous.grasp
                if previous is not None and previous.miss_mode in ("high", "short", "wide", "loose")
                else None
            )
            # Spoil only the first attempt. Recovery should correct a miss, rather than introduce
            # another deliberate error; both remaining attempts stay available for natural failures.
            inject = (
                self.rollout.total_attempts_used == 0
                and recovery.max_attempts > 1
                and self.rollout.rng.random() < recovery.miss_prob
            )
            self._miss_mode = None
            if inject:
                modes = list(recovery.miss_modes)
                weights = np.array(list(recovery.miss_modes.values()), dtype=float)
                self._miss_mode = str(self.rollout.rng.choice(modes, p=weights / weights.sum()))
            self._miss_distance = 0.0
            if self._miss_mode in ("high", "short", "wide"):
                bounds = {"high": (0.02, 0.04), "short": (0.015, 0.03), "wide": (0.015, 0.025)}
                self._miss_distance = float(self.rollout.rng.uniform(*bounds[self._miss_mode]))
            checkpoint = self._snapshot()
            original_cfg = self.cfg

            def try_styles(
                current: ObjectParams = current,
                previous: GraspAttempt | None = previous,
                checkpoint: tuple[dict, dict] = checkpoint,
                original_cfg: MotionConfig = original_cfg,
            ) -> Grasp:
                preferred = None
                if previous is not None:
                    preferred = previous.grasp.style
                last_error = None
                for proposal in range(3):
                    for straight in (False, True):
                        first_style = preferred
                        for alternate in (False, True):
                            self._restore(checkpoint)
                            self.rng.random(8 * proposal)
                            if proposal:
                                self._retry_grasp = None
                            self.grasp = None
                            self.cfg = deepcopy(original_cfg)
                            if straight:
                                self.cfg.approach_offset = 0.0
                                self._straight_entry = True
                            style = first_style
                            if alternate:
                                if first_style is None:
                                    break
                                style = "tangential" if first_style == "object_aligned" else "object_aligned"
                            if style is not None:
                                self.cfg.grasp_styles = {style: 1.0}
                            try:
                                return plan_attempt(current)
                            except PlanningError as candidate_error:
                                last_error = candidate_error
                                if first_style is None and self.grasp is not None:
                                    first_style = self.grasp.style
                            finally:
                                self.cfg = original_cfg
                raise last_error

            try:
                grasp = try_styles()
            except PlanningError:
                # In a mixed miss recipe, use a loose attempt when a spatial miss cannot fit safely.
                # Forced single-mode validation still rejects infeasible scenes explicitly.
                if self._miss_mode not in (None, "loose") and recovery.miss_modes.get("loose", 0) > 0:
                    self._restore(checkpoint)
                    self._miss_mode = "loose"
                    self._miss_distance = 0.0
                    grasp = try_styles()
                else:
                    raise
            self.rollout.attempts_used += 1
            self.rollout.total_attempts_used += 1
            body = self.rollout.episode.info.object_bodies[obj.name]
            point = grasp.point if grasp.point is not None else grasp.pos
            local = self.rollout.data.xmat[body].reshape(3, 3).T @ (point - self.rollout.data.xpos[body])
            self.attempts.append(
                GraspAttempt(
                    grasp,
                    self.marks["approach_start"],
                    self.marks["looked"],
                    self.marks["grasp_approach_end"],
                    self.marks["closing"],
                    self.marks["lifted"],
                    self._miss_mode,
                    point_local=local,
                    object_name=obj.name,
                    transfer_id=self.transfer_id,
                    objective_id=self.pick_objective_id or "",
                )
            )
            self.flush()
            success = self.rollout.lifted(obj.name)
            body_height = self.rollout.data.xpos[body, 2] - obj.half_size[-1] - obj.support_height
            if not success and self.rollout.held(obj.name) and body_height > 0.002:
                # A shallow reachable down-facing lift can leave a held object just below the
                # height threshold. Let the wrist tilt to raise it before declaring a failed pick.
                pos, _ = self.arm.tcp_pose(self.q)
                raised = pos + UP * 0.035
                q_raised, error = self.arm.ik(
                    raised, None, None, self.q, iters=150, roll=float(self.q[WRIST_ROLL])
                )
                if error < IK_TOLERANCE:
                    checkpoint = self._snapshot()
                    before = len(self.frames)
                    try:
                        self.transit(q_raised, min_z=raised[2])
                        if any(
                            not self.arm.wrist_camera_ok(q) or self.arm.collides(q, self.held_geoms)
                            for q in self.frames[before:]
                        ):
                            raise PlanningError("Held-object raise is obstructed")
                    except PlanningError:
                        self._restore(checkpoint)
                    else:
                        self.mark("lifted")
                        self.attempts[-1].lifted = len(self.frames)
                        self.flush()
                        success = self.rollout.lifted(obj.name)
            self.attempts[-1].success = success
            self._miss_mode = None
            if success:
                return grasp
            if self.rollout.attempts_used < recovery.max_attempts:
                self.recover(obj, "grasp missed")
        raise EpisodeError("attempts_exhausted")

    def _pick_once(self, obj: ObjectParams) -> Grasp:
        """From anywhere: move above the object, descend, close the jaws and lift it.

        The approach tilt is sampled, then the nearest reachable one is used: the SO-101's wrist can't
        point straight down everywhere.
        """
        cfg, rng = self.cfg, self.rng
        jitter = np.deg2rad(rng.uniform(-1, 1) * cfg.jaw_yaw_jitter)
        # Keep a tracking/pad allowance above the required physical gap. The measured geometry
        # below decides whether both sides really have room, including rotated object corners.
        offset = cfg.approach_clearance + 0.004 + rng.uniform(0.0, cfg.grasp_offset)
        if self._retry_grasp is not None:
            offset = self._retry_grasp.offset
            jitter = 0.0
        sampled_height = uniform(rng, cfg.approach_height)
        sampled_lift = uniform(rng, cfg.lift_height)
        empty_miss = self._miss_mode in ("high", "short")
        weights = np.asarray(
            [cfg.grasp_styles.get("object_aligned", 0), cfg.grasp_styles.get("tangential", 0)], dtype=float
        )
        if np.any(weights < 0) or weights.sum() <= 0:
            raise PlanningError("Invalid grasp style weights")
        # Prefer the style requiring less wrist rotation, while keeping both styles possible.
        aligned = grasp_for(self.arm, obj, 0.0, jitter, offset, current_roll=self.q[WRIST_ROLL])
        turns = np.array(
            [
                abs(aligned.roll - self.q[WRIST_ROLL]),
                min(abs(90 - self.q[WRIST_ROLL]), abs(-90 - self.q[WRIST_ROLL])),
            ]
        )
        weights *= np.exp(-turns / 60.0)
        first = rng.choice(["object_aligned", "tangential"], p=weights / weights.sum())
        styles = [first, "tangential" if first == "object_aligned" else "object_aligned"]
        if self._retry_grasp is not None and cfg.grasp_styles.get(self._retry_grasp.style, 0) > 0:
            styles = sorted(styles, key=lambda style: style != self._retry_grasp.style)
        self.held_geoms = set(self.object_geoms.get(obj.name, []))
        target_body = self.arm.model.body(obj.name).id
        start_pos, start_rot = self.arm.tcp_pose(self.q)
        object_center = np.array([*obj.pos[:2], obj.support_height + obj.half_size[-1]])
        close_start = np.linalg.norm(start_pos - object_center) < 0.09 and self.arm.wrist_camera_ok(
            self.q, target_body, self.held_geoms
        )
        if close_start:
            jitter = 0.0
        start_tilt = float(np.rad2deg(np.arccos(np.clip(-start_rot[2, 0], -1.0, 1.0))))
        found = False
        for style in styles:
            sampled = uniform(rng, cfg.inclined_tilt if rng.random() < cfg.inclined_tilt_prob else cfg.tilt)
            tilts = candidate_tilts(sampled)
            if self._retry_grasp is not None:
                tilts.insert(0, self._retry_grasp.tilt)
            if close_start and start_tilt <= 90:
                tilts.insert(0, start_tilt)
            for tilt in tilts:
                rolls = (
                    sorted((-90.0, 90.0), key=lambda r: abs(r - self.q[WRIST_ROLL]))
                    if style == "tangential"
                    else [self.q[WRIST_ROLL], self.q[WRIST_ROLL] + 180, self.q[WRIST_ROLL] - 180]
                )
                if self._retry_grasp is not None and style == "object_aligned":
                    roll = self._retry_grasp.roll
                    rolls = [roll, roll + 180, roll - 180]
                candidates = [
                    (roll, heading)
                    for roll in rolls
                    for heading in ((-3.0, 3.0) if style == "tangential" else (0.0,))
                ]
                for roll_hint, heading in candidates:
                    try:
                        if style == "tangential":
                            grasp = tangential_grasp_for(
                                self.arm,
                                obj,
                                tilt,
                                rng,
                                offset,
                                cfg.tangential_min_height,
                                roll_hint,
                                heading,
                                cfg.tangential_grip_depth,
                            )
                        else:
                            grasp = grasp_for(
                                self.arm,
                                obj,
                                tilt,
                                jitter,
                                offset,
                                current_roll=roll_hint,
                                keep_heading=close_start or self._retry_grasp is not None,
                            )
                        self._miss_shift = np.zeros(3)
                        if self._miss_mode == "wide":
                            self._miss_shift = -grasp.jaw * self._miss_distance
                            grasp.pos = grasp.pos + self._miss_shift
                            grasp.height += self._miss_shift[2]
                        q_grasp = self.solve(grasp.pos, grasp.approach, grasp.jaw, roll=grasp.roll)
                        opening = min(
                            100.0,
                            self.arm.opening_for_width(
                                grasp.width
                                + 2 * (grasp.clearance + offset + abs(np.dot(self._miss_shift, grasp.jaw)))
                                + uniform(rng, cfg.open_margin)
                            ),
                        )
                        if self._retry_grasp is not None and self._retry_grasp.opening > 0:
                            opening = self._retry_grasp.opening
                        grasp.opening = opening
                        q_grasp[GRIPPER] = opening
                        if style == "tangential":
                            height, q_pre = self._highest(
                                grasp.pos,
                                grasp.approach,
                                grasp.jaw,
                                q_grasp,
                                rng.uniform(0.04, 0.07),
                                floor=0.04,
                            )
                            q_pre[GRIPPER] = opening
                            if self.arm.collides(q_grasp, self.held_geoms) or self.arm.collides(
                                q_pre, self.held_geoms
                            ):
                                continue
                        else:
                            entry = self._off_axis(grasp.approach)
                            if self._retry_grasp is not None:
                                entry = np.zeros(3)
                            height, q_pre = self._highest(
                                grasp.pos, grasp.approach, grasp.jaw, q_grasp, sampled_height, offset=entry
                            )
                        if (
                            min(self.arm.jaw_clearances(q_grasp, self.held_geoms))
                            < cfg.approach_clearance + 0.002
                        ):
                            continue
                        if (
                            min(self.arm.jaw_clearances(q_pre, self.held_geoms))
                            < cfg.approach_clearance + 0.002
                        ):
                            continue
                        if empty_miss:
                            # Stop early on the already reachable approach, then close in free space.
                            # Re-solving a higher grasp plus a second pregrasp needlessly pushes the
                            # SO-101 beyond its vertical reach.
                            distance = self._miss_distance
                            if self._miss_mode == "high":
                                distance /= max(-grasp.approach[2], 0.01)
                            minimum = (
                                0.02 / max(-grasp.approach[2], 0.01) if self._miss_mode == "high" else 0.015
                            )
                            distance = min(distance, height - 0.005)
                            if distance < minimum:
                                continue
                            maximum = (
                                0.04 / max(-grasp.approach[2], 0.01) if self._miss_mode == "high" else 0.03
                            )
                            maximum = min(maximum, height - 0.005)
                            choices = np.linspace(distance, maximum, 4)
                            best = None
                            for stop in choices:
                                shift = -grasp.approach * stop
                                missed = grasp.pos + shift
                                try:
                                    q_missed = self.solve(missed, grasp.approach, grasp.jaw, init=q_pre)
                                except PlanningError:
                                    continue
                                q_missed[GRIPPER] = opening
                                clearance = np.inf
                                for value in np.linspace(0.0, opening, 5):
                                    trial = q_missed.copy()
                                    trial[GRIPPER] = value
                                    clearance = min(
                                        clearance, min(self.arm.jaw_clearances(trial, self.held_geoms))
                                    )
                                if best is None or clearance > best[0]:
                                    best = clearance, q_missed, missed, shift
                                if clearance >= cfg.approach_clearance + 0.002:
                                    break
                            if best is None or best[0] < 0.002:
                                continue
                            _, q_missed, missed, self._miss_shift = best
                            grasp.pos = missed
                            grasp.height += self._miss_shift[2]
                            q_grasp = q_missed
                        lift_height, _ = self._highest(
                            grasp.pos,
                            grasp.approach,
                            grasp.jaw,
                            q_grasp,
                            min(sampled_lift, 0.025) if empty_miss else sampled_lift,
                            away=UP,
                        )
                        if not self.arm.wrist_camera_ok(q_pre, target_body, self.held_geoms):
                            continue
                        if not self.arm.wrist_camera_ok(q_grasp, target_body, self.held_geoms):
                            continue
                        # Prefer a wrist sign whose turn stays above/beside the wrist en route to the
                        # look pose. The fully sampled planned path is checked again after planning.
                        clear_turn = True
                        for progress in () if close_start else np.linspace(0, 1, 9):
                            trial = self.q + progress * (q_pre - self.q)
                            if style == "tangential":
                                trial[WRIST_ROLL] = self.q[WRIST_ROLL] + (1 - (1 - progress) ** 2.5) * (
                                    q_pre[WRIST_ROLL] - self.q[WRIST_ROLL]
                                )
                            if not self.arm.wrist_camera_ok(trial):
                                clear_turn = False
                                break
                        if not clear_turn:
                            continue
                        found = True
                        break
                    except PlanningError:
                        continue
                if found:
                    break
            if found:
                break
        if not found:
            raise PlanningError(f"No reachable grasp of {obj.label}")
        self.grasp = grasp
        self.mark("approach_start")
        lift = grasp.pos + np.array([0.0, 0.0, lift_height])
        start_pos, start_rot = self.arm.tcp_pose(self.q)
        delta = grasp.pos - start_pos
        along = float(np.dot(delta, grasp.approach))
        across = float(np.linalg.norm(delta - along * grasp.approach))
        q_open = self.q.copy()
        q_open[GRIPPER] = opening
        direct = (
            0.01 <= along <= 0.08
            and across < 0.015
            and np.dot(start_rot[:, 0], grasp.approach) > 0.98
            and np.dot(start_rot[:, 2], grasp.jaw) > 0.96
            and self.arm.wrist_camera_ok(q_open, target_body, self.held_geoms)
            and not self.arm.collides(q_open)
        )
        if direct:
            self.mark("direct_grasp")
            self.gripper(opening)
        else:
            before = len(self.frames)
            start = self.q.copy()
            state = deepcopy(self.rng.bit_generator.state)
            flows = list(self._flows)
            self._blendable = 0
            for arc in (False, True):
                self._flows = list(flows)
                del self.frames[before:]
                del self.subtask_ids[before:]
                self.q = start.copy()
                self._blendable = 0
                self.rng.bit_generator.state = deepcopy(state)
                try:
                    self.transit(
                        q_pre,
                        gripper=opening,
                        early_roll=grasp.style == "tangential",
                        simple=not arc,
                        look_arc=arc,
                    )
                    if any(self.arm.collides(q) for q in [*self.frames[before::3], self.frames[-1]]):
                        raise PlanningError("Pregrasp transit collides")
                    if any(not self.arm.wrist_camera_ok(q) for q in self.frames[before:]):
                        raise PlanningError("Wrist camera is underneath")
                    break
                except PlanningError:
                    if arc:
                        raise
            self.check_initial_transit = False
        required_clearance = cfg.approach_clearance + (0.0 if direct else 0.002)
        before = len(self.frames)
        self.linear_move(
            grasp.pos,
            grasp.approach,
            grasp.jaw,
            blend_check=lambda q: (
                self.arm.wrist_camera_ok(q, target_body, self.held_geoms)
                and not self.arm.collides(q)
                and min(self.arm.jaw_clearances(q, self.held_geoms)) >= required_clearance
            ),
        )
        self.marks["looked"] = before - self._last_overlap
        self.marks["look_overlap_end"] = before
        for q in self.frames[self.marks["looked"] :]:
            if not self.arm.wrist_camera_ok(q, target_body, self.held_geoms):
                raise PlanningError("Object leaves the wrist camera during approach")
            if min(self.arm.jaw_clearances(q, self.held_geoms)) < required_clearance:
                raise PlanningError("Insufficient open-jaw approach clearance")
        for q in self.frames[self.marks["approach_start"] : self.marks["looked"]]:
            if min(self.arm.jaw_clearances(q, self.held_geoms)) < cfg.approach_clearance:
                raise PlanningError("Insufficient clearance while moving to look")
        self.mark("grasp_approach_end")
        # Let the servos settle on the grasp before closing on the object.
        self.dwell(rng.uniform(0.15, 0.3))
        self.mark("closing")
        # Squeeze: command past the object's width so the servo keeps holding it.
        close = max(0.0, self.arm.opening_for_width(grasp.width) - rng.uniform(15, 25))
        if self._miss_mode in ("loose", "wide"):
            close = self.arm.opening_for_width(grasp.width + 0.015 + np.linalg.norm(self._miss_shift))
        self.gripper(close, overlap=True)
        # Start lifting near the end of the close, once the fingers are mostly in position.
        lift_overlap = max(2, int(self._blendable * 0.1))
        self.mark("grasped")
        self.linear_move(lift, grasp.approach, grasp.jaw, max_overlap=lift_overlap, precision=1e-5)
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
        style = self.grasp.style if self.grasp is not None else "object_aligned"
        for candidate in candidate_tilts(tilt):
            approach = tilted_approach(
                pos, candidate, heading_offset=self.grasp.heading_offset if style == "tangential" else 0.0
            )
            # Carried: the jaw direction may follow, but the wrist keeps its roll.
            try:
                q_release = self.solve(pos, approach, None, keep_roll=True)
                if style == "tangential" and self.arm.collides(q_release, self.held_geoms):
                    continue
            except PlanningError:
                continue
            entry = self._off_axis(approach)
            # Down along the approach, or, when the point that high on its axis is out of reach (close
            # to the base), straight down from above the release, as one lowers something into a box.
            along = max(-approach[2], 0.3)
            options = (
                ((UP, rise, min_rise),)
                if style == "tangential"
                else ((-approach, rise / along, min_rise / along), (UP, rise, min_rise))
            )
            for away, target, floor in options:
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
        self.linear_move(
            pos,
            approach,
            None,
            via=pos + away * height / 2,
            keep_roll=True,
            blend_check=lambda q: self.arm.tcp_pose(q)[0][2] >= min_hover_z and self.arm.wrist_camera_ok(q),
        )
        self.dwell(rng.uniform(0.0, 0.08))
        self.mark("release")
        if self.rollout is not None:
            self.flush()
            name = self.active_object or self.rollout.episode.instance.roles["object"]
            if not self.rollout.held(name) and not self.rollout.transfer_satisfied(self.transfer_id):
                raise DroppedError(name)
        self.gripper(min(100.0, self.q[GRIPPER] + rng.uniform(25, 45)), overlap=True)
        self.linear_move(pos + away * height, approach, None, keep_roll=True)
        self.mark("released")

    def reach(self, pos: np.ndarray, tilt: float) -> None:
        """Move the gripper site to ``pos`` through joint space, at the reachable tilt nearest ``tilt``."""
        style = self.grasp.style if self.grasp else "object_aligned"
        start, _ = self.arm.tcp_pose(self.q)
        q = None
        for fraction in (1.0, 0.5, 0.25) if style == "tangential" else (1.0,):
            point = start + fraction * (pos - start)
            for candidate in candidate_tilts(tilt):
                try:
                    q = self.solve(
                        point,
                        tilted_approach(
                            point,
                            candidate,
                            heading_offset=self.grasp.heading_offset if style == "tangential" else 0.0,
                        ),
                        None,
                        keep_roll=True,
                    )
                    break
                except PlanningError:
                    continue
            if q is not None:
                break
        if q is None:
            return  # The object is already held at the lifted pose.
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
    grid = np.arange(0.0, 90.1, 7.5)
    return [sampled, *sorted(grid, key=lambda t: abs(t - sampled))]


def tilted_approach(pos: np.ndarray, tilt: float, heading_offset: float = 0.0) -> np.ndarray:
    """Pointing down, tilted by ``tilt`` degrees away from the robot."""
    radial = np.array([pos[0], pos[1], 0.0]) / max(float(np.hypot(pos[0], pos[1])), 1e-6)
    angle = np.deg2rad(heading_offset)
    radial[:2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ radial[:2]
    t = np.deg2rad(tilt)
    return DOWN * np.cos(t) + radial * np.sin(t)


def tangential_grasp_for(
    arm: Arm,
    obj: ObjectParams,
    tilt: float,
    rng: np.random.Generator,
    offset: float,
    min_height: float,
    current_roll: float = 0.0,
    heading_offset: float = -3.0,
    depth: float = 0.01,
) -> Grasp:
    """A grasp with tangential jaws, at a separately sampled approach tilt."""
    center = np.array([*obj.pos[:2], 0.0])
    reach = float(np.arctan2(center[1], center[0]))
    if obj.shape == "cylinder":
        width = 2 * obj.half_size[0]
    else:
        # A face must be radial so the opposing jaws close across parallel faces.
        delta = (obj.yaw - reach + np.pi / 4) % (np.pi / 2) - np.pi / 4
        if abs(delta) > np.deg2rad(20):
            raise PlanningError("Object faces are not aligned for tangential jaws")
        across = reach + np.deg2rad(heading_offset) + np.pi / 2
        angle = across - obj.yaw
        width = 2 * (obj.half_size[0] * abs(np.cos(angle)) + obj.half_size[1] * abs(np.sin(angle)))
    if width > arm.max_jaw_span:
        raise PlanningError("Object is too wide for tangential jaws")
    top = 2 * obj.half_size[-1]
    if tilt < 45:
        # At a near-vertical approach the housing has more table clearance.
        height = max(obj.half_size[-1], top - 0.03, 0.013)
    else:
        low, high = max(min_height, 0.4 * top), top - 0.01
        if low > high:
            raise PlanningError("Object is too short for a tangential inclined grasp")
        height = float(rng.uniform(low, high))
    sign = min((-1, 1), key=lambda s: abs(s * 90 - current_roll))
    # The rolled gripper's link geometry puts the TCP slightly off the pan plane. A small heading
    # correction leaves the approach radial to within a few degrees while allowing position IK to fit.
    heading = reach + np.deg2rad(heading_offset)
    t = np.deg2rad(tilt)
    approach = np.array([np.sin(t) * np.cos(heading), np.sin(t) * np.sin(heading), -np.cos(t)])
    jaw = sign * np.array([-np.sin(heading), np.cos(heading), 0.0])
    point = np.array([center[0], center[1], obj.support_height + height])
    pos = point - jaw * (width / 2 + arm.fixed_jaw_offset + offset) + approach * depth
    return Grasp(
        pos,
        approach,
        jaw,
        width,
        0.0,
        pos[2] - obj.support_height,
        tilt,
        sign * 90.0,
        "tangential",
        point,
        heading_offset,
        offset=offset,
    )


def grasp_for(
    arm: Arm,
    obj: ObjectParams,
    tilt: float,
    jaw_jitter: float,
    offset: float,
    current_roll: float = 0.0,
    keep_heading: bool = False,
) -> Grasp:
    """A grasp of a resting object from above, tilted outwards, closing across its narrow side.

    The jaw direction's sign picks which side the fixed jaw takes; the one nearest the wrist's
    ``current_roll`` (degrees) wins, as a person turns their wrist the short way from where it is.
    """
    center = np.array([obj.pos[0], obj.pos[1], obj.support_height + obj.half_size[-1]])
    approach = tilted_approach(center, tilt)
    reach = np.arctan2(center[1], center[0])
    if obj.shape == "cylinder":
        # Round: any direction, preferably as the wrist already is.
        roll = current_roll if keep_heading else np.clip(current_roll, -90, 90)
        yaw, width = reach + np.deg2rad(roll) + jaw_jitter, 2 * obj.half_size[0]
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
    jaw_length = np.linalg.norm(jaw)
    if jaw_length < 1e-5:
        raise PlanningError("Jaw direction is parallel to the approach")
    jaw /= jaw_length
    # Grip around mid-height, or higher on tall objects, the fingertips clear of the table.
    top = 2 * obj.half_size[-1]
    height = max(obj.half_size[-1], top - 0.03, 0.013)
    # A tilted gripper leans its jaws over the object: above the gripper site they shift towards the
    # robot by tan(tilt) per meter, as far as the jaw axis points along the reach.
    radial = np.array([np.cos(reach), np.sin(reach), 0.0])
    clearance = (top - height) * np.tan(np.deg2rad(tilt)) * abs(np.dot(horizontal, radial))
    # The fixed jaw sits next to the object's side; the moving jaw closes onto it.
    side = width / 2 + arm.fixed_jaw_offset + offset + clearance
    pos = np.array([center[0], center[1], obj.support_height + height]) - jaw * side
    return Grasp(
        pos=pos,
        approach=approach,
        jaw=jaw,
        width=width,
        clearance=clearance,
        height=height,
        tilt=tilt,
        roll=float(np.rad2deg(roll)),
        point=np.array([center[0], center[1], obj.support_height + height]),
        offset=offset,
    )
