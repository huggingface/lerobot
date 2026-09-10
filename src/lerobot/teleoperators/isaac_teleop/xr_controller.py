#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""XR (VR) controller device for NVIDIA Isaac Teleop, exposed to LeRobot.

A clutched device: the controller grip pose drives an in-pipeline ``SO101ClutchRetargeter``
and a safety rate limiter. :meth:`XRController._get_raw_action` (Isaac Teleop's own clutch,
session and preview machinery -- unchanged from the standalone example) returns an absolute
base-frame EE pose; the public :meth:`XRController.get_action` wraps it with the IK, gate,
hold-when-idle and reset-to-origin/declutch handling that used to live in the example's
``teleoperate.py``/``record.py`` loops, so this device is a drop-in ``--teleop.type`` for the
stock ``lerobot-teleoperate``/``lerobot-record`` (identity action processor, no bespoke loop).

The **robot twin, the drag preview, the leader ghost, the safety harness and the engage
gate** are all Isaac Teleop's ``viz.robot.ClutchPreview`` -- the same object
``examples/robot_viz`` runs. This device builds that example's retargeting graph and drives
the preview once per frame; nothing here reimplements any of it.

**The rebase is applied on the way IN**, through ``ControllersSource.transformed()``, and
its yaw is measured rather than configured -- see ``viz.robot.OperatorFrame``. That is inert
while disengaged, which is the whole reason it belongs there: the clutch emits its held pose
and never reads the controller until it latches, so a moving rebase perturbs nothing. Put
the same yaw inside the graph instead -- by re-expressing the clutch's home in it -- and a
pose that is static in the robot's frame swings at 0.69 m/s and 3.14 rad/s for a half-second
90 deg turn, past the harness's clamps.

The preview is fed the raw XR controller, which is the frame it draws in; only the clutch
sees the rebased one.

``isaacteleop`` imports are guarded behind the availability flag so this module imports
without it (construction fails fast via the base class).

**Needs the follower's measured joints** for FK-seeded clutch homing and the reset-to-origin
slew, which a ``Teleoperator`` has no channel to read on its own (``lerobot-teleoperate`` /
``lerobot-record`` construct the robot and the teleoperator independently, and only the robot
object holds live joint state). ``lerobot_teleoperate.py`` / ``lerobot_record.py`` special-case
this device to call ``send_feedback(robot.get_observation())`` once per frame, the same way
they already special-case ``unitree_g1``.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.import_utils import _isaacteleop_available

from .base import IsaacTeleopTeleoperator
from .config import XRControllerConfig
from .robot_profiles import ROBOT_PROFILES, build_xr_joint_pipeline, robot_profile_key
from .robot_twin import build_twin

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        OptionalTensorGroup,
        OutputCombiner,
        TensorGroup,
        ValueInput,
    )
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
else:
    ControllersSource = None
    ExecutionEvents = None
    ExecutionState = None
    OptionalTensorGroup = None
    OptionalType = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None

# The engage-relative clutch retargeter landed in isaacteleop 1.5; the rest of this device works
# against older releases. Resolve it tolerantly here and fail with an actionable message from
# XRController's constructor (see _require_clutch_retargeter) -- a hard import error here would
# take down the whole isaac_teleop package import instead.
SO101ClutchRetargeter = None
_CLUTCH_IMPORT_ERROR: Exception | None = None
if _isaacteleop_available:
    try:
        import isaacteleop.retargeters as _isaacteleop_retargeters

        SO101ClutchRetargeter = _isaacteleop_retargeters.SO101ClutchRetargeter
    except (ImportError, AttributeError) as exc:
        # Retained and chained below so a genuinely broken install is not misreported as
        # "upgrade isaacteleop".
        _CLUTCH_IMPORT_ERROR = exc

# `isaacteleop.viz.robot`'s package __init__ reaches the compiled Televiz extension, so a
# wheel built without -DBUILD_VIZ=ON has the clutch but no preview. Resolved tolerantly and
# reported from connect() -- a hard import here would take the whole device down.
ClutchPreview = None
clutch_preview: Any = None
if _isaacteleop_available:
    with contextlib.suppress(ImportError):
        from isaacteleop.viz.robot import ClutchPreview, OperatorFrame, clutch_preview

# The retargeters the preview's graph needs. These are plain numpy nodes and carry no viz
# dependency, so they resolve on any build.
ControllerPoseSource = None
EE_POSE_KEY = None
EePoseRateLimiter = None
RateLimiterConfig = None
SO101GripperRetargeter = None
GRIPPER_COMMAND_KEY = None
if _isaacteleop_available:
    try:
        from isaacteleop.retargeters.controller_pose import ControllerPoseSource
        from isaacteleop.retargeters.rate_limiter import (
            EE_POSE_KEY,
            EePoseRateLimiter,
            RateLimiterConfig,
        )
        from isaacteleop.retargeters.SO101.gripper_retargeter import (
            GRIPPER_COMMAND_KEY,
            SO101GripperRetargeter,
        )
    except (ImportError, AttributeError):
        pass

logger = logging.getLogger(__name__)

# What the safety harness lets through, and what the leader ghost therefore renders.
# robot_viz's own values, which its README is explicit are chosen for a demo rather than
# measured against an SO-101 -- RateLimiterConfig's own default is the more conservative
# 0.25 m/s. This bounds the pose that reaches LeRobot's IK, so raising it raises what the
# arm will be commanded to follow.
_HARNESS_DEFAULTS = {
    "max_linear_velocity": 0.5,  # m/s
    "max_angular_velocity": 2.5,  # rad/s, ~143 deg/s
    "reject_linear_velocity": 2.0,  # m/s
    "reject_angular_velocity": 10.0,  # rad/s
}

# The static-rebase leaf ControllerTransform reads. Its VALUE is not static any more --
# OperatorFrame measures the yaw and it is sent every step -- but the leaf is.
_REBASE_INPUT = "base_T_anchor"


def _transform_group(matrix: np.ndarray):
    """One frame of the rebase leaf."""
    group = TensorGroup(TransformMatrix())
    group[0] = np.asarray(matrix, dtype=np.float32)
    return group


_MIN_ISAACTELEOP_VERSION = "1.4.0"


def _require_clutch_retargeter() -> None:
    """Fail when the installed isaacteleop cannot supply the engage-relative clutch retargeter.

    Called from :meth:`XRController.__init__` rather than ``_build_pipeline``: the latter runs
    inside ``connect()``, *after* ``_ensure_cloudxr_runtime()``, so a purely static version
    mismatch would otherwise cost attaching to the CloudXR runtime before being reported.

    The probe is a CAPABILITY check, not a name check, and that distinction is load-bearing:
    ``SO101ClutchRetargeter`` also exists in isaacteleop 1.4, as a *different* retargeter (clutches
    position only, applies a fixed orientation offset, ``home_base_T_ee`` optional). Probing the
    name alone would therefore pass against 1.4 and then drive the arm wrongly, with no error.
    ``MEASURED_BASE_T_EE_INPUT`` exists only on the engage-relative implementation this device
    needs, so it is the signal that actually discriminates.
    """
    if SO101ClutchRetargeter is not None and hasattr(SO101ClutchRetargeter, "MEASURED_BASE_T_EE_INPUT"):
        return
    try:
        installed = importlib.metadata.version("isaacteleop")
    except importlib.metadata.PackageNotFoundError:
        installed = "an unknown version"
    raise ImportError(
        "XRController requires an isaacteleop whose SO101ClutchRetargeter is the engage-relative "
        f"full-pose clutch (it must expose MEASURED_BASE_T_EE_INPUT), but {installed} is "
        f"installed (>= {_MIN_ISAACTELEOP_VERSION} is necessary but not sufficient). Upgrade "
        "with:\n"
        '  uv pip install -U "isaacteleop[cloudxr,retargeters-lite]"'
    ) from _CLUTCH_IMPORT_ERROR


# Placeholder home for the retargeter, which must be constructed in ``_build_pipeline()`` (inside
# ``connect()``) — long before the arm's real EE pose is known. Safe because the session holds
# ``STOPPED`` until :meth:`XRController.start` is called, and the clutch cannot latch while
# STOPPED, so the home value is irrelevant in that window. :meth:`get_action` seeds the real one
# via :meth:`set_home_base_T_ee` before the first RUNNING frame (see :meth:`_finish_reset`).
_PLACEHOLDER_HOME_BASE_T_EE = np.eye(4, dtype=np.float64)


class XRController(IsaacTeleopTeleoperator):
    """Clutched XR controller teleoperator emitting a joint-space follower action.

    Reads the grip pose + squeeze + trigger off a ``ControllersSource`` rebased into the robot
    base frame, and drives them through an in-pipeline ``SO101ClutchRetargeter``
    (:meth:`_get_raw_action`, Isaac Teleop's own clutch/session/preview machinery). The public
    :meth:`get_action` wraps that with LeRobot's IK, the command gate, hold-when-idle, and the
    reset-to-origin slew that runs on connect and on every declutch.

    The follower's kinematics come from ``robot_profiles.ROBOT_PROFILES``, keyed off
    ``robot_config`` (see :meth:`__init__`) -- the ``--robot.*`` config from the same CLI
    invocation, passed in by ``make_teleoperator_from_config`` since the teleoperator and the
    robot are otherwise constructed independently.

    Call :meth:`send_feedback` with ``robot.get_observation()`` once per frame, BEFORE
    :meth:`get_action` -- see the module docstring.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop"

    def __init__(self, config: XRControllerConfig, *, robot_config: RobotConfig | None = None):
        # robot_config is the --robot.* config from the same CLI invocation (see
        # make_teleoperator_from_config): the only channel this device has to learn which
        # follower it's driving, since the teleoperator and the robot are otherwise
        # constructed independently. Falls back to the SO-101 profile if absent (e.g. direct
        # construction, or a caller that hasn't been updated to pass it through).
        if robot_config is not None:
            profile_key = robot_profile_key(robot_config.type, getattr(robot_config, "motor_family", None))
        else:
            profile_key = "so101_follower"
            logger.warning(
                "XRController constructed without robot_config; defaulting to the %r kinematics "
                "profile. Pass robot_config (e.g. via make_teleoperator_from_config(..., "
                "robot_config=cfg.robot)) so this is derived from --robot.type instead.",
                profile_key,
            )
        profile = ROBOT_PROFILES.get(profile_key)
        if profile is None:
            robot_type = robot_config.type if robot_config is not None else None
            raise ValueError(
                f"No RobotProfile for --robot.type={robot_type!r} (key={profile_key!r}). "
                f"Supported: {sorted(ROBOT_PROFILES)}."
            )
        self._profile = profile

        # Built BEFORE the base __init__: the twin creates the OpenXR session the trackers
        # then share, so it has to exist by the time connect() assembles the TeleopSessionConfig
        # (via the joint_publisher passed below).
        twin = None
        if config.robot_twin and profile.robot_twin:
            twin = build_twin(config)
        elif config.robot_twin:
            logger.info("robot_type=%s: not an SO-101; running without the robot twin preview.", profile_key)
        self._twin = twin

        super().__init__(config, joint_publisher=None if twin is None else twin.twin)
        self.config: XRControllerConfig = config
        # Before connect(), so a static version mismatch is reported without first paying for
        # attaching to the CloudXR runtime.
        _require_clutch_retargeter()

        self._kinematics = RobotKinematics(
            urdf_path=profile.urdf(),
            target_frame_name=profile.ee_frame,
            joint_names=profile.urdf_joint_names,
        )
        self._joint_pipeline = build_xr_joint_pipeline(profile, self._kinematics)
        if profile.clutch_position_scale is not None:
            self.config.clutch_position_scale = profile.clutch_position_scale

        # Whether the last _get_raw_action() read a tracked controller.
        self._is_tracking = False
        # The in-pipeline clutch, built in _build_pipeline() and retained so _get_raw_action()
        # can read its engagement state back after each step.
        self._retargeter: SO101ClutchRetargeter | None = None
        # Readiness interlock: STOPPED until start() is called. Never None on the wire — passing
        # None makes TeleopSession.step auto-fire RUNNING, which would defeat the interlock.
        self._execution_state: ExecutionState = ExecutionState.STOPPED
        # The arm's measured base_T_ee for this frame. CONSUMED AND CLEARED by _get_raw_action().
        self._measured_base_T_ee: np.ndarray | None = None
        # Whether set_home_base_T_ee() has run. start() refuses without it.
        self._home_seeded = False

        self._preview: Any = None
        # The rebase fed to the graph each step. Its yaw is measured off the preview arm
        # while disengaged and frozen through the engagement; until then it is the axis
        # convention from the config and nothing more.
        self._frame = OperatorFrame(np.asarray(config.base_T_anchor, dtype=np.float64))
        self._was_engaged = False
        self._clock: float | None = None

        # -- get_action()'s own state: gate, hold, IK pipeline reset edge, reset-to-origin --
        # The follower's measured joints, from the last send_feedback() call.
        self._last_observed: dict[str, float] | None = None
        # The last joint-space action get_action() returned; held while idle.
        self._last_commanded: dict[str, float] | None = None
        # Whether the joint pipeline (EEBoundsAndSafety's rate limiter, the IK warm start) was
        # anchored for the CURRENT engagement; re-anchored on every engage edge.
        self._pipeline_was_engaged = False
        # Armed on the first engaged frame of a segment; a segment that starts disengaged is
        # not treated as an immediate declutch. Mirrors the example's DeclutchLatch.
        self._declutch_armed = False
        # Set on declutch, consumed (and cleared) by get_teleop_events() -- the episode
        # boundary lerobot_record.py's record_loop polls for.
        self._episode_boundary_pending = False
        # Reset-to-origin state (see _start_reset/_step_reset/_finish_reset).
        self._resetting = False
        self._reset_start_pose: dict[str, float] | None = None
        self._reset_start_time: float | None = None
        self._reset_target: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """``examples/robot_viz``'s graph, verbatim: jaw, hand pose, clutch, safety harness.

        Everything runs in the **XR anchor frame** -- there is no ``ControllerTransform``
        rebase node, because ``ClutchPreview`` places its arm and ghost through
        ``viz.robot.frames`` and so requires that frame. :meth:`_get_raw_action` rebases the
        one pose LeRobot's IK consumes on the way out.

        ``ControllerPoseSource`` is a parallel branch rather than a link in the clutch's
        chain: its Optional output is the only tracking-validity oracle in the graph. The
        jaw is ungoverned -- the trigger is one scalar the operator drives directly, not a
        solved output that can diverge.
        """
        hand_key = f"controller_{self.config.hand_side}"
        controllers = ControllersSource(name="controllers")
        # The ONLY consumer of the rebased controller is the clutch, whose output drives a
        # real arm. The jaw retargeter reads a scalar squeeze, and the preview draws in the
        # operator's own frame, so both stay on the raw XR stream.
        rebase = ValueInput(_REBASE_INPUT, TransformMatrix())
        rebased = controllers.transformed(rebase.output("value")).output(hand_key)

        jaw = SO101GripperRetargeter(name="ghost_jaw", input_device=hand_key).connect(
            {hand_key: controllers.output(hand_key)}
        )
        hand = ControllerPoseSource(
            name="hand_pose", pose=clutch_preview.HAND_POSE, input_device=hand_key
        ).connect({hand_key: controllers.output(hand_key)})

        # OptionalType is load-bearing but is NOT permission to omit the key: a ValueInput
        # leaf is a required GRAPH input and TeleopSession validates every leaf NAME on
        # every step. It is what lets the key carry an ABSENT group, degrading to the
        # retargeter's last-commanded fallback instead of failing the graph.
        measured = ValueInput(SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT, OptionalType(TransformMatrix()))

        self._retargeter = SO101ClutchRetargeter(
            "so101_clutch",
            _PLACEHOLDER_HOME_BASE_T_EE,
            input_device=hand_key,
            position_scale=self.config.clutch_position_scale,
            squeeze_threshold=self.config.clutch_threshold,
            controller_pose=clutch_preview.HAND_POSE.value,
        )
        connections = {
            hand_key: rebased,
            SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT: measured.output("value"),
        }
        # Wired only when a preview exists to fill it. The input fails OPEN, so leaving it
        # unwired is exactly "every latch permitted" -- spelled by the absence of the edge
        # rather than by sending True forever.
        if self._twin is not None:
            permitted = ValueInput(clutch_preview.ENGAGE_PERMITTED_LEAF, clutch_preview.PERMITTED_TYPE)
            connections[SO101ClutchRetargeter.ENGAGE_PERMITTED_INPUT] = permitted.output("value")
        commanded = self._retargeter.connect(connections)

        governed = EePoseRateLimiter(name="harness", config=RateLimiterConfig(**_HARNESS_DEFAULTS)).connect(
            {EE_POSE_KEY: commanded.output(EE_POSE_KEY)}
        )

        return OutputCombiner(
            {
                ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
                ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
                GRIPPER_COMMAND_KEY: jaw.output(GRIPPER_COMMAND_KEY),
                clutch_preview.HAND_POSE_KEY: hand.output(EE_POSE_KEY),
                clutch_preview.COMMANDED_POSE_KEY: commanded.output(EE_POSE_KEY),
                EE_POSE_KEY: governed.output(EE_POSE_KEY),
            }
        )

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        try:
            if self._twin is not None:
                # After _build_pipeline, which is where the clutch retargeter it binds to
                # is created. owns_clutch_home=False: a real arm is on the other end of
                # this clutch, so its home comes from measured FK, not from wherever the
                # operator dragged the preview.
                self._preview = ClutchPreview(
                    self._twin.twin,
                    self._twin.monitor,
                    self._twin.arm,
                    self._retargeter,
                    self._twin.gate,
                    owns_clutch_home=False,
                    # The clutch runs in the ROBOT's frame now, and a pose there cannot be
                    # placed in the operator's hand -- the rebase's translation is unknown
                    # and cancels. The ghost goes on the hand; the harness speaks through
                    # colour rather than through lag.
                    ghost_pose_key=clutch_preview.HAND_POSE_KEY,
                )
                self._twin.arm.log_placement()
                clutch_preview.log_grip_posture(self._twin.arm)
        except Exception:
            # Roll the session/runtime back so a failed connect() leaves no half-state
            # (a live session behind a raised connect would leak the CloudXR runtime).
            self.disconnect()
            raise
        # Readiness interlock: seed the clutch home and re-home on every declutch. See
        # _start_reset()/_step_reset(); get_action() drains this before doing anything else.
        self._start_reset()

    def disconnect(self) -> None:
        self._execution_state = ExecutionState.STOPPED
        self._retargeter = None
        self._preview = None
        self._measured_base_T_ee = None
        self._home_seeded = False
        self._was_engaged = False
        self._clock = None
        self._last_observed = None
        self._last_commanded = None
        self._pipeline_was_engaged = False
        self._declutch_armed = False
        self._episode_boundary_pending = False
        self._resetting = False
        self._reset_start_pose = None
        self._reset_start_time = None
        self._reset_target = None
        super().disconnect()

    # ------------------------------------------------------------------
    # Readiness interlock and per-frame inputs
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Flip the session to ``RUNNING``, allowing the clutch to engage on a squeeze.

        Raises:
            RuntimeError: If :meth:`set_home_base_T_ee` has not been called. The placeholder home
                is the identity transform, and while the measured-EE input rescues the home
                *position* on engage, nothing rescues the home *orientation* -- it always comes
                from the last commanded rotation. An unseeded clutch would therefore snap the
                wrist to base-frame identity on the first squeeze: real arm motion, no exception,
                and it reads like an IK bug rather than a missing call.
        """
        if not self._home_seeded:
            raise RuntimeError(
                "set_home_base_T_ee() must be called before start(): the clutch would otherwise "
                "latch its home orientation from the identity placeholder and snap the wrist to "
                "base-frame identity on the first squeeze."
            )
        self._execution_state = ExecutionState.RUNNING

    def stop(self) -> None:
        """Return the session to ``STOPPED``, disengaging the clutch and re-arming its latch."""
        self._execution_state = ExecutionState.STOPPED

    def set_home_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Seed the clutch's held pose from the arm's measured ``base_T_ee`` [m].

        **Call this while the clutch is not engaged** -- in practice before :meth:`start`, while
        the session still holds ``STOPPED`` and latching is impossible.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected or self._retargeter is None:
            raise RuntimeError("Not connected. Call connect() first.")
        self._retargeter.set_home_base_T_ee(np.asarray(base_T_ee, dtype=np.float64))
        self._home_seeded = True

    def set_measured_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Supply the arm's measured ``base_T_ee`` [m] for the NEXT :meth:`_get_raw_action` only.

        The clutch latches its home *position* from this on the engage frame, so an arm that
        sagged or was pushed while disengaged is not commanded back to a stale target. The value
        is **consumed and cleared** on read -- see the example this was ported from for why.
        """
        self._measured_base_T_ee = np.asarray(base_T_ee, dtype=np.float64)

    # ------------------------------------------------------------------
    # Frame alignment
    # ------------------------------------------------------------------

    def _preview_bearing(self):
        """``(direction_xr, direction_base)`` for :class:`OperatorFrame`, or ``(None, None)``.

        The correspondence is the **preview arm's base against the real arm's base**, the
        one pair the operator can see both of: turn the wrist until the virtual arm is
        parallel to the real one, then squeeze.
        """
        if self._twin is None:
            return None, None
        # The preview's base yaw is a rotation about XR up, so its bearing comes straight
        # out of the quaternion, and the direction it faces is that bearing off XR forward.
        yaw = self._twin.arm.base_yaw_xr
        bearing = 2.0 * float(np.arctan2(yaw[2], yaw[0]))
        direction_xr = np.array([-np.sin(bearing), 0.0, -np.cos(bearing)])
        # The real base's own forward is +X, by definition of its frame.
        return direction_xr, np.array([1.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Engage gate (owned by ClutchPreview)
    # ------------------------------------------------------------------

    @property
    def engageable(self) -> bool:
        """Whether the clutch would latch on a squeeze right now, as of the last frame.

        Always ``True`` without a preview, so a caller needs no second branch.
        """
        return True if self._twin is None else bool(self._twin.gate.permitted)

    # ------------------------------------------------------------------
    # Action / feedback features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {f"{name}.pos": float for name in self._profile.motor_names}

    @property
    def feedback_features(self) -> dict:
        return {f"{name}.pos": float for name in self._profile.motor_names}

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`_get_raw_action` read a tracked controller."""
        return self._is_tracking

    # ------------------------------------------------------------------
    # Raw action extraction (Isaac Teleop's clutch/session/preview -- unchanged from the
    # standalone example; get_action() below is the only new consumer of this.)
    # ------------------------------------------------------------------

    def _get_raw_action(self) -> RobotAction:
        """Drive the preview and the graph one frame, and return the raw EE target.

        The frame order is ``robot_viz``'s, and the whole of it lives in ``ClutchPreview``:
        ``before_step`` anchors the arm to the head and emits the engage-permission leaf,
        ``step`` runs the graph, ``after_step`` advances the phase, drags the arm, writes
        the ghost's mocap rows and re-judges the gate.

        Two things this device owns on top. The session's ``execution_state`` is the
        readiness interlock and overrides the preview's own (which is always RUNNING -- it
        has no arm to home). And the pose handed back is rebased out of the anchor frame
        the graph runs in, into the robot base frame LeRobot's IK wants.

        The returned pose is the **rate limiter's** output, not the clutch's raw one. That
        is what the leader ghost renders, so commanding anything else would make the ghost
        a lie -- an intervention the operator can see is the entire reason the harness is
        in the graph.

        Returns:
            ``{"ee_pose": (7,), "trigger": float, "engaged": bool, "is_tracking": bool}``,
            with ``ee_pose`` in the robot base frame and ``trigger`` the in-graph gripper
            closedness (the same scalar that swings the ghost's jaw).
        """
        now = time.perf_counter()
        dt = 1.0 / 30.0 if self._clock is None else max(1e-4, now - self._clock)
        self._clock = now

        if self._preview is not None:
            external_inputs, events = self._preview.before_step(self.head_pose)
            reset = bool(events.reset)
        else:
            external_inputs, reset = {}, False

        external_inputs[_REBASE_INPUT] = {"value": _transform_group(self._frame.transform)}

        measured = self._measured_base_T_ee
        # Consume: the value is valid for exactly this frame (see set_measured_base_T_ee).
        self._measured_base_T_ee = None
        if measured is not None:
            measured_group = TensorGroup(TransformMatrix())
            measured_group[0] = measured.astype(np.float32)
        else:
            # NOT redundant -- do not "optimise" this branch away. TeleopSession.step
            # validates that every external leaf NAME appears on every step, independently
            # of the leaf's OptionalType. An absent OptionalTensorGroup satisfies that and
            # reaches the retargeter as is_none, landing on its last-commanded fallback.
            measured_group = OptionalTensorGroup(TransformMatrix())
        external_inputs[SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT] = {"value": measured_group}

        result = self._step(
            execution_events=ExecutionEvents(execution_state=self._execution_state, reset=reset),
            external_inputs=external_inputs,
        )

        if self._preview is not None:
            self._preview.after_step(result, dt)

        # The limiter's output, in the anchor frame, rebased into the robot's.
        governed = np.asarray(np.from_dlpack(result[EE_POSE_KEY][0]), dtype=np.float64)

        # HAND_POSE_KEY is the graph's only tracking-validity oracle -- it goes absent on an
        # invalid pose rather than holding the last one, which is exactly the gap a consumer
        # needs to see. Derived here rather than from the controller group so the device and
        # the preview agree on what "tracked" means.
        hand = result[clutch_preview.HAND_POSE_KEY]
        self._is_tracking = not hand.is_none
        if self._is_tracking:
            try:
                hand[0]
            except ValueError:
                self._is_tracking = False

        # The in-graph closedness rather than a raw trigger read: it is the same scalar that
        # drives the ghost's jaw, so what the operator sees and what the arm is commanded
        # cannot diverge.
        try:
            trigger = float(result[GRIPPER_COMMAND_KEY][0])
        except (IndexError, KeyError, TypeError, ValueError):
            trigger = 0.0

        engaged = bool(self._retargeter.is_engaged) if self._retargeter is not None else False

        # Held while engaged: the frame the operator engaged under has to be the one they
        # finish in, or the rebased controller -- and with it the arm -- would jump.
        if self.config.align_frame_to_preview:
            self._frame.update(*self._preview_bearing(), engaged=engaged)
        if engaged and not self._was_engaged:
            logger.info(
                "Clutch engaged; XR frame %s.",
                f"measured {np.degrees(self._frame.yaw_rad):+.0f} deg off base_T_anchor"
                if self._frame.measured
                else "as configured (no preview bearing yet)",
            )
        self._was_engaged = engaged

        # Already in the robot's frame: the clutch was handed a rebased controller.
        ee_pose = governed.astype(np.float32)

        return {
            "ee_pose": ee_pose,
            "trigger": trigger,
            "engaged": engaged,
            "is_tracking": self._is_tracking,
        }

    # ------------------------------------------------------------------
    # Measured-pose feedback (see the module docstring)
    # ------------------------------------------------------------------

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Cache the follower's measured joints, from ``robot.get_observation()``.

        :meth:`get_action` needs this every frame for the clutch's engage-edge homing (see
        :meth:`set_measured_base_T_ee`) and for the reset-to-origin slew's IK seed and
        interpolation start. Filters to this device's own ``robot_profile`` joints; extra keys
        (cameras, a different robot's motors) are ignored.
        """
        self._last_observed = {
            name: float(feedback[f"{name}.pos"])
            for name in self._profile.motor_names
            if f"{name}.pos" in feedback
        }
        if len(self._last_observed) != len(self._profile.motor_names):
            # Partial observation (e.g. the follower's motors don't match this profile) --
            # not enough to FK or slew from. Treat as "no observation yet".
            self._last_observed = None

    def _fk(self, joints: dict[str, float]) -> np.ndarray:
        q = np.array([joints[name] for name in self._profile.motor_names], dtype=float)
        return self._kinematics.forward_kinematics(q)

    # ------------------------------------------------------------------
    # get_action(): gate, IK, hold-when-idle, reset-to-origin/declutch (embeds what the
    # standalone example did in teleoperate.py/record.py/common.py).
    # ------------------------------------------------------------------

    def _step_session(self) -> RobotAction:
        """Feed the current measured pose in, then step the raw session one frame."""
        if self._last_observed is not None:
            self.set_measured_base_T_ee(self._fk(self._last_observed))
        return self._get_raw_action()

    def _start_reset(self) -> None:
        """Begin the reset-to-origin slew (or an in-place re-home if ``reset_to_origin=False``).

        Runs on connect and on every declutch. Holds the session ``STOPPED`` throughout: a
        squeeze mid-reset must not latch the clutch against a home the arm has not reached.
        """
        self.stop()
        self._resetting = True
        self._reset_start_pose = None  # captured lazily in _step_reset(), once observed
        self._reset_start_time = None
        self._reset_target = None
        # Re-arm the engage edge: the rate limiter still references the pre-reset command.
        self._pipeline_was_engaged = False

    def _step_reset(self) -> RobotAction:
        """One frame of the reset-to-origin slew.

        Always steps the raw session (STOPPED, so the clutch cannot latch) so tracking/preview
        state stays live. Interpolates in wall-clock time rather than a fixed step count, so it
        is invariant to the caller's ``--fps``.
        """
        self._step_session()

        if self._last_observed is None:
            # No measured pose yet: nothing to interpolate from or seed the clutch with.
            return self._hold()

        if self._reset_start_pose is None:
            self._reset_start_pose = dict(self._last_observed)
            self._reset_start_time = time.perf_counter()
            self._reset_target = {
                name: self._profile.reset_pose.get(name, self._reset_start_pose.get(name, 0.0))
                for name in self._profile.motor_names
            }

        if self.config.reset_to_origin:
            elapsed = time.perf_counter() - self._reset_start_time
            alpha = min(1.0, elapsed / max(self.config.reset_duration, 1e-6))
        else:
            alpha = 1.0  # re-home in place, no motion

        # Bare joint names for FK; robot.send_action() (like every other teleoperator's
        # get_action()) needs the ".pos"-suffixed action-feature convention instead.
        joints = {
            name: self._reset_start_pose[name]
            + alpha * (self._reset_target[name] - self._reset_start_pose[name])
            for name in self._profile.motor_names
        }
        action = {f"{name}.pos": value for name, value in joints.items()}
        self._last_commanded = action

        if alpha >= 1.0:
            self._resetting = False
            self.set_home_base_T_ee(self._fk(joints))
            self.start()

        return action

    def _hold(self) -> RobotAction:
        """Hold the last commanded pose while idle.

        Re-sending the freshly measured joints instead would ratchet the arm downward: under
        gravity a P-only servo settles below its goal by a steady-state error, so each
        re-command of the measurement would lower the goal by that error again. Falls back to
        the measured pose before any command has been sent (right after connect()).
        """
        if self._last_commanded is not None:
            return dict(self._last_commanded)
        if self._last_observed is not None:
            # _last_observed is keyed by bare joint name (see send_feedback); the action-feature
            # convention (and robot.send_action()) needs the ".pos" suffix.
            return {f"{name}.pos": value for name, value in self._last_observed.items()}
        raise RuntimeError(
            "XRController.get_action() called before any observation arrived via send_feedback(); "
            "the owning script must call send_feedback(robot.get_observation()) once per frame, "
            "before get_action()."
        )

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Report the declutch-ends-episode signal via the standard ``TeleopEvents`` protocol.

        Mirrors ``GamepadTeleop``/``KeyboardEndEffectorTeleop``. ``lerobot_record.py``'s
        ``record_loop`` polls this each frame; a declutch is reported once as
        ``TERMINATE_EPISODE`` and cleared here, so it is seen exactly once regardless of polling
        rate.
        """
        pending = self._episode_boundary_pending
        self._episode_boundary_pending = False
        return {
            TeleopEvents.IS_INTERVENTION: self._last_commanded is not None,
            TeleopEvents.TERMINATE_EPISODE: pending,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def get_action(self) -> RobotAction:
        """Return a joint-space action for the configured ``robot_profile``.

        Composes, every frame: the reset-to-origin slew (see :meth:`_start_reset`), the command
        gate (clutch engaged AND controller tracked), LeRobot's EE->joint IK, and the
        hold-when-idle latch -- so the caller can just do
        ``robot.send_action(teleop.get_action())`` like any other teleoperator.
        """
        if self._resetting:
            return self._step_reset()

        raw = self._step_session()
        engaged = bool(raw["engaged"])

        if engaged:
            self._declutch_armed = True
        elif self._declutch_armed:
            # Declutch: end the segment and re-home. The example's DeclutchLatch, inlined.
            self._declutch_armed = False
            self._episode_boundary_pending = True
            self._start_reset()
            return self._step_reset()

        if self._profile.home_orientation_from_measured and not engaged and self._last_observed is not None:
            # Re-home the clutch's ORIENTATION off the live pose while disengaged, so an arm
            # that sags there does not have the sag reconciled in one frame on the next engage.
            # Per-arm (see RobotProfile.home_orientation_from_measured): only safe where
            # measured-minus-commanded is sag alone.
            self.set_home_base_T_ee(self._fk(self._last_observed))

        if engaged and not self._pipeline_was_engaged:
            # Re-anchor the pipeline at the measured pose: EEBoundsAndSafety's rate limiter and
            # the IK warm start otherwise still reference the stale pre-disengage command.
            self._joint_pipeline.reset()
        self._pipeline_was_engaged = engaged

        gated = engaged and bool(raw["is_tracking"]) and self._last_observed is not None
        if not gated:
            return self._hold()

        ee_action = {
            "ee_pose": np.asarray(raw["ee_pose"], dtype=np.float32),
            "closedness": float(raw["trigger"]),
        }
        # _last_observed is keyed by bare joint name (see send_feedback); the joint pipeline's
        # IK step reads its ".pos"-suffixed seed straight off the raw observation convention
        # (like robot.get_observation()'s own keys), so it needs the suffix restored here too.
        obs_for_pipeline = {f"{name}.pos": value for name, value in self._last_observed.items()}
        action = self._joint_pipeline((ee_action, obs_for_pipeline))
        self._last_commanded = dict(action)
        return action
