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
and a safety rate limiter, so :meth:`XRController.get_action` returns an absolute base-frame
EE pose rather than a raw controller pose. Unlike the other devices here this one **holds
state across frames** — the clutch's latched home and origin live in the retargeter — so it
must be stepped every frame with real ``ExecutionEvents``. The gripper mapping stays in the
owning loop.

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
"""

from __future__ import annotations

import importlib.metadata
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction

from .base import IsaacTeleopTeleoperator, _isaacteleop_available
from .config_isaac_teleop import XRControllerConfig

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
    from isaacteleop.retargeting_engine.interface.tensor_group_type import (
        OptionalType,
        TensorGroupType,
    )
    from isaacteleop.retargeting_engine.tensor_types import (
        BoolType,
        ControllerInput,
        TransformMatrix,
    )
else:
    ControllersSource = None
    ControllerInput = None
    ExecutionEvents = None
    ExecutionState = None
    OptionalTensorGroup = None
    OptionalType = None
    TensorGroupType = None
    BoolType = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None

# The engage-relative clutch retargeter landed in isaacteleop 1.5; the rest of this example works
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
# reported from connect() -- a hard import here would take the whole example down.
ClutchPreview = None
_PREVIEW_IMPORT_ERROR: Exception | None = None
clutch_preview: Any = None
if _isaacteleop_available:
    try:
        from isaacteleop.viz.robot import ClutchPreview, OperatorFrame, clutch_preview
    except ImportError as exc:
        _PREVIEW_IMPORT_ERROR = exc

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


# There is deliberately no constant for the measured-EE leaf: the producer side reads
# ``SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT`` directly, so the producer key here and
# the consumer key there cannot drift apart. Re-declaring the literal would defeat that.

_MIN_ISAACTELEOP_VERSION = "1.4.0"


# There is deliberately no constant for the measured-EE leaf: the producer side reads
# ``SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT`` directly, so the producer key here and
# the consumer key there cannot drift apart. Re-declaring the literal would defeat that.

_MIN_ISAACTELEOP_VERSION = "1.4.0"


# Below this a direction is too near vertical to carry a bearing and its azimuth is noise.
# A gripper aimed straight down is a real posture, so this holds the last yaw, never raises.
_MIN_HORIZONTAL = 1e-3


def _require_clutch_retargeter() -> None:
    """Fail when the installed isaacteleop cannot supply the engage-relative clutch retargeter.

    Called from :meth:`XRController.__init__` rather than ``_build_pipeline``: the latter runs
    inside ``connect()``, *after* ``_ensure_cloudxr_runtime()``, so a purely static version
    mismatch would otherwise cost a ~30 s runtime launch and possibly an interactive EULA prompt
    before being reported.

    The probe is a CAPABILITY check, not a name check, and that distinction is load-bearing:
    ``SO101ClutchRetargeter`` also exists in isaacteleop 1.4, as a *different* retargeter (clutches
    position only, applies a fixed orientation offset, ``home_base_T_ee`` optional). Probing the
    name alone would therefore pass against 1.4 and then drive the arm wrongly, with no error.
    ``MEASURED_BASE_T_EE_INPUT`` exists only on the engage-relative implementation this device
    needs, so it is the signal that actually discriminates.

    ``_MIN_ISAACTELEOP_VERSION`` is deliberately still ``1.4.0``: the engage-relative clutch has
    not shipped in a published wheel yet, so naming a version PyPI cannot resolve would be worse
    advice than the capability probe above. Bump it -- and the install pins in ``README.md`` and
    ``docs/source/isaac_teleop.mdx`` -- when that wheel ships.
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
# STOPPED, so the home value is irrelevant in that window. The owning loop supplies the real one
# via :meth:`XRController.set_home_base_T_ee` before the first RUNNING frame.
_PLACEHOLDER_HOME_BASE_T_EE = np.eye(4, dtype=np.float64)


class XRController(IsaacTeleopTeleoperator):
    """Clutched XR controller teleoperator emitting an absolute base-frame EE pose.

    Reads the grip pose + squeeze + trigger off a ``ControllersSource`` rebased into the robot
    base frame, and drives them through an in-pipeline ``SO101ClutchRetargeter``.
    :meth:`get_action` returns the clutch-rebased absolute EE pose, the raw analog trigger, and
    whether the clutch is engaged; the owning loop owns the gripper mapping and the safety gate.

    Lifecycle, which the owning loop must drive:

    1. :meth:`connect` builds the pipeline and opens the session. The session holds ``STOPPED``,
       so the clutch cannot latch.
    2. The loop waits for the headset and homes the arm, stepping this device throughout.
    3. The loop calls :meth:`set_home_base_T_ee` with the arm's measured EE pose, then
       :meth:`start` — which flips the session to ``RUNNING`` and allows the clutch to engage.

    Holding ``STOPPED`` for steps 1-2 is a readiness interlock, not a formality: the graph is
    stepped throughout the connect wait and the homing slew, and the operator is tracked during
    both. Without it a squeeze while donning the headset would latch a home the arm has not
    reached.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig, *, twin: Any | None = None):
        super().__init__(config, joint_publisher=None if twin is None else twin.twin)
        self.config: XRControllerConfig = config
        # Before connect(), so a static version mismatch is reported without first paying for the
        # CloudXR runtime launch (and a possible interactive EULA prompt).
        _require_clutch_retargeter()

        # Whether the last get_action() read a tracked controller; the owning loop polls this
        # to wait for the operator to connect before driving the arm.
        self._is_tracking = False
        # The in-pipeline clutch, built in _build_pipeline() and retained so get_action() can read
        # its engagement state back after each step.
        self._retargeter: SO101ClutchRetargeter | None = None
        # Readiness interlock: STOPPED until start() is called. Never None on the wire — passing
        # None makes TeleopSession.step auto-fire RUNNING, which would defeat the interlock.
        # Safe to name the enum here: the base __init__ above calls _require_isaacteleop(), which
        # raises before returning when isaacteleop is absent.
        self._execution_state: ExecutionState = ExecutionState.STOPPED
        # The arm's measured base_T_ee for this frame. CONSUMED AND CLEARED by get_action().
        self._measured_base_T_ee: np.ndarray | None = None
        # Whether set_home_base_T_ee() has run. start() refuses without it: the placeholder home
        # is identity, and an unseeded ORIENTATION has no measured-input rescue path.
        self._home_seeded = False

        # Isaac Teleop's own ClutchPreview, built in connect() once the graph exists (it
        # binds to the clutch retargeter). None when there is no twin to drive.
        self._twin = twin
        self._preview: Any = None
        # anchor_T_base and its inverse, from the config's static rebase. The graph runs in
        # the anchor frame; these carry the arm's measured pose in and the commanded pose
        # back out. See the module docstring on why that is equivalent.
        # The rebase fed to the graph each step. Its yaw is measured off the preview arm
        # while disengaged and frozen through the engagement; until then it is the axis
        # convention from the config and nothing more.
        self._frame = OperatorFrame(np.asarray(config.base_T_anchor, dtype=np.float64))
        self._was_engaged = False
        self._clock: float | None = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """``examples/robot_viz``'s graph, verbatim: jaw, hand pose, clutch, safety harness.

        Everything runs in the **XR anchor frame** -- there is no ``ControllerTransform``
        rebase node, because ``ClutchPreview`` places its arm and ghost through
        ``viz.robot.frames`` and so requires that frame. :meth:`get_action` rebases the one
        pose LeRobot's IK consumes on the way out.

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

    def disconnect(self) -> None:
        self._execution_state = ExecutionState.STOPPED
        self._retargeter = None
        self._preview = None
        self._measured_base_T_ee = None
        self._home_seeded = False
        self._was_engaged = False
        self._clock = None
        super().disconnect()

    # ------------------------------------------------------------------
    # Readiness interlock and per-frame inputs (driven by the owning loop)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Flip the session to ``RUNNING``, allowing the clutch to engage on a squeeze.

        Call once the arm is at its home pose and :meth:`set_home_base_T_ee` has been given that
        pose. Before this, squeezing does nothing. Takes effect on the next :meth:`get_action`.

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
        """Return the session to ``STOPPED``, disengaging the clutch and re-arming its latch.

        Takes effect on the next :meth:`get_action`; the value reported by ``get_action`` still
        reflects the last computed frame until then.
        """
        self._execution_state = ExecutionState.STOPPED

    def set_home_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Seed the clutch's held pose from the arm's measured ``base_T_ee`` [m].

        The retargeting graph is built in :meth:`connect`, long before the arm has been homed, so
        the retargeter starts on an identity placeholder. **Call this while the clutch is not
        engaged** -- in practice before :meth:`start`, while the session still holds ``STOPPED``
        and latching is impossible, which makes the ordering unambiguous: the new home takes
        effect before the first ``RUNNING`` frame. Calling it later re-arms the clutch's pending
        latch rather than jumping the arm, but is not the intended use.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected or self._retargeter is None:
            raise RuntimeError("Not connected. Call connect() first.")
        # Into the anchor frame the graph runs in. The clutch's algebra is equivariant under
        # this rebase, so seeding here and rebasing the output back out is exactly the same
        # command as rebasing the controller on the way in.
        # No conversion: the clutch consumes a controller already rebased into the robot's
        # frame, so its home is in that frame too, and a moving rebase never touches it.
        self._retargeter.set_home_base_T_ee(np.asarray(base_T_ee, dtype=np.float64))
        self._home_seeded = True

    def set_measured_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Supply the arm's measured ``base_T_ee`` [m] for the NEXT :meth:`get_action` only.

        The clutch latches its home *position* from this on the engage frame, so an arm that
        sagged or was pushed while disengaged is not commanded back to a stale target. The home
        orientation is never taken from it.

        The value is **consumed and cleared** by :meth:`get_action`. That is deliberate: a value
        that persisted would silently feed a stale forward-kinematics result forever the day the
        loop stopped calling this, whereas consume-on-read makes "stale by one frame"
        unrepresentable — the pose is either this frame's or absent, and absent lands on the
        retargeter's documented last-commanded fallback.

        No timestamp travels with the pose, and none is checked. "This frame" therefore means the
        caller's frame, not the retargeting graph's: on a ``frame_deadline_miss``
        (see ``base.py``'s stale-frame warning) the clutch can latch its home off forward
        kinematics that is one or two frames — roughly 33–66 ms at 30 Hz — old. At clutch speeds
        that is a small position error on the engage frame, not a stability problem, so it is
        recorded rather than mechanised.
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

        Not the aim ray against the jaw, which was tried first and is degenerate by
        construction -- the preview's ``base_yaw_bias`` exists precisely to keep its JAW
        pointing where the controller does, so the aim ray carries no information about the
        arm's heading. And with ``shoulder_pan`` near zero an SO-101's jaw sits within a few
        degrees of base +X, so the measured yaw collapsed to nothing.
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

        ``True`` throughout an engagement as well -- it is the gate's own ``engaged or
        aligned``, which is what an affordance should say: *the clutch is yours*. Always
        ``True`` without a preview, so a caller needs no second branch. The preview already
        paints this onto the arm; the property is for anything else that wants to know.
        """
        return True if self._twin is None else bool(self._twin.gate.permitted)

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "ee_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"x": 0, "y": 1, "z": 2, "qx": 3, "qy": 4, "qz": 5, "qw": 6},
            },
            # ``get_action`` returns a scalar for this, so the advertised shape is () (0-d)
            # to stay consistent with the returned value.
            "trigger": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "engaged": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
            # Returned per-frame as well as via the :attr:`is_tracking` property, so that both
            # halves of the command gate travel in one object and omitting one is not possible.
            "is_tracking": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read a tracked controller. ``False`` until the
        headset is connected over CloudXR and its controllers are live; the owning loop polls
        it to wait for the operator before commanding the arm."""
        return self._is_tracking

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Drive the preview and the graph one frame, and return the EE target.

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
            closedness (the same scalar that swings the ghost's jaw). **The command gate is
            the conjunction** ``engaged and is_tracking``; both are returned together so
            omitting one is not possible.
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
