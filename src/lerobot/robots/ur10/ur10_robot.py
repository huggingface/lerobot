"""
UR10e robot integration for HIL-SERL.

Wraps ur_rtde RTDE interfaces + the same custom serial gripper used on RC10 + OpenCV cameras
into a lerobot Robot-like interface, and provides a Gym environment with task-space servoL
streaming control.

Control model: a dedicated background thread inside `UR10Robot` continuously issues
`rtde_ctrl.servoL(target, ...)` at ~200 Hz. `env.step()` is non-blocking — it just updates the
shared target via `robot.set_target_pose(...)`. This keeps URScript traffic continuous (so the
robot-side watchdog never fires across long idle gaps such as between-episode video encoding)
AND gives UR's velocity-FF interpolator a smooth high-rate reference (so stick releases settle
cleanly without the coast-and-reverse artifacts seen with per-`step()` 10 Hz `servoL` calls).

Observation model:
    `tcp_xyz` is RELATIVE to the per-episode home pose, captured at the end of
    `env.reset()`. This follows the HIL-SERL paper (arXiv 2410.21845):
        "the robot's proprioceptive information is expressed with respect to the frame of
         the end-effector's initial pose"
    Effect: at reset the policy sees `tcp_xyz ≈ (0, 0, 0)` regardless of where in the
    workspace the home pose was randomized — the policy generalizes across spatial
    placement of the task object.

    When `use_yaw=True`, the env layer ALSO inserts a `yaw_offset` slot (radians, tool-Z
    rotation relative to the per-episode home orientation) between `tcp_xyz` and
    `gripper_state`, giving a 17-D observation. Yaw is decoded from RTDE TCP via
    `R_initial.inv() * R_current`, the exact inverse of the action-side
    `R_home * R_z(target_yaw)` composition — so commanded and observed yaw stay
    consistent. See `UR10RobotEnv._augment_observation` and `get_measured_yaw_offset`.

    `joint_pos`, `joint_vel`, `gripper_state` are absolute — matches gym-hil's convention.

    UR10e has no real wrist F/T sensor; `getActualTCPForce()` is an inverse-dynamics
    estimate from joint motor currents and exhibits ±5 N noise plus tens of N of
    pose-dependent bias drift. We deliberately do NOT include it in the observation —
    the policy relies on visual + proprioceptive signals for contact.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401  # registers "intelrealsense" with draccus
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep


if TYPE_CHECKING:
    import rtde_control
    import rtde_io
    import rtde_receive
    from rc10_api.gripper import Gripper


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@RobotConfig.register_subclass("ur10")
@dataclass
class UR10RobotConfig(RobotConfig):
    """Connection and hardware configuration for the UR10e robot."""

    # Network / RTDE
    ip: str = "192.168.0.100"
    rtde_frequency: int = 500  # UR10e native 500 Hz; clamp to [1, 500]

    # Tool centre point and payload — applied at connect time.
    # tcp_offset is [x, y, z, rx, ry, rz] (m, axis-angle radians) from the flange.
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload_mass: float = 0.0  # kg
    payload_cog: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x,y,z] m

    # Gripper (same custom serial gripper that RC10 used; reuse rc10_api.gripper.Gripper)
    gripper_port: str = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200

    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@dataclass
class UR10RobotEnvConfig:
    """Task-space servoL streaming parameters for the UR10e environment."""

    # End-effector step sizes in metres per unit action (action range is [-1, 1]).
    # Matches gym-hil's `DEFAULT_EE_STEP_SIZE` (`gym_hil/wrappers/hil_wrappers.py:26`):
    # 1 mm per unit action → at policy fps 10 Hz, max sprint speed is ~1 cm/s, giving a
    # ~1-second human reaction window before the EE has moved 1 cm. Higher values cause
    # the policy to over-shoot during early training before the user can intervene; for
    # contact-rich tasks (USB insertion, peg-in-hole) keep at 0.001 or lower.
    #
    # When `use_yaw=True`, this dict gains an optional "yaw" key (radians per unit
    # action). Default 0.01 rad/unit is roughly 0.57°/unit — comparable in magnitude
    # to the xyz mm/unit scale at 10 Hz.
    ee_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.001, "y": 0.001, "z": 0.001}
    )

    # Cartesian workspace bounds [x, y, z] (metres). Populated via ur10_find_limits.
    # When `use_yaw=True`, the lists optionally carry a 4th element interpreted as a
    # RELATIVE yaw OFFSET window (radians) around `fixed_rz`. The env latches
    # `target_yaw = 0.0` at reset and clips accumulated yaw to [min[3], max[3]]. The
    # mixing of absolute xyz and relative yaw at index 3 is intentional — it keeps
    # all EE limits in one place. If `use_yaw=True` and the 4th element is missing,
    # the env falls back to ±π/2 (1.5708 rad).
    ee_bounds_min: list[float] = field(default_factory=lambda: [-0.5, -0.5, 0.05])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.7])

    # Fixed wrist orientation in AXIS-ANGLE (UR-native), NOT Euler. When `use_yaw=True`
    # this is the *home* orientation; the env accumulates a tool-Z yaw offset on top.
    fixed_rx: float = 3.14159
    fixed_ry: float = 0.0
    fixed_rz: float = 0.0

    # Home TCP for reset: [x, y, z, rx, ry, rz]
    home_tcp: list[float] = field(
        default_factory=lambda: [0.5, 0.0, 0.4, 3.14159, 0.0, 0.0]
    )
    reset_time_s: float = 5.0

    use_gripper: bool = True
    # When True, the action gains a 4th continuous DOF (yaw) at index 3 — see step().
    # Default False keeps existing recorded datasets / JSON configs working unchanged.
    use_yaw: bool = False

    # Start-position randomization (metres) — uniform [-r, +r] applied at reset.
    randomization_xy: float = 0.0
    randomization_z: float = 0.0

    # Background streaming-thread frequency (Hz). 200 Hz = 5 ms per servoL call, well above
    # the 10 Hz policy rate and well below UR10e's 500 Hz RTDE max. The thread keeps servoL
    # traffic continuous so (a) the URScript watchdog never fires across idle gaps and (b)
    # UR's velocity-FF estimator sees a smooth high-rate reference rather than 100 ms steps.
    stream_frequency_hz: int = 200

    # servoL streaming tuning. Defaults are the *conservative* end of the ur_rtde range:
    # gain at the floor and lookahead longer than the streaming dt for smooth tracking.
    # Higher gain or shorter lookahead causes the controller to overshoot and ring. Lower
    # gain first when tuning, not raise it.
    servo_lookahead_time: float = 0.15  # range [0.03, 0.2]
    servo_gain: float = 100.0  # range [100, 2000]

    # Reset move parameters (blocking moveL)
    reset_speed: float = 0.1  # m/s
    reset_acceleration: float = 0.1  # m/s^2


# ---------------------------------------------------------------------------
# UR10e Robot driver
# ---------------------------------------------------------------------------


class UR10Robot:
    """Low-level interface for UR10e via ur_rtde + serial gripper + cameras.

    A background "streaming thread" owns all `servoL` traffic. The thread runs at the
    configured `stream_dt` cadence (servoL itself blocks for `dt` inside ur_rtde's C++ layer,
    so the loop self-times). External callers update the target via `set_target_pose(pose)`,
    which is a sub-microsecond locked write — there is no public per-call servo entry point.
    Blocking `moveL` (used by reset) pauses the streaming thread, executes, then resumes.

    Concurrency:
      - `_target_lock` guards `_target` (the shared list[6] target pose).
      - `_ctrl_lock` guards every `RTDEControlInterface` call (servoL/moveL/servoStop/
        stopScript/setTcp/setPayload). RTDEControlInterface is NOT thread-safe per ur_rtde
        docs, so all access is serialized.
      - `RTDEReceiveInterface` (telemetry) lives on a separate socket and is read freely
        from the main thread without locking.
    """

    def __init__(self, config: UR10RobotConfig):
        self.config = config
        self.rtde_ctrl: rtde_control.RTDEControlInterface | None = None
        self.rtde_rec: rtde_receive.RTDEReceiveInterface | None = None
        self.rtde_io: rtde_io.RTDEIOInterface | None = None
        self.gripper: Gripper | None = None
        self.cameras: dict[str, Camera] = {}
        self._connected = False

        # Streaming-thread state.
        self._target: list[float] | None = None
        self._target_lock = threading.Lock()
        self._ctrl_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        # Set when the streaming loop pauses *because of an exception* (vs. an intentional
        # pause from move_to_pose). Lets `streaming_healthy()` distinguish "robot's fine,
        # we paused for reset" from "servoL crashed, robot is now uncommanded".
        self._stream_failed = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._stream_dt: float = 0.005
        self._stream_lookahead: float = 0.15
        self._stream_gain: float = 100.0

        # Per-episode TCP-xyz baseline for the HIL-SERL relative-position observation.
        # Populated by `capture_baselines()` (called from `UR10RobotEnv.reset()` after the
        # arm has settled at the home pose) and subtracted in `get_observation()`. None
        # until the first reset — `get_observation()` falls back to absolute readings in
        # that case so observation-space sampling at construction time still works.
        # `tcp_xyz` gets relative semantics: encoder readings are noise-free and the
        # paper explicitly requires this.
        self._initial_tcp_xyz: np.ndarray | None = None
        # Per-episode wrist-orientation baseline, captured by `capture_baselines()` next
        # to `_initial_tcp_xyz`. Stored as a scipy Rotation so per-step `get_measured_
        # yaw_offset()` is a cheap `R_initial.inv() * R_current` composition. Always
        # populated even if `use_yaw=False` at the env layer — the cost is one matrix
        # multiply per reset and lets the env opt into the yaw observation without the
        # driver needing to know about the use_yaw flag.
        self._initial_tcp_rotation: Rotation | None = None

    # -- lifecycle ----------------------------------------------------------

    def _reset_realsense_devices(self) -> None:
        """Hardware-reset every configured RealSense before opening it.

        D405 devices left in a stuck state by a prior process produce zero color frames
        until reset. Doing this in connect() makes startup deterministic. No-op when no
        RealSense is configured — pyrealsense2 is only imported if needed.
        """
        rs_serials = {
            cfg.serial_number_or_name
            for cfg in self.config.cameras.values()
            if getattr(cfg, "type", None) == "intelrealsense"
        }
        if not rs_serials:
            return

        import pyrealsense2 as rs

        ctx = rs.context()
        reset_any = False
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial in rs_serials:
                logger.info("Hardware-resetting RealSense %s", serial)
                dev.hardware_reset()
                reset_any = True
        if reset_any:
            time.sleep(5.0)

    @property
    def is_connected(self) -> bool:
        if not self._connected or self.rtde_ctrl is None:
            return False
        try:
            return bool(self.rtde_ctrl.isConnected())
        except Exception:
            return False

    def connect(self) -> None:
        import rtde_control
        import rtde_io
        import rtde_receive
        from rc10_api.gripper import Gripper

        logger.info(
            "Connecting to UR10e at %s (RTDE %d Hz) ...",
            self.config.ip, self.config.rtde_frequency,
        )

        # Three RTDE handles. Receive and Control both consume RTDE bandwidth — keep the
        # frequency identical to avoid mismatched packet rates.
        self.rtde_ctrl = rtde_control.RTDEControlInterface(
            self.config.ip, float(self.config.rtde_frequency)
        )
        self.rtde_rec = rtde_receive.RTDEReceiveInterface(
            self.config.ip, float(self.config.rtde_frequency)
        )
        self.rtde_io = rtde_io.RTDEIOInterface(self.config.ip)

        # Apply TCP offset and payload before any motion command — telemetry and gravity
        # compensation depend on these. Lock for invariant correctness even though no
        # streaming thread is running yet.
        with self._ctrl_lock:
            self.rtde_ctrl.setTcp(list(self.config.tcp_offset))
            self.rtde_ctrl.setPayload(
                float(self.config.payload_mass), list(self.config.payload_cog)
            )

        # Same custom serial gripper as RC10.
        self.gripper = Gripper(
            device=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
        )

        self._reset_realsense_devices()
        self.cameras = make_cameras_from_configs(self.config.cameras)
        for cam in self.cameras.values():
            cam.connect()

        self._connected = True
        logger.info("UR10e connected.")

    def _ctrl_call_with_deadline(self, name: str, fn, deadline_s: float) -> None:
        """Run a blocking RTDE control call on a side thread with a hard deadline.

        Side thread acquires ``_ctrl_lock`` for the duration so the runtime mutual-
        exclusion invariant is preserved. On deadline:
          1. Force-disconnect ``rtde_ctrl`` -- kills the side thread's blocking RTDE
             socket call so it unwinds and releases the lock.
          2. Wait briefly for the side thread to exit.
          3. Reconnect ``rtde_ctrl`` and reapply TCP / payload so the next ctrl
             call (e.g. ``moveL`` from ``move_to_pose``, or the streaming thread's
             ``servoL``) hits a fresh URScript.

        If the side thread doesn't unwind, or reconnect fails, we set
        ``_stream_failed`` so ``streaming_healthy()`` flips and the next
        ``env.step()`` raises rather than recording frames against a dead handle.

        Used by both ``_pause_streaming`` (mid-run; recovery matters) and
        ``disconnect`` (teardown; recovery is best-effort).
        """
        done = threading.Event()
        err: list[BaseException] = []

        def target() -> None:
            try:
                with self._ctrl_lock:
                    fn()
            except BaseException as e:
                err.append(e)
            finally:
                done.set()

        t = threading.Thread(target=target, name=f"ur10-{name}", daemon=True)
        t.start()

        if done.wait(deadline_s):
            if err:
                logger.exception("%s failed", name, exc_info=err[0])
            return

        # Deadline blown. Empirically, closing the socket does NOT wake the wedged
        # C++ call (ur_rtde's servoStop isn't sitting in a plain recv). We can't
        # interrupt the side thread, so we abandon it and replace the state it owns:
        #   - close the old rtde_ctrl socket so the controller releases its URScript
        #   - construct a fresh RTDEControlInterface (controller accepts us back)
        #   - swap in a fresh _ctrl_lock so subsequent code paths don't block on the
        #     lock that the dead side thread will hold forever
        # The dead thread is daemon; it dies with the process. Old rtde_ctrl object
        # is kept alive only by the side thread's stack frame and GC'd when (if) it
        # ever returns.
        logger.warning(
            "%s wedged for %.1fs; replacing rtde_ctrl + ctrl_lock to recover",
            name, deadline_s,
        )
        try:
            self.rtde_ctrl.disconnect()
        except Exception:
            logger.exception("old rtde_ctrl.disconnect failed (continuing anyway)")

        try:
            import rtde_control
            new_ctrl = rtde_control.RTDEControlInterface(
                self.config.ip, float(self.config.rtde_frequency)
            )
            new_lock = threading.Lock()
            # Reapply TCP / payload before exposing the new handle so gravity-comp
            # stays consistent.
            with new_lock:
                new_ctrl.setTcp(list(self.config.tcp_offset))
                new_ctrl.setPayload(
                    float(self.config.payload_mass), list(self.config.payload_cog)
                )
            # Atomic-enough swap: streaming thread re-reads ``self._ctrl_lock`` and
            # ``self.rtde_ctrl`` each loop iteration.
            self.rtde_ctrl = new_ctrl
            self._ctrl_lock = new_lock
            logger.info("rtde_ctrl + ctrl_lock replaced after wedge")
        except Exception:
            logger.exception(
                "rtde_ctrl reconnect failed; flagging streaming as unhealthy"
            )
            self._stream_failed.set()

    def disconnect(self) -> None:
        # 1. Stop the streaming thread first so it can't race servoStop / stopScript.
        try:
            self.stop_streaming()
        except Exception:
            logger.exception("stop_streaming failed during disconnect")

        # 2. Tear down the control script cleanly. Both calls are watchdogged so a
        # wedged URScript or stale RTDE socket can't pin the process forever.
        if self.rtde_ctrl is not None:
            self._ctrl_call_with_deadline(
                "servoStop", lambda: self.rtde_ctrl.servoStop(10.0), deadline_s=12.0,
            )
            self._ctrl_call_with_deadline(
                "stopScript", lambda: self.rtde_ctrl.stopScript(), deadline_s=5.0,
            )

        # 3. Close the receive / IO handles, gripper, cameras.
        if self.rtde_rec is not None:
            try:
                self.rtde_rec.disconnect()
            except Exception:
                logger.exception("rtde_rec.disconnect failed")
        if self.rtde_io is not None:
            try:
                self.rtde_io.disconnect()
            except Exception:
                logger.exception("rtde_io.disconnect failed")
        if self.gripper is not None:
            self.gripper.close()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self.clear_baselines()
        self._connected = False
        logger.info("UR10e disconnected.")

    # -- telemetry ----------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.rtde_rec.getActualQ(), dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array(self.rtde_rec.getActualQd(), dtype=np.float32)

    def get_current_tcp(self) -> np.ndarray:
        """Current TCP pose [x, y, z, rx, ry, rz] (m, axis-angle radians)."""
        return np.array(self.rtde_rec.getActualTCPPose(), dtype=np.float32)

    def get_tcp_force(self) -> np.ndarray:
        """Current TCP wrench estimate [Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm).

        WARNING — this is NOT a calibrated F/T sensor reading. The UR10e has
        no wrist F/T hardware; `rtde_rec.getActualTCPForce()` returns the
        controller's inverse-dynamics estimate from joint motor currents.
        Per the module docstring at the top of this file:

            "exhibits ±5 N noise plus tens of N of pose-dependent bias drift.
            We deliberately do NOT include it in the observation — the policy
            relies on visual + proprioceptive signals for contact."

        Useful for logging / monitoring / emergency-stop heuristics, but
        don't feed it into a policy without low-pass filtering and per-pose
        bias compensation. The wrench is in the BASE frame (not the tool
        frame); transform via `Rotation.from_rotvec(tcp[3:6])` if you need
        the gripper-frame components.

        Reads the live RTDE state under no lock — `RTDEReceiveInterface` is
        on a separate socket from the streaming thread's
        `RTDEControlInterface`, so this doesn't contend with `servoL`.
        """
        return np.array(self.rtde_rec.getActualTCPForce(), dtype=np.float32)

    def is_protective_stopped(self) -> bool:
        return bool(self.rtde_rec.isProtectiveStopped())

    def is_emergency_stopped(self) -> bool:
        return bool(self.rtde_rec.isEmergencyStopped())

    # -- HIL-SERL relative-observation baselines ---------------------------

    def capture_baselines(self) -> None:
        """Snapshot the current TCP xyz AND wrist orientation as the per-episode anchors.

        Called by `UR10RobotEnv.reset()` (and `auto_reset_to_home`) AFTER the blocking
        `moveL` / interpolated motion has driven the arm to the home pose and the settling
        sleep has elapsed. All subsequent `get_observation()` calls return
        `tcp_xyz - initial_tcp_xyz`. The orientation baseline is consumed by
        `get_measured_yaw_offset()` to produce the relative-yaw observation.

        Reads the live RTDE state under no lock — `RTDEReceiveInterface` is on a
        separate socket from the streaming thread's `RTDEControlInterface`, so reads
        here don't contend with `servoL`. We pay one extra `Rotation.from_rotvec` for
        the orientation baseline regardless of `use_yaw` at the env layer — keeps the
        driver oblivious to the env-side flag.
        """
        tcp_full = self.get_current_tcp()
        self._initial_tcp_xyz = tcp_full[:3].copy()
        self._initial_tcp_rotation = Rotation.from_rotvec(tcp_full[3:6])
        logger.info(
            "Captured per-episode baselines: tcp_xyz=%s, tcp_rotvec=%s",
            np.array2string(self._initial_tcp_xyz, precision=4, suppress_small=True),
            np.array2string(tcp_full[3:6], precision=4, suppress_small=True),
        )

    def clear_baselines(self) -> None:
        """Drop the per-episode baselines so the next `get_observation()` returns absolute
        xyz and `get_measured_yaw_offset()` returns 0.0.

        Called from `disconnect()` and useful in tests / CLI utilities (find-limits)
        that want to inspect raw absolute telemetry.
        """
        self._initial_tcp_xyz = None
        self._initial_tcp_rotation = None

    def get_measured_yaw_offset(self) -> float:
        """Measured wrist yaw offset (radians) about the tool Z axis, relative to the
        per-episode home orientation captured by `capture_baselines()`.

        Mathematically: `R_delta = R_initial.inv() * R_current` (left-multiply by the
        inverse of the home rotation); the rotation about tool Z extracted as the first
        component of `R_delta.as_euler("zyx")`. This is the EXACT inverse of the action
        side's `R_target = R_home * R_z(target_yaw)` composition — so commanded yaw and
        measured yaw stay consistent by construction (sub-mrad agreement on a healthy
        UR10e).

        Returns 0.0 when the baseline is not yet set (e.g. before the first reset, or
        between `clear_baselines()` and the next reset) so observation-space probes at
        env construction time get a coherent value.
        """
        if self._initial_tcp_rotation is None:
            return 0.0
        tcp = self.get_current_tcp()
        R_current = Rotation.from_rotvec(tcp[3:6])
        R_delta = self._initial_tcp_rotation.inv() * R_current
        return float(R_delta.as_euler("zyx")[0])

    # -- observation --------------------------------------------------------

    def get_observation(self) -> dict:
        """Collect a 16-D observation.

        Returns:
            dict with:
                - agent_pos (np.ndarray): a (16,) float32 vector consisting of:
                    - joint_pos (6)    : Current joint positions [rad]            (absolute)
                    - joint_vel (6)    : Current joint velocities [rad/s]         (absolute)
                    - tcp_xyz (3)      : `tcp - initial_tcp_xyz` [m]              (RELATIVE
                                         to per-episode position anchor; HIL-SERL paper)
                    - gripper_state (1): 1.0 if open, 0.0 if closed               (absolute)
                - pixels (dict[str, np.ndarray]): camera-name -> latest (H, W, 3) RGB frame.

        If `capture_baselines()` has not yet been called (e.g. during the construction-
        time observation-space sample, or in CLI utilities that operate without an env
        wrapper), `tcp_xyz` falls back to absolute. The vector SHAPE is invariant —
        (16,) in either case — so observation-space probes match the runtime layout.

        Yaw observation extension: when `UR10RobotEnvConfig.use_yaw=True`, the env layer
        (NOT this driver) augments `agent_pos` to 17-D by inserting the measured tool-Z
        yaw offset (from `get_measured_yaw_offset()`) between `tcp_xyz` and
        `gripper_state` — see `UR10RobotEnv._augment_observation()`. The driver stays
        oblivious to `use_yaw` so CLI utilities reading raw telemetry get the
        canonical 16-D layout.
        """
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        tcp = self.get_current_tcp()
        gripper_state = float(self.gripper.is_open)

        # Position is encoder-derived and clean — apply the per-episode anchor.
        tcp_xyz = tcp[:3]
        if self._initial_tcp_xyz is not None:
            tcp_xyz = tcp_xyz - self._initial_tcp_xyz

        agent_pos = np.concatenate([
            joint_pos,         # 6
            joint_vel,         # 6
            tcp_xyz,           # 3 (relative to initial_tcp_xyz when baseline is set)
            [gripper_state],   # 1
        ]).astype(np.float32)  # total: 16

        pixels = {name: cam.async_read() for name, cam in self.cameras.items()}
        return {"agent_pos": agent_pos, "pixels": pixels}

    # -- streaming-thread API ----------------------------------------------

    def start_streaming(
        self,
        dt: float,
        lookahead_time: float,
        gain: float,
    ) -> None:
        """Spawn the background thread that continuously streams `servoL` to the latest
        target. Initialises the target to the live TCP so the very first command is a no-op
        and the robot does not jump.

        Idempotent: a no-op if streaming is already running.
        """
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        if self.rtde_ctrl is None or not self._connected:
            raise RuntimeError("UR10Robot.connect() must be called before start_streaming()")

        self._stream_dt = float(dt)
        self._stream_lookahead = float(lookahead_time)
        self._stream_gain = float(gain)

        with self._target_lock:
            self._target = list(self.get_current_tcp())

        self._stop_event.clear()
        self._pause_event.clear()
        self._stream_failed.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="UR10Stream",
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(
            "UR10e streaming thread started (dt=%.4fs, lookahead=%.3fs, gain=%.0f).",
            self._stream_dt, self._stream_lookahead, self._stream_gain,
        )

    def stop_streaming(self, timeout: float = 1.0) -> None:
        """Signal the streaming thread to exit and join with a timeout.

        Idempotent: a no-op if streaming was never started or already stopped.
        """
        if self._stream_thread is None:
            return
        self._stop_event.set()
        # Wake a paused thread so it observes the stop event and exits.
        self._pause_event.set()
        self._stream_thread.join(timeout=timeout)
        if self._stream_thread.is_alive():
            logger.warning(
                "UR10e streaming thread did not exit within %.1fs; leaving as daemon.",
                timeout,
            )
        self._stream_thread = None
        # Reset events for any future start_streaming.
        self._stop_event.clear()
        self._pause_event.clear()
        self._stream_failed.clear()

    def set_target_pose(self, pose: list[float]) -> None:
        """Update the streaming-thread's target pose. Non-blocking; sub-microsecond."""
        with self._target_lock:
            self._target = list(pose)

    def streaming_healthy(self) -> bool:
        """True iff the streaming thread is alive and hasn't paused due to an exception.

        Distinguishes "fine, paused by move_to_pose" from "servoL crashed and the robot is
        no longer being commanded". Use as a precondition to env.step() so a silent failure
        doesn't masquerade as a frozen-but-recording episode.
        """
        if self._stream_thread is None or not self._stream_thread.is_alive():
            return False
        if self._stream_failed.is_set():
            return False
        return True

    def _pause_streaming(self) -> None:
        """Pause the streaming thread and exit servoL mode so moveL can run.

        ``servoStop`` is required: on this UR10 firmware ``moveL`` does not implicitly
        leave servoL mode, so without an explicit stop the next ``moveL`` is silently
        dropped and the arm doesn't reset. ``servoStop`` itself is known to wedge
        against this controller after a few cycles; ``_ctrl_call_with_deadline``
        force-resets ``rtde_ctrl`` on hang so we recover without leaking the lock.
        """
        if self._stream_thread is None or not self._stream_thread.is_alive():
            return
        self._pause_event.set()
        self._ctrl_call_with_deadline(
            "servoStop", lambda: self.rtde_ctrl.servoStop(10.0), deadline_s=2.0,
        )

    def _resume_streaming(self) -> None:
        if self._stream_thread is None or not self._stream_thread.is_alive():
            return
        self._pause_event.clear()

    def _stream_loop(self) -> None:
        """Background-thread main loop. Runs until `_stop_event` is set.

        Each iteration:
          1. If paused, block on `_pause_event` until either resume or stop.
          2. Snapshot the latest target under `_target_lock`.
          3. Call `servoL` under `_ctrl_lock`. servoL blocks for `_stream_dt` in C++,
             which self-paces the loop to ~1/dt Hz with no extra sleep.
        """
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Block until pause is cleared or stop is requested. wait() returns when the
                # event is cleared by `_resume_streaming` (Event semantics: wait blocks while
                # the event is *unset*, so we invert: wait on stop_event instead).
                # Simpler: just sleep briefly and re-check.
                self._stop_event.wait(timeout=0.01)
                continue

            with self._target_lock:
                pose = None if self._target is None else list(self._target)

            if pose is None:
                self._stop_event.wait(timeout=self._stream_dt)
                continue

            try:
                with self._ctrl_lock:
                    self.rtde_ctrl.servoL(
                        pose,
                        0.0,
                        0.0,
                        self._stream_dt,
                        self._stream_lookahead,
                        self._stream_gain,
                    )
            except Exception:
                logger.exception("UR10e streaming servoL failed; pausing thread")
                self._stream_failed.set()
                self._pause_event.set()

    # -- motion -------------------------------------------------------------

    def move_to_pose(
        self,
        pose: list[float],
        speed: float,
        acceleration: float,
    ) -> None:
        """Blocking moveL — used only by reset() to drive precisely to the home pose.

        Pauses the streaming thread, executes moveL under `_ctrl_lock`, updates the shared
        target to the new pose so the resumed thread tracks home immediately without
        snapping back to the old target, then resumes streaming.
        """
        streaming = self._stream_thread is not None and self._stream_thread.is_alive()
        if streaming:
            self._pause_streaming()
        try:
            with self._ctrl_lock:
                self.rtde_ctrl.moveL(
                    list(pose), float(speed), float(acceleration), False
                )
            with self._target_lock:
                self._target = list(pose)
        finally:
            if streaming:
                self._resume_streaming()

    def send_gripper(self, command: int) -> None:
        """Send a discrete gripper command (matches RC10 encoding for dataset parity).

        Args:
            command:
                - 0 = close
                - 1 = stay (no-op)
                - 2 = open
        """
        if command == 0:
            self.gripper.send(-1)  # close
        elif command == 2:
            self.gripper.send(1)  # open
        # command == 1 → no-op


# ---------------------------------------------------------------------------
# UR10e Gym environment
# ---------------------------------------------------------------------------


class UR10RobotEnv(gym.Env):
    """Gym environment for the UR10e with Cartesian xyz delta-action streaming.

    Action convention (matches RC10 for dataset parity in the no-yaw case):
        - Shape (3,): [dx, dy, dz] normalised to [-1, 1]                          # use_yaw=F, use_gripper=F
        - Shape (4,): [dx, dy, dz, gripper_cmd] where gripper ∈ {0, 1, 2}         # use_yaw=F, use_gripper=T
        - Shape (4,): [dx, dy, dz, dyaw] (continuous, [-1, 1])                    # use_yaw=T, use_gripper=F
        - Shape (5,): [dx, dy, dz, dyaw, gripper_cmd]                             # use_yaw=T, use_gripper=T

    Gripper, when present, is ALWAYS the last index — keeps SAC's
    `DISCRETE_DIMENSION_INDEX = -1` and `UR10GripperPenaltyProcessorStep`'s
    `action[-1]` indexing valid across yaw modes.

    Wrist orientation: when `use_yaw=False`, held fixed at [fixed_rx, fixed_ry, fixed_rz]
    (axis-angle). When `use_yaw=True`, the env latches `target_yaw=0` at reset and
    accumulates `dyaw * yaw_step_size` each step (clipped to yaw_bounds). The
    streaming target's [rx, ry, rz] is computed by composing the home rotation with a
    rotation about tool Z: `R_target = R_home @ Rotation.from_euler('z', target_yaw)`
    (`Rotation.as_rotvec()` → axis-angle for servoL). This is the mathematically
    correct yaw — simply adding to `fixed_rz` would treat axis-angle components as
    Euler angles, which they are not.

    Lifecycle: the env starts the robot's streaming thread in `__init__` (if connected),
    and the robot's `disconnect()` (called from `close()`) stops the thread cleanly.
    """

    def __init__(self, robot: UR10Robot, config: UR10RobotEnvConfig):
        super().__init__()
        self.robot = robot
        self.config = config

        self.ee_step = np.array([
            config.ee_step_sizes["x"],
            config.ee_step_sizes["y"],
            config.ee_step_sizes["z"],
        ], dtype=np.float32)
        # ee_bounds_min/max may carry a 4th element for the yaw offset window (radians)
        # when use_yaw=True. Slice xyz off the front so the existing clipping code stays
        # 3-D regardless of whether yaw is configured.
        self.ee_min = np.array(config.ee_bounds_min[:3], dtype=np.float32)
        self.ee_max = np.array(config.ee_bounds_max[:3], dtype=np.float32)

        self.use_gripper = config.use_gripper
        self.use_yaw = config.use_yaw

        # Yaw parameters (only consulted when use_yaw=True). Defaults match the
        # InverseKinematicsConfig defaults so a JSON that flips use_yaw=true without
        # filling in the yaw entries still gets sensible behaviour.
        self.yaw_step = float(config.ee_step_sizes.get("yaw", 0.01))
        if len(config.ee_bounds_min) >= 4 and len(config.ee_bounds_max) >= 4:
            self.yaw_min = float(config.ee_bounds_min[3])
            self.yaw_max = float(config.ee_bounds_max[3])
        else:
            self.yaw_min, self.yaw_max = -1.5708, 1.5708

        # Cache the home rotation so per-step composition is a fast multiply rather than
        # a fresh rotvec→Rotation conversion every call.
        self._R_home = Rotation.from_rotvec([
            float(config.fixed_rx), float(config.fixed_ry), float(config.fixed_rz),
        ])

        self.current_step = 0

        # Latched commanded xyz target. Deltas are applied to THIS (the previous command),
        # not to the live measured TCP — otherwise the robot's position lag turns into a
        # receding-carrot effect: held-stick motion feels laggy, and stick-release causes
        # the controller to reverse a few mm because the new target lands behind where it
        # was already planning. Initialised from the live TCP at first reset.
        self.target_xyz: np.ndarray | None = None
        # Latched commanded yaw OFFSET from the home wrist orientation, in radians.
        # 0.0 on reset; clipped to [yaw_min, yaw_max] each step. Only used when use_yaw.
        self.target_yaw: float | None = None

        # -- spaces ---------------------------------------------------------
        # Layout: [xyz (3)] + [yaw (1) if use_yaw] + [gripper (1) if use_gripper].
        # Gripper stays last so SAC's discrete-critic indexing is preserved.
        action_dim = 3 + int(self.use_yaw) + int(self.use_gripper)
        low = [-1.0, -1.0, -1.0]
        high = [1.0, 1.0, 1.0]
        if self.use_yaw:
            low.append(-1.0)
            high.append(1.0)
        if self.use_gripper:
            low.append(0.0)
            high.append(2.0)
        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )

        if self.robot.is_connected:
            # Sample the env's actual observation layout, including the optional yaw slot
            # when use_yaw=True. Sizes the gym space from the augmented shape so policies
            # see the same dim at probe time and at runtime.
            sample = self._augment_observation(self.robot.get_observation())
            obs_spaces: dict[str, gym.spaces.Box] = {
                OBS_STATE: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=sample["agent_pos"].shape,
                    dtype=np.float32,
                ),
            }
            for cam_name, img in sample["pixels"].items():
                obs_spaces[f"{OBS_IMAGES}.{cam_name}"] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=img.shape,
                    dtype=np.uint8,
                )
            self.observation_space = gym.spaces.Dict(obs_spaces)

            # Start the background streaming thread. Idempotent if already running.
            self.robot.start_streaming(
                dt=1.0 / max(1, int(config.stream_frequency_hz)),
                lookahead_time=config.servo_lookahead_time,
                gain=config.servo_gain,
            )

    # -- gym interface ------------------------------------------------------

    def _augment_observation(self, obs: dict) -> dict:
        """Insert the measured yaw offset into `agent_pos` when `use_yaw=True`.

        Layout (use_yaw=True): `[joint_pos(6), joint_vel(6), tcp_xyz_rel(3),
        yaw_offset(1), gripper(1)]` (17-D). Layout (use_yaw=False): unchanged 16-D from
        the robot driver. Gripper stays at index `-1` in BOTH layouts, preserving SAC's
        `DISCRETE_DIMENSION_INDEX = -1` and the gripper-penalty processor's
        `observation.state[-1]` indexing.

        Mutates `obs["agent_pos"]` in place when augmenting; returns `obs` either way
        so the call site can read it inline.
        """
        if not self.use_yaw:
            return obs
        agent_pos = obs["agent_pos"]
        yaw_obs = self.robot.get_measured_yaw_offset()
        # Insert yaw before the gripper slot. `np.concatenate` over three small slices is
        # ~50 ns at this scale — negligible next to the RTDE read inside get_observation.
        obs["agent_pos"] = np.concatenate([
            agent_pos[:-1],
            np.array([yaw_obs], dtype=agent_pos.dtype),
            agent_pos[-1:],
        ])
        return obs

    # ------------------------------------------------------------------------
    # v2 ACT-mode API (absolute target poses, RC10-style)
    # ------------------------------------------------------------------------
    # The HIL-SERL path (step / reset / _augment_observation) above is unchanged.
    # The methods below provide a parallel API for ACT training that mirrors
    # RC10FollowerCut's working setup:
    #   - Observation: 11-D ABSOLUTE state `[joint_pos(6), tcp_x, tcp_y, tcp_z,
    #     tcp_yaw_offset, gripper]` + cameras. No joint velocities, no relative
    #     anchoring on xyz, no per-episode home subtraction.
    #   - Action: 5-D ABSOLUTE target pose `[target_x, target_y, target_z,
    #     target_yaw_offset, gripper_state]` in the env's existing frames
    #     (xyz in base-frame metres, yaw_offset in radians from R_home,
    #     gripper_state ∈ {0.0=closed, 1.0=open}).
    #
    # The v2 recording flow uses the existing `step(delta_action)` to drive the
    # wrist via gamepad — the only thing that changes is what gets stored in the
    # dataset (the env's resulting absolute target, not the sparse bang-bang
    # gamepad delta that caused the v1 ACT collapse-to-zero failure mode).

    def get_act_observation(self) -> dict:
        """Return the v2 ACT-mode observation: 11-D absolute state + cameras.

        Layout of `agent_pos` (all float32, all absolute):
            indices 0..5 : `joint_pos` (rad) — RTDE encoder reading
            index    6   : `tcp_x` (m, base frame)
            index    7   : `tcp_y` (m, base frame)
            index    8   : `tcp_z` (m, base frame)
            index    9   : `tcp_yaw_offset` (rad, relative to R_home — measured
                           via `get_measured_yaw_offset()`)
            index   10   : `gripper` (1.0 if open else 0.0)

        Joint velocities are intentionally omitted — they were a noise source in
        the v1 observation and the v2 policy doesn't need them. xyz is reported
        ABSOLUTE (not anchored to per-episode home) so the policy can directly
        ground its position in the workspace frame, matching RC10FollowerCut's
        proven setup.

        Yaw is reported as an OFFSET from the per-episode home wrist orientation
        for consistency with `set_act_target` (which composes targets the same
        way). Both sides of the loop see yaw in the same frame.

        Returns:
            dict with:
              - `agent_pos`: np.ndarray shape (11,), float32
              - `pixels`: dict[name, np.ndarray] — one (H, W, 3) RGB frame per camera
        """
        tcp = self.robot.get_current_tcp()  # [x, y, z, rx, ry, rz] absolute
        joint_pos = self.robot.get_joint_positions()  # (6,) absolute joint angles
        yaw_offset = self.robot.get_measured_yaw_offset()  # rad, relative to home
        gripper = float(self.robot.gripper.is_open)  # 0.0 (closed) | 1.0 (open)

        agent_pos = np.concatenate([
            joint_pos.astype(np.float32),
            np.array([tcp[0], tcp[1], tcp[2], yaw_offset, gripper], dtype=np.float32),
        ])

        pixels = {name: cam.async_read() for name, cam in self.robot.cameras.items()}
        return {"agent_pos": agent_pos, "pixels": pixels}

    def set_act_target(self, action) -> None:
        """Apply an absolute target pose + gripper command from a v2 ACT action.

        Action layout (5 elements; accepts dict or array-like):
            [target_x, target_y, target_z, target_yaw_offset, gripper_state]
        with dict keys ``{"x.pos", "y.pos", "z.pos", "yaw.pos", "gripper.pos"}``
        (matches RC10FollowerCut's naming convention so `make_robot_action`
        from `lerobot.policies.utils` builds the dict correctly).

        Semantics:
          - xyz are absolute base-frame targets in metres. Clipped to
            ``ee_bounds`` (the same workspace bounds the env's `step` uses).
          - yaw is an offset from the per-episode home orientation in radians,
            clipped to ``yaw_bounds``. The full target rotation is composed as
            ``R_home * R_z(yaw)`` and converted back to axis-angle — exact
            inverse of `get_measured_yaw_offset`, so commanded and observed
            yaw stay consistent by construction.
          - gripper_state in [0, 1]: ``< 0.5`` means "target closed",
            ``>= 0.5`` means "target open". The driver expects a tri-state
            command `{0=close, 1=stay, 2=open}` — we translate via current
            gripper state so we only emit a transition command when the
            desired state differs (avoids redundant motor commands).

        Side effects:
          - Updates ``self.target_xyz`` and ``self.target_yaw`` so anything
            else reading env state (e.g. `auto_reset_to_home`) sees a coherent
            view.
          - Calls ``self.robot.set_target_pose(...)`` (non-blocking; the
            streaming thread picks the new target up on its next 5 ms tick).
          - Calls ``self.robot.send_gripper(cmd)`` when ``use_gripper=True``.
        """
        # Accept both dict and array-like for caller convenience.
        if isinstance(action, dict):
            tx = float(action["x.pos"])
            ty = float(action["y.pos"])
            tz = float(action["z.pos"])
            tyaw = float(action["yaw.pos"])
            tgrip = float(action["gripper.pos"])
        else:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.shape[0] < 5:
                raise ValueError(
                    f"set_act_target expects 5 values (x, y, z, yaw, gripper); got shape {a.shape}"
                )
            tx, ty, tz, tyaw, tgrip = (float(v) for v in a[:5])

        # Clip to workspace bounds — same invariants as the delta-action step().
        tx = float(np.clip(tx, self.ee_min[0], self.ee_max[0]))
        ty = float(np.clip(ty, self.ee_min[1], self.ee_max[1]))
        tz = float(np.clip(tz, self.ee_min[2], self.ee_max[2]))
        if self.use_yaw:
            tyaw = float(np.clip(tyaw, self.yaw_min, self.yaw_max))
        else:
            tyaw = 0.0  # ignore yaw component when yaw is disabled at config level

        # Compose final TCP rotation. R_home is cached at env __init__; composing
        # with R_z(tyaw) gives the correct axis-angle rotvec for servoL.
        if self.use_yaw:
            R_target = self._R_home * Rotation.from_euler("z", tyaw)
            rx, ry, rz = (float(v) for v in R_target.as_rotvec())
        else:
            rx = self.config.fixed_rx
            ry = self.config.fixed_ry
            rz = self.config.fixed_rz

        # Mirror env internal state so HIL-SERL helpers reading these (e.g.
        # `auto_reset_to_home`) see a consistent commanded pose.
        self.target_xyz = np.array([tx, ty, tz], dtype=np.float32)
        if self.use_yaw:
            self.target_yaw = tyaw

        self.robot.set_target_pose([tx, ty, tz, rx, ry, rz])

        if self.use_gripper:
            current_state = float(self.robot.gripper.is_open)
            desired_state = 1.0 if tgrip >= 0.5 else 0.0
            if desired_state == current_state:
                cmd = 1  # STAY
            elif desired_state > current_state:
                cmd = 2  # OPEN  (was closed)
            else:
                cmd = 0  # CLOSE (was open)
            self.robot.send_gripper(cmd)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed, options=options)

        logger.info("Resetting UR10e to home TCP pose ...")
        self.robot.send_gripper(2)  # open gripper

        home = list(self.config.home_tcp)
        rng = self.np_random
        if self.config.randomization_xy > 0:
            r = self.config.randomization_xy
            home[0] += float(rng.uniform(-r, r))
            home[1] += float(rng.uniform(-r, r))
        if self.config.randomization_z > 0:
            r = self.config.randomization_z
            home[2] += float(rng.uniform(-r, r))

        # Clip xyz to workspace bounds so randomisation can't push outside.
        home[0] = float(np.clip(home[0], self.ee_min[0], self.ee_max[0]))
        home[1] = float(np.clip(home[1], self.ee_min[1], self.ee_max[1]))
        home[2] = float(np.clip(home[2], self.ee_min[2], self.ee_max[2]))

        logger.info("  Randomised start: x=%.4f, y=%.4f, z=%.4f", home[0], home[1], home[2])

        # Blocking moveL drives precisely to the home pose. The streaming thread is paused
        # internally for the duration; on return it resumes tracking the (now updated) target.
        self.robot.move_to_pose(
            home,
            speed=self.config.reset_speed,
            acceleration=self.config.reset_acceleration,
        )
        precise_sleep(self.config.reset_time_s)

        # Latch the commanded xyz to the actual reset pose so the first step delta is
        # applied to a coherent setpoint, not to a stale value. The streaming target is
        # always in *absolute* base-frame coordinates — only the policy-facing observation
        # is relative.
        self.target_xyz = np.array(self.robot.get_current_tcp()[:3], dtype=np.float32)
        # Reset the yaw offset to 0 every episode (home orientation = R_home, unchanged).
        # The first step delta accumulates from here and stays inside [yaw_min, yaw_max].
        self.target_yaw = 0.0

        # HIL-SERL relative-observation baseline: anchor `tcp_xyz` to the post-settling
        # reset pose. From this point on `get_observation()` returns `tcp_xyz` expressed
        # relative to this anchor.
        self.robot.capture_baselines()

        self.current_step = 0
        obs = self._augment_observation(self.robot.get_observation())
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        # Safety / liveness: refuse to record a frozen-robot frame masquerading as a real
        # step. Both checks are cheap (RTDEReceiveInterface has its own socket; the health
        # flag is a non-blocking read of two threading.Event objects).
        if self.robot.is_protective_stopped() or self.robot.is_emergency_stopped():
            raise RuntimeError(
                "UR10e is in protective/emergency stop — refusing to record this step. "
                "Release the stop on the teach pendant and restart the run."
            )
        if not self.robot.streaming_healthy():
            raise RuntimeError(
                "UR10e streaming thread is not running (likely crashed in servoL). The "
                "robot is no longer being commanded; aborting before frozen frames are "
                "recorded into the dataset."
            )

        # Action layout (see class docstring):
        #   [dx, dy, dz, (dyaw if use_yaw,) (gripper_cmd if use_gripper)]
        # Gripper is always the LAST element when present — keeps SAC's discrete-critic
        # index and the gripper-penalty processor's `action[-1]` valid in all modes.
        continuous_xyz = np.asarray(action[:3], dtype=np.float32)
        if self.use_yaw:
            dyaw = float(action[3])
        gripper_cmd = (
            int(round(float(action[-1])))
            if self.use_gripper and len(action) >= (3 + int(self.use_yaw) + 1)
            else 1
        )

        # Defensive: if reset() wasn't called yet, latch onto the live TCP / yaw once.
        if self.target_xyz is None:
            self.target_xyz = np.array(self.robot.get_current_tcp()[:3], dtype=np.float32)
        if self.use_yaw and self.target_yaw is None:
            self.target_yaw = 0.0

        # Apply delta to the previously commanded target (NOT the live TCP) and clip.
        delta_xyz = continuous_xyz * self.ee_step
        self.target_xyz = np.clip(self.target_xyz + delta_xyz, self.ee_min, self.ee_max)
        if self.use_yaw:
            self.target_yaw = float(np.clip(
                self.target_yaw + dyaw * self.yaw_step, self.yaw_min, self.yaw_max,
            ))

        # Orientation: when yaw is enabled, compose the home rotation with an additional
        # rotation about *tool* Z (post-multiply). `as_rotvec()` returns the axis-angle
        # 3-vector that servoL consumes. When yaw is disabled, fall through to the
        # legacy fixed orientation — guarantees byte-for-byte identical behaviour for
        # the non-yaw path.
        if self.use_yaw:
            R_target = self._R_home * Rotation.from_euler("z", self.target_yaw)
            rx, ry, rz = (float(v) for v in R_target.as_rotvec())
        else:
            rx = self.config.fixed_rx
            ry = self.config.fixed_ry
            rz = self.config.fixed_rz

        target_pose = [
            float(self.target_xyz[0]),
            float(self.target_xyz[1]),
            float(self.target_xyz[2]),
            rx,
            ry,
            rz,
        ]

        # Non-blocking: just hand the new target to the streaming thread.
        self.robot.set_target_pose(target_pose)
        if self.use_gripper:
            self.robot.send_gripper(gripper_cmd)

        obs = self._augment_observation(self.robot.get_observation())
        self.current_step += 1
        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    def close(self) -> None:
        self.robot.disconnect()
