#!/usr/bin/env python
"""lerobot-rollout with gold-standard MuJoCo EE retargeting at the robot boundary.

The policy was trained on the LONG arm (upper arm +5 cm). The real robot is the
SHORT (stock) arm. We bridge the morphology gap with the *exact* MuJoCo FK/IK that
produced the validated cyan/red overlay video, wrapping the raw robot so that:

  * get_observation():  SHORT joints --FK(short)--> gripper-tip pose --IK(long)-->
                        LONG joints   (the state the policy expects)
  * send_action():      LONG joint targets --FK(long)--> pose --IK(short)-->
                        SHORT joint targets  (clamped to the real per-arm limits)

Between those two boundaries everything (state, relative-action anchor, policy
output) lives consistently in LONG space, so no other part of the rollout stack
needs to change. We inject the wrapper by patching
``lerobot.rollout.context.make_robot_from_config`` and then hand off to the normal
``lerobot-rollout`` entry point, so every CLI flag behaves identically.

Usage: same args as ``lerobot-rollout``, e.g.

    python examples/openarm/rollout_retarget.py \
        --policy.path=/home/yope/Documents/sonic/data/folding_latest \
        --robot.type=bi_openarm_follower --robot.id=openarms \
        --robot.cameras='{ ... }' \
        --robot.left_arm_config.port=can1 ... --task="Fold the T-shirt properly" \
        --fps=30 --duration=2000 --device=cuda --display_data=true

Env toggles (optional):
    RETARGET_OBS=0        disable observation retargeting (short->long)
    RETARGET_ACT=0        disable action retargeting (long->short)
    RETARGET_ITERS=25     IK iterations per tick (warm-started)
    RETARGET_ITERS0=80    IK iterations on the very first tick (cold seed)
    RETARGET_NULL_GAIN=0.3   nullspace bias: pull short joints toward long joints (0 = off,
                             EE-only). Higher keeps the elbow closer to the long pose.

  Joint-space smoothing streamer (decouples motor rate from the slow control loop):
    STREAM=1              enable the background smoothing streamer
    STREAM_HZ=60          motor command rate of the streamer thread (Hz)
    STREAM_SMOOTH_TIME=0.10   SmoothDamp time constant (s); larger = smoother/laggier
    STREAM_MAX_SPEED=150  per-joint speed cap (deg/s); 0 disables the cap
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

logger = logging.getLogger("rollout_retarget")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the *validated* geometry + FK/IK helpers (same code that made the overlay video).
ov = _load("ov", os.path.join(_REPO, ".overlay_rest.py"))
rt = _load("rt", os.path.join(_REPO, ".roundtrip_overlay.py"))

from lerobot.robots.openarm_follower.config_openarm_follower import (  # noqa: E402
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
)

SIDES = ("right", "left")
LIMITS = {"right": RIGHT_DEFAULT_JOINTS_LIMITS, "left": LEFT_DEFAULT_JOINTS_LIMITS}

# POLICY_ORDER index of each gripper in the 16-vector (right block 0..7, left block 8..15).
GRIPPER_IDX = {"right": 7, "left": 15}


def smooth_damp(
    current: np.ndarray,
    target: np.ndarray,
    velocity: np.ndarray,
    smooth_time: float,
    dt: float,
    max_speed: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Critically-damped 2nd-order smoothing toward ``target`` (vectorized SmoothDamp).

    Gives C1-continuous position + velocity with no overshoot, and re-plans every tick,
    so it degrades gracefully when targets arrive irregularly or late. Returns the new
    (position, velocity). ``max_speed <= 0`` disables the per-joint speed cap.
    """
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    original_to = target.copy()
    if max_speed and max_speed > 0.0:
        max_change = max_speed * smooth_time
        change = np.clip(change, -max_change, max_change)
    shifted_target = current - change
    temp = (velocity + omega * change) * dt
    velocity = (velocity - omega * temp) * exp
    output = shifted_target + (change + temp) * exp
    # Kill overshoot: if we crossed the original target, snap to it and match velocity.
    overshoot = (original_to - current > 0.0) == (output > original_to)
    output = np.where(overshoot, original_to, output)
    velocity = np.where(overshoot, (output - original_to) / dt, velocity)
    return output, velocity


class RetargetRobot:
    """Boundary wrapper that retargets state (short->long) and actions (long->short).

    All non-overridden attributes/methods proxy transparently to the wrapped robot
    (``connect``, ``disconnect``, ``observation_features``, ``cameras``, ...), so the
    rollout stack treats it exactly like the underlying ``bi_openarm_follower``.
    """

    def __init__(
        self,
        robot,
        obs_iters: int = 25,
        act_iters: int = 25,
        iters0: int = 80,
        retarget_obs: bool = True,
        retarget_act: bool = True,
        null_gain: float = 0.3,
        stream: bool = False,
        stream_hz: float = 60.0,
        smooth_time: float = 0.10,
        max_speed: float = 150.0,
    ) -> None:
        self._robot = robot
        self._lock = threading.Lock()
        # Serialises *all* real-robot bus I/O (obs reads + streamer writes) so the
        # observation thread and the streamer thread never touch the CAN bus at once.
        self._io_lock = threading.Lock()
        self._obs_iters = obs_iters
        self._act_iters = act_iters
        self._iters0 = iters0
        self._do_obs = retarget_obs
        self._do_act = retarget_act
        # Nullspace secondary-task gain: pull the redundant DOF toward the reference
        # (other-arm) joints so the two arms match in joint space without moving the EE.
        self._null_gain = null_gain
        # EE-priority guard for the nullspace bias (metres of 6-vector residual norm).
        self._ee_tol = 0.01  # only consider a fallback when biased residual exceeds this
        self._ee_slack = 0.005  # ...and EE-only beats it by more than this
        # Task-only iterations appended after the biased iterations (re-tightens EE a bit
        # without fully washing out the elbow bias; the fallback is the real EE guarantee).
        self._final_task_iters = int(os.environ.get("RETARGET_FINAL_TASK_ITERS", "3"))

        # --- joint-space smoothing streamer ---
        self._stream = stream
        self._stream_hz = stream_hz
        self._smooth_time = smooth_time
        self._max_speed = max_speed
        self._goal_lock = threading.Lock()
        self._goal: np.ndarray | None = None  # latest short target (16-vec, deg), POLICY order
        self._stream_current: np.ndarray | None = None  # smoothed setpoint (16-vec, deg)
        self._stream_vel = np.zeros(16)
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        # Full last real short pose (16-vec, deg) incl grippers, for streamer seeding.
        self._last_short_full: np.ndarray | None = None
        # Present short pose captured at connect() -> exact home to return to on shutdown.
        self._home: np.ndarray | None = None

        m_short = mujoco.MjModel.from_xml_path(ov.MJCF)
        m_long, _ = ov.make_long(m_short)
        self.m_short, self.m_long = m_short, m_long
        # Separate MjData per direction so obs- and action-side solves never share buffers
        # (the ThreadSafeRobot lock already serialises calls, but this is belt-and-braces).
        self.d_short_o = mujoco.MjData(m_short)  # obs: FK on short
        self.d_long_o = mujoco.MjData(m_long)  # obs: IK on long
        self.d_long_a = mujoco.MjData(m_long)  # action: FK on long
        self.d_short_a = mujoco.MjData(m_short)  # action: IK on short

        self.qadr_s = rt.joint_adr(m_short)
        self.qadr_l = rt.joint_adr(m_long)
        self.tcp_s = {s: mujoco.mj_name2id(m_short, mujoco.mjtObj.mjOBJ_BODY, rt.TCP[s]) for s in SIDES}
        self.tcp_l = {s: mujoco.mj_name2id(m_long, mujoco.mjtObj.mjOBJ_BODY, rt.TCP[s]) for s in SIDES}
        self.dofs_s = {s: rt.arm_dofs(m_short, s) for s in SIDES}  # (qposadr, dofadr, range)
        self.dofs_l = {s: rt.arm_dofs(m_long, s) for s in SIDES}
        # Real per-arm limits (radians) so retargeted commands are executable without clipping.
        self.rng_real = {
            s: np.deg2rad(np.array([LIMITS[s][f"joint_{i}"] for i in range(1, 8)], float)) for s in SIDES
        }

        self._obs_seed: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        self._act_seed: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        # Last *real* short-arm joints (rad) seen in get_observation. Used to seed the
        # first action-side IK so the initial command lands in the null-space branch
        # nearest the robot's actual pose (the 7-DOF arm is redundant, so the same EE
        # admits many joint configs; without this the first tick could command a large
        # elbow-swivel reconfiguration toward an arbitrary branch).
        self._last_short: dict[str, np.ndarray] = {}
        self._first_obs = True
        self._first_act = True

        if self._stream:
            logger.info(
                "Smoothing streamer ENABLED (hz=%.0f, smooth_time=%.3fs, max_speed=%.0f deg/s)",
                self._stream_hz,
                self._smooth_time,
                self._max_speed,
            )
        logger.info(
            "RetargetRobot ready (obs=%s, act=%s, iters=%d, iters0=%d)",
            self._do_obs,
            self._do_act,
            self._obs_iters,
            self._iters0,
        )

    # -- transparent proxy for everything else (connect/disconnect/features/...) --
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_robot"), name)

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _read_arm_deg(d: dict, side: str) -> np.ndarray:
        return np.array([d[f"{side}_joint_{i}.pos"] for i in range(1, 8)], float)

    def _has_full_arms(self, d: dict) -> bool:
        return all(f"{s}_joint_{i}.pos" in d for s in SIDES for i in range(1, 8))

    def _ik(
        self,
        m,
        d,
        tcp_id: int,
        dofadr: np.ndarray,
        qadr7: np.ndarray,
        rng7: np.ndarray,
        pt: np.ndarray,
        Rt: np.ndarray,
        seed: np.ndarray,
        q_ref: np.ndarray,
        iters: int,
        use_null: bool = True,
        lam: float = 0.06,
        step: float = 0.25,
        null_step: float = 0.08,
    ) -> tuple[np.ndarray, float]:
        """Damped least-squares IK for the 6-DOF EE pose with a nullspace bias toward ``q_ref``.

        Primary task: reach (pt, Rt). Secondary task (projected into the task nullspace so it
        never disturbs the EE): minimise ||q - q_ref||, using the 7-DOF arm's 1 redundant DOF
        to keep the elbow close to the reference (long/short) configuration.
        """
        q = seed.copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        eye6 = np.eye(6)
        eye7 = np.eye(7)
        last = 1e9
        for it in range(iters):
            d.qpos[qadr7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)  # required by mj_jac
            p = d.xpos[tcp_id].copy()
            R = d.xmat[tcp_id].reshape(3, 3)
            e = np.concatenate([pt - p, rt.rot_err(R, Rt)])
            last = float(np.linalg.norm(e))
            mujoco.mj_jac(m, d, jacp, jacr, p, tcp_id)
            J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])  # 6x7
            Jt = J.T
            # Task step: DAMPED pinv for stability near singularities. Clipped on its own so
            # the primary always gets its full authority.
            dq = np.clip(Jt @ np.linalg.solve(J @ Jt + (lam**2) * eye6, e), -step, step)
            # Apply the nullspace bias only in the early iterations; the last
            # ``final_task_iters`` are task-only so the EE is always re-tightened from the
            # biased configuration (strict EE priority even if q_ref is far/infeasible).
            null_active = use_null and self._null_gain > 0.0 and (iters - it) > self._final_task_iters
            if null_active:
                # Nullspace projector from the TRUE pinv so J @ N == 0 exactly: the secondary
                # (elbow-toward-q_ref) task lives purely in the redundant DOF and never moves
                # the EE. Clipped small so it stays strictly secondary to the task step.
                Jpinv_true = np.linalg.pinv(J)  # 7x6
                nullproj = eye7 - Jpinv_true @ J  # 7x7
                dq_null = np.clip(nullproj @ (self._null_gain * (q_ref - q)), -null_step, null_step)
                dq = dq + dq_null
            q = q + dq
            q = np.clip(q, rng7[:, 0], rng7[:, 1])
        return q, last

    def _solve(self, m, d, tcp_id, dofadr, qadr7, rng7, pt, Rt, seed, q_ref, iters):
        """Nullspace-biased IK with strict EE priority.

        Solve with the elbow-toward-``q_ref`` bias; if that leaves an EE residual that a
        pure EE-only solve would beat by more than ``_ee_slack``, fall back to EE-only. So the
        bias is applied only when it is (nearly) free in EE terms — a weird/infeasible target
        can never trade away end-effector accuracy for elbow matching.
        """
        q, res = self._ik(m, d, tcp_id, dofadr, qadr7, rng7, pt, Rt, seed, q_ref, iters)
        if self._null_gain > 0.0 and res > self._ee_tol:
            q0, res0 = self._ik(
                m, d, tcp_id, dofadr, qadr7, rng7, pt, Rt, seed, q_ref, iters, use_null=False
            )
            if res0 + self._ee_slack < res:
                return q0, res0
        return q, res

    # -- observation: SHORT joints -> LONG joints (FK short, IK long) ------------
    def get_observation(self) -> dict:
        with self._io_lock:
            obs = self._robot.get_observation()
        # Record the full real short pose (arms + grippers) for streamer seeding.
        if self._has_full_arms(obs):
            full = np.zeros(16)
            for s in SIDES:
                full[rt.ARM_JOINT_SLICES[s]] = self._read_arm_deg(obs, s)
                gk = f"{s}_gripper.pos"
                if gk in obs:
                    full[GRIPPER_IDX[s]] = float(obs[gk])
            self._last_short_full = full
        if not self._do_obs or not self._has_full_arms(obs):
            return obs
        with self._lock:
            state = np.zeros(16)
            for s in SIDES:
                arm = self._read_arm_deg(obs, s)
                state[rt.ARM_JOINT_SLICES[s]] = arm
                self._last_short[s] = np.deg2rad(arm)
            rt.set_arms(self.m_short, self.d_short_o, self.qadr_s, state)
            mujoco.mj_forward(self.m_short, self.d_short_o)
            out = dict(obs)
            for s in SIDES:
                pt = self.d_short_o.xpos[self.tcp_s[s]].copy()
                Rt = self.d_short_o.xmat[self.tcp_s[s]].reshape(3, 3)
                q_ref = np.deg2rad(state[rt.ARM_JOINT_SLICES[s]])  # bias long pose toward short obs
                seed = self._obs_seed[s]
                if seed is None:
                    seed = q_ref.copy()
                iters = self._iters0 if self._first_obs else self._obs_iters
                q7, _ = self._solve(
                    self.m_long,
                    self.d_long_o,
                    self.tcp_l[s],
                    self.dofs_l[s][1],
                    self.dofs_l[s][0],
                    self.dofs_l[s][2],  # clamp to LONG model ranges (policy's training space)
                    pt,
                    Rt,
                    seed,
                    q_ref,
                    iters,
                )
                self._obs_seed[s] = q7
                deg = np.rad2deg(q7)
                for i in range(1, 8):
                    out[f"{s}_joint_{i}.pos"] = float(deg[i - 1])
            self._first_obs = False
            return out

    # -- action: LONG joint targets -> SHORT joint targets (FK long, IK short) ---
    def _retarget_action(self, action: dict) -> dict:
        """Return the short-arm action dict (IK-retargeted if enabled, else a copy)."""
        if not self._do_act or not isinstance(action, dict) or not self._has_full_arms(action):
            return action
        with self._lock:
            long_q = np.zeros(16)
            for s in SIDES:
                long_q[rt.ARM_JOINT_SLICES[s]] = self._read_arm_deg(action, s)
            rt.set_arms(self.m_long, self.d_long_a, self.qadr_l, long_q)
            mujoco.mj_forward(self.m_long, self.d_long_a)
            out = dict(action)
            for s in SIDES:
                pt = self.d_long_a.xpos[self.tcp_l[s]].copy()
                Rt = self.d_long_a.xmat[self.tcp_l[s]].reshape(3, 3)
                q_ref = np.deg2rad(long_q[rt.ARM_JOINT_SLICES[s]])  # bias short pose toward long target
                seed = self._act_seed[s]
                if seed is None:
                    # Prefer the robot's real current short pose (nearest branch, minimal
                    # startup motion); fall back to the long target angles if unseen.
                    seed = (
                        self._last_short[s].copy()
                        if s in self._last_short
                        else q_ref.copy()
                    )
                iters = self._iters0 if self._first_act else self._act_iters
                q7, _ = self._solve(
                    self.m_short,
                    self.d_short_a,
                    self.tcp_s[s],
                    self.dofs_s[s][1],
                    self.dofs_s[s][0],
                    self.rng_real[s],  # clamp to REAL limits -> safe on hardware
                    pt,
                    Rt,
                    seed,
                    q_ref,
                    iters,
                )
                self._act_seed[s] = q7
                deg = np.rad2deg(q7)
                for i in range(1, 8):
                    out[f"{s}_joint_{i}.pos"] = float(deg[i - 1])
            self._first_act = False
            return out

    def send_action(self, action: dict):
        short = self._retarget_action(action)
        # Streaming path: just publish the target; the streamer thread writes to the bus.
        if self._stream and isinstance(short, dict) and self._has_full_arms(short):
            self._set_goal(short)
            return short
        with self._io_lock:
            return self._robot.send_action(short)

    # -- streamer: joint-space smoothing at a fixed high rate ---------------------
    @staticmethod
    def _dict_to_vec(d: dict, fallback: np.ndarray | None = None) -> np.ndarray:
        vec = np.zeros(16) if fallback is None else fallback.copy()
        for s in SIDES:
            base = rt.ARM_JOINT_SLICES[s].start
            for i in range(1, 8):
                vec[base + i - 1] = float(d[f"{s}_joint_{i}.pos"])
            gk = f"{s}_gripper.pos"
            if gk in d:
                vec[GRIPPER_IDX[s]] = float(d[gk])
        return vec

    @staticmethod
    def _vec_to_action(vec: np.ndarray) -> dict:
        out = {}
        for s in SIDES:
            base = rt.ARM_JOINT_SLICES[s].start
            for i in range(1, 8):
                out[f"{s}_joint_{i}.pos"] = float(vec[base + i - 1])
            out[f"{s}_gripper.pos"] = float(vec[GRIPPER_IDX[s]])
        return out

    def _set_goal(self, short: dict) -> None:
        with self._goal_lock:
            base = self._goal if self._goal is not None else self._last_short_full
            self._goal = self._dict_to_vec(short, fallback=base)

    def _seed_current(self) -> None:
        """Seed the streamer setpoint from the robot's actual present pose (ramp start)."""
        try:
            with self._io_lock:
                obs = self._robot.get_observation()
            if self._has_full_arms(obs):
                self._stream_current = self._dict_to_vec(obs)
                self._stream_vel = np.zeros(16)
                if self._home is None:
                    self._home = self._stream_current.copy()
                logger.info("Streamer seeded from present robot pose")
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("Streamer seed failed (%s); will seed from first goal", e)
        self._stream_current = None

    def _stream_loop(self) -> None:
        dt = 1.0 / self._stream_hz
        pos_eps = 0.05  # deg: below this distance to goal we consider the axis settled
        vel_eps = 0.5  # deg/s: below this speed we consider motion stopped
        while not self._stream_stop.is_set():
            t0 = time.perf_counter()
            with self._goal_lock:
                goal = None if self._goal is None else self._goal.copy()
            if goal is None:
                time.sleep(dt)
                continue
            if self._stream_current is None:
                self._stream_current = goal.copy()
                self._stream_vel = np.zeros(16)
            self._stream_current, self._stream_vel = smooth_damp(
                self._stream_current, goal, self._stream_vel, self._smooth_time, dt, self._max_speed
            )
            # Damiao MIT mode holds the last command, so when we've converged on the
            # goal and stopped moving we skip the CAN write entirely. This keeps the
            # bus from saturating (Errno 105) during the pauses when the policy goal
            # is constant, and frees the _io_lock so the main loop's reads stay fast.
            settled = (
                float(np.max(np.abs(self._stream_current - goal))) < pos_eps
                and float(np.max(np.abs(self._stream_vel))) < vel_eps
            )
            if not settled:
                act = self._vec_to_action(self._stream_current)
                try:
                    with self._io_lock:
                        self._robot.send_action(act)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Streamer send_action failed: %s", e)
            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # -- lifecycle (start/stop the streamer around the real connect/disconnect) ---
    def connect(self, *args, **kwargs):
        result = self._robot.connect(*args, **kwargs)
        if self._stream and self._stream_thread is None:
            self._seed_current()
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_loop, name="RetargetStreamer", daemon=True
            )
            self._stream_thread.start()
            logger.info("Smoothing streamer thread started")
        return result

    def _drain_home(self, timeout_s: float = 6.0, tol_deg: float = 0.7) -> None:
        """Command the captured home pose and keep the streamer running until it
        actually converges (or times out), so the arms fully reach home before we
        cut the streamer thread. Only meaningful when streaming is enabled."""
        if self._home is None or self._stream_thread is None:
            return
        logger.info("Returning arms to home pose (draining streamer)...")
        with self._goal_lock:
            self._goal = self._home.copy()
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < timeout_s:
            cur = self._stream_current
            if cur is not None and float(np.max(np.abs(cur - self._home))) < tol_deg:
                logger.info("Home pose reached")
                return
            time.sleep(0.05)
        logger.warning("Home drain timed out after %.1fs; stopping streamer anyway", timeout_s)

    def disconnect(self, *args, **kwargs):
        if self._stream_thread is not None:
            try:
                self._drain_home()
            except Exception as e:  # noqa: BLE001
                logger.warning("Home drain failed: %s", e)
            self._stream_stop.set()
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
            logger.info("Smoothing streamer thread stopped")
        return self._robot.disconnect(*args, **kwargs)


def _wrap_factory(orig):
    obs_iters = int(os.environ.get("RETARGET_ITERS", "25"))
    act_iters = int(os.environ.get("RETARGET_ITERS", "25"))
    iters0 = int(os.environ.get("RETARGET_ITERS0", "80"))
    do_obs = os.environ.get("RETARGET_OBS", "1") != "0"
    do_act = os.environ.get("RETARGET_ACT", "1") != "0"
    null_gain = float(os.environ.get("RETARGET_NULL_GAIN", "0.3"))
    stream = os.environ.get("STREAM", "0") != "0"
    stream_hz = float(os.environ.get("STREAM_HZ", "40"))
    smooth_time = float(os.environ.get("STREAM_SMOOTH_TIME", "0.10"))
    max_speed = float(os.environ.get("STREAM_MAX_SPEED", "150"))

    def factory(cfg):
        real = orig(cfg)
        logger.info("Wrapping %s with RetargetRobot (MuJoCo long<->short EE retarget)", type(real).__name__)
        return RetargetRobot(
            real,
            obs_iters=obs_iters,
            act_iters=act_iters,
            iters0=iters0,
            retarget_obs=do_obs,
            retarget_act=do_act,
            null_gain=null_gain,
            stream=stream,
            stream_hz=stream_hz,
            smooth_time=smooth_time,
            max_speed=max_speed,
        )

    return factory


def _patch_rtc_realtime():
    """Make RTC re-anchor new chunks on the *actual* number of actions consumed
    during inference instead of the fps-based estimate.

    The stock ActionQueue computes ``real_delay = ceil(latency * fps)`` and discards
    that many actions from every new chunk. When the control loop runs below --fps
    (e.g. 20 Hz while fps=30), real_delay (18) overshoots the actions that were truly
    consumed (indexes_diff=12), so it skips ~6 extra actions per inference and the
    trajectory plays faster than real time. ``indexes_diff = last_index - idx_before``
    is the ground-truth count of what the robot actually executed during inference, so
    skipping exactly that many gives real-time playback independent of the loop rate.
    """
    try:
        from lerobot.policies.rtc.action_queue import ActionQueue
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not patch RTC action queue for real-time playback: %s", e)
        return

    def _resolve(self, real_delay, action_index_before_inference=None):
        if action_index_before_inference is not None:
            return max(0, self.last_index - action_index_before_inference)
        return max(0, real_delay)

    ActionQueue._check_and_resolve_delays = _resolve
    logger.info("Patched RTC ActionQueue: re-anchor on real consumed index (no over-skip)")


def main():
    import lerobot.rollout.context as context

    context.make_robot_from_config = _wrap_factory(context.make_robot_from_config)
    _patch_rtc_realtime()

    from lerobot.scripts.lerobot_rollout import main as rollout_main

    rollout_main()


if __name__ == "__main__":
    main()
