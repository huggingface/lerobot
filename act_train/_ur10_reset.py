"""Shared between-episode auto-reset for UR10 record / eval scripts.

Both ``record_ur10_act.py`` and ``eval_ur10_act.py`` need to drive the arm
back to home between episodes without going through ``UR10RobotEnv.reset()``
— that path calls ``_pause_streaming`` → ``servoStop`` → ``moveL``, and
``servoStop`` is the well-known intermittent wedge site on this controller.

This helper bypasses ``servoStop`` entirely: it feeds interpolated targets
to the already-running 200 Hz streaming thread via ``set_target_pose``.
The stream is never paused; servoL's lookahead/gain smooth the motion.

Reset has two phases inside the ``reset_time_s`` total budget:
  1. Motion: drive home at ``reset_speed_mps`` (constant velocity). Duration
     = distance / speed.
  2. Hold: park at home for the remainder of ``reset_time_s``. Gives the
     user time to reposition the target object before the next episode.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


def auto_reset_to_home(
    env,
    dt: float,
    reset_time_s: float,
    reset_speed_mps: float,
    fps: int,
) -> None:
    """Drive the arm to (optionally randomized) home WITHOUT calling env.reset().

    Args:
        env: UR10RobotEnv instance.
        dt: Main loop period (= 1 / fps).
        reset_time_s: Total reset budget (motion + hold). Pre-episode pause for
            re-staging the target object happens inside this.
        reset_speed_mps: Linear velocity for the motion phase, m/s. 0.1 m/s
            matches ``UR10RobotEnvConfig.reset_speed`` (the speed env.reset's
            moveL uses).
        fps: Policy / record FPS. Defines the granularity at which interpolated
            targets are handed to the streaming thread.

    Post-conditions (mirror env.reset's invariants so the caller can resume the
    main loop as if env.reset had run):
        - gripper is open
        - env.target_xyz latched to the achieved home
        - env.robot.capture_baselines() has re-anchored relative tcp_xyz
    """
    # Compute the (optionally randomized) home target — same convention as env.reset().
    home = list(env.config.home_tcp[:3])
    if env.config.randomization_xy > 0:
        r = env.config.randomization_xy
        home[0] += float(np.random.uniform(-r, r))
        home[1] += float(np.random.uniform(-r, r))
    if env.config.randomization_z > 0:
        r = env.config.randomization_z
        home[2] += float(np.random.uniform(-r, r))
    home[0] = float(np.clip(home[0], env.ee_min[0], env.ee_max[0]))
    home[1] = float(np.clip(home[1], env.ee_min[1], env.ee_max[1]))
    home[2] = float(np.clip(home[2], env.ee_min[2], env.ee_max[2]))

    # Phase 1 — motion: constant-velocity linear interpolation at reset_speed_mps.
    start_xyz = list(env.robot.get_current_tcp()[:3])
    distance = float(np.linalg.norm(np.array(home) - np.array(start_xyz)))
    motion_duration_s = distance / max(reset_speed_mps, 1e-6)
    motion_duration_s = min(motion_duration_s, reset_time_s)  # never overrun the budget
    n_motion_steps = max(1, int(motion_duration_s * fps))
    motion_start = time.perf_counter()
    for i in range(n_motion_steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / n_motion_steps
        target_pose = [
            start_xyz[0] + alpha * (home[0] - start_xyz[0]),
            start_xyz[1] + alpha * (home[1] - start_xyz[1]),
            start_xyz[2] + alpha * (home[2] - start_xyz[2]),
            env.config.fixed_rx,
            env.config.fixed_ry,
            env.config.fixed_rz,
        ]
        env.robot.set_target_pose(target_pose)
        precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    # Open the gripper (matches env.reset()).
    env.robot.send_gripper(2)

    # Latch the env's commanded xyz to the achieved home so the first delta of the
    # next episode is applied to a coherent setpoint — same invariant as env.reset().
    env.target_xyz = np.array(home, dtype=np.float32)

    # Phase 2 — hold at home for whatever's left of the reset budget. The streaming
    # thread keeps issuing servoL at the held target, so the arm stays put. The user
    # can use this window to reposition the object in the scene.
    remaining_s = reset_time_s - (time.perf_counter() - motion_start)
    if remaining_s > 0:
        logger.info("  reset: holding at home for %.1fs (object setup window)", remaining_s)
        precise_sleep(remaining_s)

    # Re-anchor the relative tcp_xyz baseline so the next episode's observation has
    # the same semantics as if env.reset() had been called. Done AFTER the hold so
    # any small drift during the hold is absorbed into the baseline.
    env.robot.capture_baselines()
