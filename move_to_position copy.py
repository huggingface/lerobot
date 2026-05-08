#!/usr/bin/env python
"""
Interactive end-effector controller for SO-101.
Uses ikpy for FK/IK — pure Python, works on Windows natively.

Install dependency:  pip install ikpy

Commands (at the prompt):
  status / s              - print current EE position
  go <x|y|z> <meters>    - move relative to current position along one axis
  move <x> <y> <z>       - move to absolute position (meters)
  quit / q                - exit

Examples:
  > go z 0.05       # 5 cm upward
  > go x -0.03      # 3 cm in -x
  > move 0.15 0 0.10
"""

import re
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── Configuration ─────────────────────────────────────────────────────────────
PORT            = "COM5"   # change to your port
ROBOT_ID        = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(r"C:\Users\plata\robots")
URDF_PATH       = r"C:\Users\plata\robots\lerobot\calibration\so101_new_calib.urdf"
FPS             = 30

# The 5 revolute joints that control EE position (gripper open/close is separate)
ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]

# Safe workspace bounds in meters
WS_MIN      = np.array([-0.35, -0.35,  0.00])
WS_MAX      = np.array([ 0.35,  0.35,  0.50])
MAX_MOVE_M  = 0.30   # maximum single-move distance (safety clamp)
# ─────────────────────────────────────────────────────────────────────────────


class SO101Kinematics:
    """
    Wraps ikpy for FK and IK on the SO-101 arm.

    Only the 5 arm joints are active; the gripper joint is kept passive so it
    does not interfere with the solver and is handled by the caller.
    """

    def __init__(self, urdf_path: str):
        # Load full chain so we can discover link names and build the mask
        _probe = Chain.from_urdf_file(urdf_path)
        self._link_names = [lnk.name for lnk in _probe.links]

        # Active = only the 5 arm joints; everything else (base, gripper, tip) passive
        mask = [name in ARM_JOINTS for name in self._link_names]

        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx = [self._link_names.index(n) for n in ARM_JOINTS]

        print("Kinematic chain:")
        for i, name in enumerate(self._link_names):
            tag = " ← active" if mask[i] else ""
            print(f"  [{i:2d}] {name}{tag}")
        print()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _to_ikpy(self, arm_deg: np.ndarray) -> np.ndarray:
        """5 arm joint degrees → full ikpy angle vector (radians, zeros elsewhere)."""
        q = np.zeros(len(self._link_names))
        for ikpy_i, deg in zip(self._arm_idx, arm_deg):
            q[ikpy_i] = np.deg2rad(deg)
        return q

    def _from_ikpy(self, q: np.ndarray) -> np.ndarray:
        """Full ikpy angle vector → 5 arm joint degrees."""
        return np.array([np.rad2deg(q[i]) for i in self._arm_idx])

    def _clip_to_bounds(self, q: np.ndarray) -> np.ndarray:
        """Clip full ikpy angle vector to each joint's URDF limits."""
        q = q.copy()
        for i, link in enumerate(self.chain.links):
            if link.bounds is not None:
                lo, hi = link.bounds
                if lo is not None:
                    q[i] = max(q[i], lo)
                if hi is not None:
                    q[i] = min(q[i], hi)
        return q

    # ── public API ────────────────────────────────────────────────────────────

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        joint_pos_deg: 6-element array (5 arm joints + gripper) in degrees.
        Returns 4×4 world-to-EE transformation matrix.
        """
        return self.chain.forward_kinematics(self._to_ikpy(joint_pos_deg[:5]))

    def inverse_kinematics(
        self, current_deg: np.ndarray, target_T: np.ndarray
    ) -> np.ndarray:
        """
        current_deg: 6-element array (used as IK warm-start and for gripper passthrough).
        target_T:    4×4 desired EE pose.
        Returns new 6-element degree array (gripper value preserved from current_deg).
        """
        # Clip initial guess to URDF joint limits — motor degree values can
        # sit just outside the limits due to calibration tolerances.
        initial = self._clip_to_bounds(self._to_ikpy(current_deg[:5]))

        result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            initial_position=initial,
        )
        out = np.array(current_deg, dtype=float)
        out[:5] = self._from_ikpy(result)
        return out


# ── Helpers ───────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def _parse_num(token: str) -> float:
    """Extract a float from a token like 'x=+0.24m' or '-0.0135'."""
    m = _NUM_RE.search(token)
    if m is None:
        raise ValueError(token)
    return float(m.group())


def _joints_from_obs(obs: dict) -> np.ndarray:
    return np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)


def _print_ee(T: np.ndarray, prefix: str = ""):
    p = T[:3, 3]
    print(f"{prefix}EE → x={p[0]:+.4f}m  y={p[1]:+.4f}m  z={p[2]:+.4f}m")


def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target, WS_MIN, WS_MAX)
    dist = float(np.linalg.norm(target - current))
    if dist > MAX_MOVE_M:
        target = current + (target - current) * (MAX_MOVE_M / dist)
        print(f"  [safety] clamped to {MAX_MOVE_M:.2f}m from start")
    return target


# ── Motion ────────────────────────────────────────────────────────────────────

def smooth_move(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    speed_m_per_s: float = 0.10,
    stop_thresh_m: float = 0.004,
):
    """
    Closed-loop proportional controller.

    Each cycle:
      1. Read the actual robot joint positions.
      2. Compute current EE position via FK.
      3. Step toward target proportionally, capped at (speed / FPS) per cycle.
      4. Run IK from actual joints (warm-start) with orientation preserved.
      5. Send command.

    Because we always re-read the real robot state, motor-tracking lag and small
    IK residuals are corrected every cycle instead of accumulating.
    """
    obs = robot.get_observation()
    q = _joints_from_obs(obs)
    T_start = kin.forward_kinematics(q)
    p_start = T_start[:3, 3].copy()
    gripper_pos = obs["gripper.pos"]

    target_pos = _clamp_target(target_pos, p_start)
    total_dist = float(np.linalg.norm(target_pos - p_start))
    if total_dist < stop_thresh_m:
        print("  Already at target.")
        return

    # Build full target pose: position = target, orientation = start (preserved)
    T_target = T_start.copy()
    T_target[:3, 3] = target_pos

    max_step = speed_m_per_s / FPS          # max meters to move per cycle
    max_iters = int(total_dist / max_step * 4) + 60   # generous timeout
    print(f"  Moving {total_dist * 100:.1f} cm …")

    for _ in range(max_iters):
        t0 = time.perf_counter()

        # Re-read actual robot state every cycle — this is the key difference
        obs = robot.get_observation()
        q = _joints_from_obs(obs)
        T_curr = kin.forward_kinematics(q)
        p_curr = T_curr[:3, 3]

        error = target_pos - p_curr
        err_dist = float(np.linalg.norm(error))
        if err_dist < stop_thresh_m:
            break

        # P step: move a fraction of the error, capped at max_step
        step = error * min(1.0, max_step / err_dist)

        # Waypoint pose: step in position, preserve start orientation
        T_wp = T_target.copy()
        T_wp[:3, 3] = p_curr + step

        # IK warm-started from actual current joints
        q = kin.inverse_kinematics(q, T_wp)

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)

        time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    obs = robot.get_observation()
    _print_ee(kin.forward_kinematics(_joints_from_obs(obs)), prefix="  done → ")


# ── REPL ──────────────────────────────────────────────────────────────────────

HELP = """\
  status / s              print current EE position
  go <x|y|z> <meters>    relative move along one axis
  move <x> <y> <z>       absolute move to position (meters)
  quit / q                disconnect and exit"""


def run_repl(robot: SO101Follower, kin: SO101Kinematics):
    print("SO-101 EE Controller  —  type 'help' for commands\n")

    while True:
        obs = robot.get_observation()
        q = _joints_from_obs(obs)
        T = kin.forward_kinematics(q)
        _print_ee(T)

        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        tokens = raw.lower().split()
        cmd = tokens[0]

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd in ("s", "status"):
            continue

        elif cmd == "help":
            print(HELP)

        elif cmd == "go":
            if len(tokens) != 3:
                print("  Usage: go <x|y|z> <delta_meters>")
                continue
            axis_map = {"x": 0, "y": 1, "z": 2}
            if tokens[1] not in axis_map:
                print("  Axis must be x, y, or z")
                continue
            try:
                delta = _parse_num(tokens[2])
            except ValueError:
                print("  Delta must be a number")
                continue

            target = T[:3, 3].copy()
            target[axis_map[tokens[1]]] += delta
            smooth_move(robot, kin, target)

        elif cmd == "move":
            # Accept plain numbers, labels (x=0.1), or copy-pasted EE output (x=+0.24m)
            nums = []
            for t in tokens[1:]:
                try:
                    nums.append(_parse_num(t))
                except ValueError:
                    pass
            if len(nums) != 3:
                print("  Usage: move <x> <y> <z>  (e.g. move 0.15 0 0.10)")
                continue

            target = np.array(nums)
            smooth_move(robot, kin, target)

        else:
            print(f"  Unknown command '{cmd}'. Type 'help'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    kin = SO101Kinematics(URDF_PATH)   # prints chain on startup, no robot needed yet

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = SO101Follower(config)
    robot.connect()

    try:
        run_repl(robot, kin)
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
