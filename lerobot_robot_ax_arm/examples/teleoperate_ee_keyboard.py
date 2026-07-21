#!/usr/bin/env python

"""Keyboard end-effector teleoperation for the 4-DoF AX arm (velocity IK, position output).

Adapted from a resolved-rate (twist) servoing controller: instead of commanding joint
velocities, the per-tick joint velocity is integrated into a joint *position* target and sent as
``Goal_Position`` (the AX arm runs in position mode).

Each frame we read the raw motor ticks, map them to URDF joint angles, solve for a joint velocity
that produces the requested Cartesian motion, scale it to a fixed end-effector Cartesian step
``CART_STEP_M`` per tick (capped in joint space near singularities), and command ``Goal_Position``.

Two IK solvers are selectable at runtime (``--solver`` / ``m`` key):
  - "dq"  : dual-quaternion resolved-rate (matches the full pose velocity via ``scipy.least_squares``;
            faithful to the source controller, but rotation/translation coupling on a 3-DoF arm
            causes axis leakage),
  - "pos" : position-only Jacobian ``dEE_pos/dq`` solved with least-squares (crisp axis-aligned
            Cartesian motion).

The motion frame is also selectable (``--frame`` / ``t`` key): "local" moves along the
end-effector's own axes, "global" along the fixed world axes. Global always uses the position
Jacobian (the dq solver's held-orientation constraint leaks axes on this underactuated arm).

The tick<->URDF mapping is established once by ``lerobot-calibrate`` (reference pose + travel
limits), so no separate alignment step is needed here.

Controls (letter keys; hold to keep moving via terminal key-repeat):
    - w / s : +X / -X   (forward / back)
    - a / d : +Y / -Y   (left / right)
    - r / f : +Z / -Z   (up / down)
    - o / c : open / close gripper
    - t     : toggle global <-> local frame
    - m     : toggle dq <-> pos solver
    - ESC / q : stop

Run:
    python examples/teleoperate_ee_keyboard.py --port /dev/tty.usbserial-XXXX --id my_ax_arm
"""

import argparse
import time
from importlib.resources import files

import casadi as cs
import numpy as np
import scipy as sp
from urdf2casadi import urdfparser as u2c
from urdf2casadi.geometry import dual_quaternion, quaternion

from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_ax_arm import AXArm, AXArmConfig
from lerobot_robot_ax_arm.urdf_mapping import (
    ARM_JOINTS,
    REFERENCE_URDF_DEG,
    ticks_to_urdf_vector,
    urdf_vector_to_ticks,
)

FPS = 30
HOME_TIME_S = 2.0  # duration of the ramped move to the reference pose at startup
CART_STEP_M = 0.008  # default end-effector Cartesian motion per tick, meters (override with --speed)
MAX_JOINT_STEP_RAD = 0.15  # safety cap on joint motion per tick (keeps motion bounded near singularities)
DLS_LAMBDA = 0.02  # damping factor for the position IK, well-behaved near singularities
GRIP_STEP_TICK = 15  # gripper ticks per press
FIT_THRESHOLD = 0.1  # only fit dual-quaternion derivative components above this magnitude

ROOT_LINK = "base_link"
TIP_LINK = "gripper_link"
CHAIN_LINKS = ("robot_link_1", "robot_link_2", "robot_link_3", "gripper_link")


def _skew(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def _build_kinematics(urdf_path: str) -> dict:
    """FK + Jacobians (dual-quaternion and position-only) and joint limits from the URDF."""
    parser = u2c.URDFparser()
    parser.from_file(urdf_path)
    fk = parser.get_forward_kinematics(ROOT_LINK, TIP_LINK)
    q_sym = fk["q"]
    fk_dq = fk["dual_quaternion_fk"]
    fk_T = fk["T_fk"]
    return {
        "fk_dq": fk_dq,
        "fk_T": fk_T,
        "dq_jac": cs.Function("dq_jac", [q_sym], [cs.jacobian(fk_dq(q_sym), q_sym)]),
        "pos_jac": cs.Function("pos_jac", [q_sym], [cs.jacobian(fk_T(q_sym)[:3, 3], q_sym)]),
        "lower": np.array(fk["lower"], dtype=float).flatten(),
        "upper": np.array(fk["upper"], dtype=float).flatten(),
    }


def build_link_fks(urdf_path: str):
    """Casadi T_fk functions base->each link in CHAIN_LINKS, for drawing the arm."""
    fks = []
    for tip in CHAIN_LINKS:
        parser = u2c.URDFparser()
        parser.from_file(urdf_path)
        fk = parser.get_forward_kinematics(ROOT_LINK, tip)
        fks.append((fk["T_fk"], fk["q"].shape[0]))
    return fks


def chain_points(link_fks, q_rad: np.ndarray) -> np.ndarray:
    """3D positions of the base and each link origin along the kinematic chain."""
    pts = [np.zeros(3)]
    for T, n in link_fks:
        pts.append(np.array(T(q_rad[:n]))[:3, 3])
    return np.array(pts)


def open_live_view(link_fks, reach: float = 0.42):
    """Open a non-blocking 3D plot; returns an update(q_rad, title) callback (False once closed)."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    (chain_line,) = ax.plot([], [], [], "-o", lw=3, color="tab:blue")
    (ee_pt,) = ax.plot([], [], [], "o", ms=10, color="tab:red")
    ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach); ax.set_zlim(-0.05, reach)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.ion(); plt.show(block=False)

    def update(q_rad: np.ndarray, title: str) -> bool:
        if not plt.fignum_exists(fig.number):
            return False
        pts = chain_points(link_fks, q_rad)
        chain_line.set_data(pts[:, 0], pts[:, 1]); chain_line.set_3d_properties(pts[:, 2])
        ee_pt.set_data(pts[-1:, 0], pts[-1:, 1]); ee_pt.set_3d_properties(pts[-1:, 2])
        ax.set_title(title)
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        return True

    return update


def _world_twist_to_dq_dot(pose: np.ndarray, twist_world: np.ndarray) -> np.ndarray:
    """World-frame twist [v, w] -> dual-quaternion derivative x_dot at the current pose."""
    T = dual_quaternion.to_numpy_transformation_matrix(pose)
    adj = np.zeros((6, 6))
    adj[:3, :3] = T[:3, :3]
    adj[3:, 3:] = T[:3, :3]
    adj[:3, 3:] = _skew(T[:3, -1]) @ T[:3, :3]
    twist = adj @ twist_world

    v = np.append(twist[:3], 0)
    w = np.append(twist[3:], 0)
    primal = pose[:4]
    dual = pose[4:]
    primal_conj = np.append(-primal[:3], primal[3])
    p = 2 * quaternion.numpy_product(dual, primal_conj)

    xi = np.zeros(8)
    xi[:4] = np.append(0, twist[3:])
    xi[4:] = v + (quaternion.numpy_product(p, w) - quaternion.numpy_product(w, p)) / 2
    return 0.5 * dual_quaternion.numpy_product(xi, pose)


def _fitness(q_dot, jacobian, x_dot):
    index = np.where(np.abs(x_dot) > FIT_THRESHOLD)
    delta = (np.dot(jacobian, q_dot) - x_dot)[index]
    return np.dot(delta.T, delta)


def _fitness_jacobian(q_dot, jacobian, x_dot):
    return 2 * np.dot(jacobian.T, np.dot(jacobian, q_dot) - x_dot)


def _joint_velocity(kin: dict, q_rad: np.ndarray, cmd: np.ndarray, frame: str, solver: str) -> np.ndarray:
    """Joint velocity producing the requested unit Cartesian motion ``cmd`` (x, y, z)."""
    rot = np.array(kin["fk_T"](q_rad))[:3, :3]
    # World-frame ("global") translation on this 3-DoF (no-wrist) arm is only reliable via the position
    # Jacobian: the dq resolved-rate tries to hold EE orientation fixed, which an underactuated arm
    # cannot do, leaking the motion onto other axes as the arm reorients. So global always uses pos.
    if solver == "pos" or frame == "global":
        v_world = cmd.astype(float) if frame == "global" else rot @ cmd
        J = np.array(kin["pos_jac"](q_rad))
        # Damped least squares: bounded, well-conditioned joint velocity even near singularities.
        return J.T @ np.linalg.solve(J @ J.T + DLS_LAMBDA**2 * np.eye(3), v_world)

    # Dual-quaternion resolved-rate (local frame only): move along the end-effector's own axes.
    lin = cmd.astype(float)
    pose = np.array(kin["fk_dq"](q_rad)).flatten()
    x_dot = _world_twist_to_dq_dot(pose, np.concatenate([lin, np.zeros(3)]))
    jacobian = np.array(kin["dq_jac"](q_rad))
    sol = sp.optimize.least_squares(
        _fitness, np.zeros(len(ARM_JOINTS)), args=(jacobian, x_dot), xtol=1e-4, jac=_fitness_jacobian,
    )
    return sol.x if sol.success else np.zeros(len(ARM_JOINTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port of the AX arm")
    parser.add_argument("--id", default="my_ax_arm", help="Robot id used for calibration files")
    parser.add_argument("--speed", type=float, default=CART_STEP_M,
                        help="End-effector Cartesian motion per tick in meters (higher = faster)")
    parser.add_argument("--frame", choices=("local", "global"), default="local",
                        help="Motion frame: 'local' = end-effector axes, 'global' = world axes")
    parser.add_argument("--solver", choices=("dq", "pos"), default="dq",
                        help="IK solver: 'dq' = dual-quaternion resolved-rate, 'pos' = position Jacobian")
    parser.add_argument("--view", action="store_true",
                        help="Show a live 3D plot of the real arm (digital twin) alongside teleop")
    args = parser.parse_args()
    cart_step = args.speed

    robot = AXArm(AXArmConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect(calibrate=False)
    if not robot.calibration:
        raise RuntimeError(f"No calibration found for id '{args.id}'. Run lerobot-calibrate first.")

    urdf_path = str(files("lerobot_robot_ax_arm") / "urdf" / "ax_arm.urdf")
    kin = _build_kinematics(urdf_path)
    q_lower, q_upper = kin["lower"], kin["upper"]
    view_update = open_live_view(build_link_fks(urdf_path)) if args.view else None

    grip_calib = robot.calibration["gripper"]
    grip_tick = int(robot.bus.read("Present_Position", "gripper", normalize=False))

    # Slowly ramp to the reference ("zero") pose (0, 45, 90) before starting teleop.
    home_deg = np.array([REFERENCE_URDF_DEG[j] for j in ARM_JOINTS])
    home_ticks = urdf_vector_to_ticks(home_deg, robot.calibration)
    start_ticks = {j: float(robot.bus.read("Present_Position", j, normalize=False)) for j in ARM_JOINTS}
    print("Homing to zero pose (0, 45, 90)...")
    steps = max(1, int(HOME_TIME_S * FPS))
    for i in range(1, steps + 1):
        alpha = i / steps
        for j in ARM_JOINTS:
            c = robot.calibration[j]
            tick = (1 - alpha) * start_ticks[j] + alpha * home_ticks[j]
            robot.bus.write("Goal_Position", j, int(np.clip(tick, c.range_min, c.range_max)), normalize=False)
        precise_sleep(1.0 / FPS)

    pending = {"x": 0.0, "y": 0.0, "z": 0.0, "g": 0.0}
    state = {"quit": False, "frame": args.frame, "solver": args.solver}
    keymap = {"w": ("x", 1), "s": ("x", -1), "a": ("y", 1), "d": ("y", -1), "r": ("z", 1), "f": ("z", -1),
              "o": ("g", 1), "c": ("g", -1)}

    def on_key(name: str) -> None:
        k = name.lower()
        if k in ("esc", "q"):
            state["quit"] = True
        elif k == "t":
            state["frame"] = "global" if state["frame"] == "local" else "local"
        elif k == "m":
            state["solver"] = "pos" if state["solver"] == "dq" else "dq"
        elif k in keymap:
            axis, direction = keymap[k]
            pending[axis] += direction

    listener = create_key_listener(
        on_key, controls_help="w/s a/d r/f = XYZ, o/c = gripper, t = frame, m = solver, esc = stop"
    )
    if listener is None:
        raise RuntimeError("Needs an interactive terminal with a usable key listener.")

    print("Keyboard EE teleop (velocity IK -> position). w/s=X a/d=Y r/f=Z o/c=gripper, t=frame, m=solver, ESC=stop.")
    try:
        while not state["quit"]:
            t0 = time.perf_counter()

            ticks = {j: float(robot.bus.read("Present_Position", j, normalize=False)) for j in ARM_JOINTS}
            q_deg = ticks_to_urdf_vector(ticks, robot.calibration)
            q_rad = np.deg2rad(q_deg)

            lin = np.array([np.sign(pending["x"]), np.sign(pending["y"]), np.sign(pending["z"])])
            cmd_disp = lin.copy()
            pending["x"] = pending["y"] = pending["z"] = 0.0

            q_target_rad = q_rad
            if np.any(lin):
                q_dot = _joint_velocity(kin, q_rad, lin.astype(float), state["frame"], state["solver"])
                # Scale for a consistent end-effector Cartesian speed (uniform across axes/poses),
                # then cap the joint step so motion stays bounded near singularities.
                ee_speed = float(np.linalg.norm(np.array(kin["pos_jac"](q_rad)) @ q_dot))
                if ee_speed > 1e-6:
                    q_step = q_dot * (cart_step / ee_speed)
                    step_norm = np.linalg.norm(q_step)
                    if step_norm > MAX_JOINT_STEP_RAD:
                        q_step *= MAX_JOINT_STEP_RAD / step_norm
                    q_target_rad = np.clip(q_rad + q_step, q_lower, q_upper)

            target_ticks = urdf_vector_to_ticks(np.rad2deg(q_target_rad), robot.calibration)
            for j in ARM_JOINTS:
                c = robot.calibration[j]
                tick = int(np.clip(target_ticks[j], c.range_min, c.range_max))
                robot.bus.write("Goal_Position", j, tick, normalize=False)

            g = np.sign(pending["g"])
            pending["g"] = 0.0
            if g:
                grip_tick = int(np.clip(grip_tick + g * GRIP_STEP_TICK, grip_calib.range_min, grip_calib.range_max))
                robot.bus.write("Goal_Position", "gripper", grip_tick, normalize=False)

            urdf_str = " ".join(f"{j}={v:+6.1f}" for j, v in zip(ARM_JOINTS, np.rad2deg(q_target_rad)))
            print(f"[{state['frame']:>6}|{state['solver']}] cmd[x={cmd_disp[0]:+.0f} y={cmd_disp[1]:+.0f}"
                  f" z={cmd_disp[2]:+.0f} g={g:+.0f}] -> urdf[{urdf_str}] gripper={grip_tick}",
                  end="\r", flush=True)

            if view_update is not None:
                # Draw the arm from the measured joint angles (real state, not the command).
                view_update(q_rad, f"REAL arm | frame={state['frame']} solver={state['solver']}")

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print()
        robot.disconnect()


if __name__ == "__main__":
    main()
