import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lerobot.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.robots.so101_follower_torque.so101_follower_t import SO101FollowerT
from lerobot.utils.robot_utils import busy_wait

FRQ = 300
PRINT_HZ = 10
ESC_CLR_EOL = "\x1b[K"
CURSOR_UP = "\x1b[F"


def vec(obs, key_suffix, joints):
    """dict → 1-D numpy array (selected joints)"""
    return np.array([obs[f"{m}{key_suffix}"] for m in joints])


def to_cmd(tau_vec, joints):
    """1-D array → {'joint.effort': value}"""
    return {f"{m}.effort": v for m, v in zip(joints, tau_vec, strict=False)}


follower_cfg = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_awesome_follower_arm",
)

leader_cfg = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760428721",
    id="my_awesome_leader_arm",
)

follower = SO101FollowerT(follower_cfg)
leader = SO101FollowerT(leader_cfg)
follower.connect()
leader.connect()

exec_f = ThreadPoolExecutor(max_workers=1)
exec_l = ThreadPoolExecutor(max_workers=1)

print("Starting 4-channel bilateral teleoperation")
first_print = True
loop_count = 0
tic_prev = time.perf_counter()

while True:
    tic = time.perf_counter()

    fut_l = exec_l.submit(leader.get_observation)
    fut_f = exec_f.submit(follower.get_observation)
    obs_l, obs_f = fut_l.result(), fut_f.result()

    dt = tic - tic_prev
    tic_prev = tic
    if dt <= 0.0:
        dt = 1e-3  # avoid div-by-zero on very first pass

    # Simplified model-based bilateral control
    tau_cmd_f, tau_cmd_l = [], []
    debug_info_f, debug_info_l = {}, {}

    # Collect data for all motors
    pos_f = {j: obs_f[f"{j}.pos"] for j in follower.bus.motors}
    vel_f = {j: obs_f[f"{j}.vel"] for j in follower.bus.motors}
    acc_f = {j: obs_f[f"{j}.acc"] for j in follower.bus.motors}
    tau_meas_f = {j: obs_f[f"{j}.tau_meas"] for j in follower.bus.motors}

    pos_l = {j: obs_l[f"{j}.pos"] for j in leader.bus.motors}
    vel_l = {j: obs_l[f"{j}.vel"] for j in leader.bus.motors}
    acc_l = {j: obs_l[f"{j}.acc"] for j in leader.bus.motors}
    tau_meas_l = {j: obs_l[f"{j}.tau_meas"] for j in leader.bus.motors}

    # Compute reaction torques using model-based approach
    tau_reaction_f = follower._compute_model_based_disturbance(pos_f, vel_f, acc_f, tau_meas_f)
    tau_reaction_l = leader._compute_model_based_disturbance(pos_l, vel_l, acc_l, tau_meas_l)

    # Get gravity for feed-forward compensation
    grav_f = follower._gravity_from_q(pos_f)
    grav_l = leader._gravity_from_q(pos_l)

    # Get inertia torques
    inertia_f = follower._inertia_from_q_dq(pos_f, vel_f, acc_f)
    inertia_l = leader._inertia_from_q_dq(pos_l, vel_l, acc_l)

    Kp = 8.0  # Position gain
    Kd = 0.1  # Velocity gain
    Kf = 0.1  # Force reflection gain

    for j in follower.bus.motors:
        # Bilateral control references (eqs. 5-7)
        theta_cmd_f, omega_cmd_f = pos_l[j], vel_l[j]  # Follower follows leader
        theta_cmd_l, omega_cmd_l = pos_f[j], vel_f[j]  # Leader follows follower

        tau_cmd_f_ref = -tau_reaction_l[j]  # Reflect leader reaction torque
        tau_cmd_l_ref = -tau_reaction_f[j]  # Reflect follower reaction torque

        # Follower control
        tau_pos_f = Kp * (theta_cmd_f - pos_f[j])
        tau_vel_f = Kd * (omega_cmd_f - vel_f[j])
        tau_force_f = Kf * (tau_cmd_f_ref - tau_reaction_f[j])
        tau_ref_f = tau_pos_f + tau_vel_f + tau_force_f + grav_f[j]

        # Leader control
        tau_pos_l = Kp * (theta_cmd_l - pos_l[j])
        tau_vel_l = Kd * (omega_cmd_l - vel_l[j])
        tau_force_l = Kf * (tau_cmd_l_ref - tau_reaction_l[j])
        tau_ref_l = tau_pos_l + tau_vel_l + tau_force_l + grav_l[j]

        tau_cmd_f.append(tau_ref_f)
        tau_cmd_l.append(tau_ref_l)

        # Store debug info
        debug_info_f[j] = {
            "τ_measured": tau_meas_f[j],
            "τ_pos": tau_pos_f,
            "τ_vel": tau_vel_f,
            "τ_force": tau_force_f,
            "τ_gravity": grav_f[j],
            "τ_inertia": inertia_f[j],
            "τ_reaction": tau_reaction_f[j],
            "τ_ref": tau_ref_f,
            "θ_err": theta_cmd_f - pos_f[j],
            "ω_err": omega_cmd_f - vel_f[j],
            "τ_err": tau_cmd_f_ref - tau_reaction_f[j],
        }
        debug_info_l[j] = {
            "τ_measured": tau_meas_l[j],
            "τ_pos": tau_pos_l,
            "τ_vel": tau_vel_l,
            "τ_force": tau_force_l,
            "τ_gravity": grav_l[j],
            "τ_inertia": inertia_l[j],
            "τ_reaction": tau_reaction_l[j],
            "τ_ref": tau_ref_l,
            "θ_err": theta_cmd_l - pos_l[j],
            "ω_err": omega_cmd_l - vel_l[j],
            "τ_err": tau_cmd_l_ref - tau_reaction_l[j],
        }

    # Send torques to both arms
    follower.send_action(to_cmd(np.array(tau_cmd_f), list(follower.bus.motors.keys())))
    leader.send_action(to_cmd(np.array(tau_cmd_l), list(leader.bus.motors.keys())))

    # Console diagnostics (10 Hz)
    loop_count += 1
    if loop_count % (FRQ // PRINT_HZ) == 0:
        hz = 1.0 / dt

        # Detailed torque analysis mode
        lines = [f"Loop {hz:6.1f} Hz    Δt {dt * 1e3:5.2f} ms"]
        lines.append("=" * 100)
        lines.append(
            f"{'Joint':<13}{'L':>8}{'F':>8}{'Grav':>6}{'Inert':>6}{'Pos':>6}{'Vel':>6}{'Force':>6}{'React':>6}{'Meas':>6}{'Cmd':>6}"
        )
        lines.append(
            f"{'':13}{'(deg)':>8}{'(deg)':>8}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}"
        )
        lines.append("-" * 100)

        for i, j in enumerate(follower.bus.motors):
            debug_f = debug_info_f[j]
            debug_l = debug_info_l[j]

            lines.append(
                f"{j:<13s}"
                f"{math.degrees(pos_l[j]):+8.1f}"
                f"{math.degrees(pos_f[j]):+8.1f}"
                f"{debug_f['τ_gravity']:+6.2f}"
                f"{debug_f['τ_inertia']:+6.2f}"
                f"{debug_f['τ_pos']:+6.2f}"
                f"{debug_f['τ_vel']:+6.2f}"
                f"{debug_f['τ_force']:+6.2f}"
                f"{debug_f['τ_reaction']:+6.2f}"
                f"{debug_f['τ_measured']:+6.2f}"
                f"{tau_cmd_f[i]:+6.2f}"
            )

        lines.append("-" * 100)
        lines.append("Model-Based Analysis:")
        lines.append(
            f"{'Joint':<13} {'Gravity':<8} {'Inertia':<8} {'Reaction':<8} {'Errors':<12} {'Balance':<8}"
        )
        for j in follower.bus.motors:
            debug_f = debug_info_f[j]
            g_dir = "↑" if debug_f["τ_gravity"] > 0 else "↓" if debug_f["τ_gravity"] < 0 else "≈0"
            i_dir = "+" if debug_f["τ_inertia"] > 0.05 else "-" if debug_f["τ_inertia"] < -0.05 else "≈0"
            r_dir = "+" if debug_f["τ_reaction"] > 0.05 else "-" if debug_f["τ_reaction"] < -0.05 else "≈0"

            pos_err_sign = "+" if debug_f["θ_err"] > 0 else "-" if debug_f["θ_err"] < 0 else "0"
            vel_err_sign = "+" if debug_f["ω_err"] > 0 else "-" if debug_f["ω_err"] < 0 else "0"

            # Check balance: τ_ref should equal sum of components
            expected_sum = debug_f["τ_pos"] + debug_f["τ_vel"] + debug_f["τ_force"] + debug_f["τ_gravity"]
            balance_check = "✓" if abs(expected_sum - debug_f["τ_ref"]) < 0.01 else "✗"

            lines.append(
                f"{j:<13s}"
                f"{g_dir:<8s}"
                f"{i_dir:<8s}"
                f"{r_dir:<8s}"
                f"θ:{pos_err_sign}ω:{vel_err_sign}    "
                f"{balance_check}"
            )

        lines.append("-" * 100)
        lines.append(
            f"Model-Based Control: React = τ_meas - τ_grav - τ_inert | Gains: Kp={Kp} Kd={Kd} Kf={Kf}"
        )

        block = "\n".join(lines)
        if first_print:
            sys.stdout.write(block + "\n")
            first_print = False
        else:
            sys.stdout.write(CURSOR_UP * len(lines) + ESC_CLR_EOL + block + "\n")
        sys.stdout.flush()

    busy_wait(max(0.0, 1.0 / FRQ - (time.perf_counter() - tic)))
