import math
import sys
import time

import numpy as np

from lerobot.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.robots.so101_follower_torque.so101_follower_t import SO101FollowerT
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FRQ = 200
PRINT_HZ = 10
RERUN_HZ = 100
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
    id="follower_arm_tt",
)

leader_cfg = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760428721",
    id="leader_arm_tt",
)

follower = SO101FollowerT(follower_cfg)
leader = SO101FollowerT(leader_cfg)
follower.connect()
leader.connect()

# Initialize Rerun for visualization
_init_rerun("bilateral_teleoperation")


print("Starting 4-channel bilateral teleoperation")
first_print = True
loop_count = 0
tic_prev = time.perf_counter()

use_friction_comp = True  # Enable/disable friction compensation
use_gravity_comp = True  # Enable gravity compensation with corrected signs

# Initialize previous positions for velocity calculation
pos_f_prev = None
pos_l_prev = None

while True:
    tic = time.perf_counter()

    obs_l, obs_f = leader.get_observation(), follower.get_observation()

    dt = tic - tic_prev
    tic_prev = tic
    if dt <= 0.0:
        dt = 1e-3  # avoid div-by-zero on very first pass

    # Simplified model-based bilateral control
    tau_cmd_f, tau_cmd_l = [], []
    debug_info_f, debug_info_l = {}, {}

    # Collect torques with deterministic components removed for model
    tau_interaction_l = {}
    tau_interaction_f = {}

    # Collect data for all motors
    pos_f = {j: obs_f[f"{j}.pos"] for j in follower.bus.motors}
    vel_f = {j: obs_f[f"{j}.vel"] for j in follower.bus.motors}
    acc_f = {j: obs_f[f"{j}.acc"] for j in follower.bus.motors}
    tau_meas_f = {j: obs_f[f"{j}.tau_meas"] for j in follower.bus.motors}

    pos_l = {j: obs_l[f"{j}.pos"] for j in leader.bus.motors}
    vel_l = {j: obs_l[f"{j}.vel"] for j in leader.bus.motors}
    acc_l = {j: obs_l[f"{j}.acc"] for j in leader.bus.motors}
    tau_meas_l = {j: obs_l[f"{j}.tau_meas"] for j in leader.bus.motors}

    # Update previous positions for next iteration
    pos_f_prev = pos_f.copy()
    pos_l_prev = pos_l.copy()

    # Compute reaction torques using model-based approach
    # Include friction in model-based disturbance for consistency
    tau_reaction_f = follower._compute_model_based_disturbance(
        pos_f, vel_f, acc_f, tau_meas_f, include_friction=use_friction_comp
    )
    tau_reaction_l = leader._compute_model_based_disturbance(
        pos_l, vel_l, acc_l, tau_meas_l, include_friction=use_friction_comp
    )

    # Get gravity for feed-forward compensation
    grav_f = follower._gravity_from_q(pos_f)
    grav_l = leader._gravity_from_q(pos_l)

    # Get inertia torques
    inertia_f = follower._inertia_from_q_dq(pos_f, vel_f, acc_f)
    inertia_l = leader._inertia_from_q_dq(pos_l, vel_l, acc_l)

    # Joint-specific control gains for better tracking
    Kp_gains = follower.kp_gains  # Use robot class properties
    Kd_gains = follower.kd_gains  # Use robot class properties

    Kf = 0.05  # Force reflection gain (global)

    # Friction model parameters from robot class
    friction_viscous = follower.friction_viscous  # Use robot class properties
    friction_coulomb = follower.friction_coulomb  # Use robot class properties

    for j in follower.bus.motors:
        # Bilateral control references (eqs. 5-7)
        theta_cmd_f, omega_cmd_f = pos_l[j], vel_l[j]  # Follower follows leader
        theta_cmd_l, omega_cmd_l = pos_f[j], vel_f[j]  # Leader follows follower

        tau_cmd_f_ref = -tau_reaction_l[j]  # Reflect leader reaction torque
        tau_cmd_l_ref = -tau_reaction_f[j]  # Reflect follower reaction torque

        # Friction compensation (feedforward)
        if use_friction_comp:
            # Follower friction compensation
            tau_friction_f = friction_viscous[j] * vel_f[j] + friction_coulomb[j] * (
                1.0 if vel_f[j] > 0.01 else -1.0 if vel_f[j] < -0.01 else 0.0
            )

            # Leader friction compensation
            tau_friction_l = friction_viscous[j] * vel_l[j] + friction_coulomb[j] * (
                1.0 if vel_l[j] > 0.01 else -1.0 if vel_l[j] < -0.01 else 0.0
            )
        else:
            tau_friction_f = 0.0
            tau_friction_l = 0.0

        # Follower control with joint-specific gains and friction compensation
        tau_pos_f = Kp_gains[j] * (theta_cmd_f - pos_f[j])
        tau_vel_f = Kd_gains[j] * (omega_cmd_f - vel_f[j])
        tau_force_f = Kf * (tau_cmd_f_ref - tau_reaction_f[j])

        # Apply gravity compensation conditionally
        tau_gravity_f = grav_f[j] if use_gravity_comp else 0.0
        tau_inertia_f = inertia_f[j] if use_gravity_comp else 0.0

        tau_ref_f = tau_pos_f + tau_vel_f + tau_force_f + tau_gravity_f + tau_inertia_f + tau_friction_f

        # Leader control with joint-specific gains and friction compensation
        tau_pos_l = Kp_gains[j] * (theta_cmd_l - pos_l[j])
        tau_vel_l = Kd_gains[j] * (omega_cmd_l - vel_l[j])
        tau_force_l = Kf * (tau_cmd_l_ref - tau_reaction_l[j])

        # Apply gravity compensation conditionally
        tau_gravity_l = grav_l[j] if use_gravity_comp else 0.0
        tau_inertia_l = inertia_l[j] if use_gravity_comp else 0.0

        tau_ref_l = tau_pos_l + tau_vel_l + tau_force_l + tau_gravity_l + tau_inertia_l + tau_friction_l

        tau_cmd_f.append(tau_ref_f)
        tau_cmd_l.append(tau_ref_l)

        # Store interaction torques (cmd minus deterministic components) for model
        if use_friction_comp:
            # If friction compensation is enabled, friction is already removed in model-based disturbance
            tau_interaction_l[j] = tau_ref_l - tau_gravity_l - tau_inertia_l - tau_friction_l
            tau_interaction_f[j] = -tau_reaction_f[j]  # Already has friction removed
        else:
            # If friction compensation is disabled, use original logic
            tau_interaction_l[j] = tau_ref_l - tau_gravity_l - tau_inertia_l - tau_friction_l
            tau_interaction_f[j] = -(
                tau_meas_f[j] - tau_gravity_f - tau_inertia_f - tau_friction_f
            )  # Invert to match leader

        # Store debug info
        debug_info_f[j] = {
            "τ_measured": tau_meas_f[j],
            "τ_pos": tau_pos_f,
            "τ_vel": tau_vel_f,
            "τ_force": tau_force_f,
            "τ_gravity": tau_gravity_f,  # Use conditional gravity value
            "τ_inertia": tau_inertia_f,  # Use conditional inertia value
            "τ_friction": tau_friction_f,  # Add friction to debug info
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
            "τ_gravity": tau_gravity_l,  # Use conditional gravity value
            "τ_inertia": tau_inertia_l,  # Use conditional inertia value
            "τ_friction": tau_friction_l,  # Add friction to debug info
            "τ_reaction": tau_reaction_l[j],
            "τ_ref": tau_ref_l,
            "θ_err": theta_cmd_l - pos_l[j],
            "ω_err": omega_cmd_l - vel_l[j],
            "τ_err": tau_cmd_l_ref - tau_reaction_l[j],
        }

    # Send torques to both arms
    follower_action = to_cmd(np.array(tau_cmd_f), list(follower.bus.motors.keys()))
    leader_action = to_cmd(np.array(tau_cmd_l), list(leader.bus.motors.keys()))

    follower.send_action(follower_action)
    leader.send_action(leader_action)

    # Create structured observation and action for rerun
    # Observation: follower side only (θ_f, ω_f, τ_ext)
    observation = {
        "follower_joint_angles": pos_f,  # θ_f: current angles
        "follower_angular_velocities": vel_f,  # ω_f: current velocities
        "follower_external_torques": tau_interaction_f,  # τ_ext: measured minus deterministic components
    }

    # Action: leader targets (θ_leader[τ], ω_leader[τ], τ_leader[τ])
    action = {
        "leader_target_angles": pos_l,  # θ_leader[τ]: absolute target angles
        "leader_target_velocities": vel_l,  # ω_leader[τ]: absolute target velocities
        "leader_interaction_torques": tau_interaction_l,  # τ_leader[τ]: cmd minus deterministic components
    }

    # Log data for visualization (100 Hz)
    if loop_count % (FRQ // RERUN_HZ) == 0:
        log_rerun_data(observation, action)

    # Console diagnostics (10 Hz)
    loop_count += 1
    if loop_count % (FRQ // PRINT_HZ) == 0:
        hz = 1.0 / dt

        # Detailed torque analysis mode - LEADER
        lines = [f"Loop {hz:6.1f} Hz    Δt {dt * 1e3:5.2f} ms"]
        lines.append("=" * 106)
        lines.append("LEADER ARM TORQUE ANALYSIS:")
        lines.append(
            f"{'Joint':<13}{'Pos':>8}{'Grav':>6}{'Inert':>6}{'Track':>6}{'Vel':>6}{'Force':>6}{'Frict':>6}{'React':>6}{'Meas':>6}{'Cmd':>6}"
        )
        lines.append(
            f"{'':13}{'(deg)':>8}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}"
        )
        lines.append("-" * 86)

        for i, j in enumerate(leader.bus.motors):
            debug_l = debug_info_l[j]

            lines.append(
                f"{j:<13s}"
                f"{math.degrees(pos_l[j]):+8.1f}"
                f"{debug_l['τ_gravity']:+6.2f}"
                f"{debug_l['τ_inertia']:+6.2f}"
                f"{debug_l['τ_pos']:+6.2f}"
                f"{debug_l['τ_vel']:+6.2f}"
                f"{debug_l['τ_force']:+6.2f}"
                f"{debug_l['τ_friction']:+6.2f}"
                f"{debug_l['τ_reaction']:+6.2f}"
                f"{debug_l['τ_measured']:+6.2f}"
                f"{tau_cmd_l[i]:+6.2f}"
            )

        lines.append("")
        lines.append("FOLLOWER ARM TORQUE ANALYSIS:")
        lines.append(
            f"{'Joint':<13}{'Pos':>8}{'Grav':>6}{'Inert':>6}{'Track':>6}{'Vel':>6}{'Force':>6}{'Frict':>6}{'React':>6}{'Meas':>6}{'Cmd':>6}"
        )
        lines.append(
            f"{'':13}{'(deg)':>8}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}"
        )
        lines.append("-" * 86)

        for i, j in enumerate(follower.bus.motors):
            debug_f = debug_info_f[j]

            lines.append(
                f"{j:<13s}"
                f"{math.degrees(pos_f[j]):+8.1f}"
                f"{debug_f['τ_gravity']:+6.2f}"
                f"{debug_f['τ_inertia']:+6.2f}"
                f"{debug_f['τ_pos']:+6.2f}"
                f"{debug_f['τ_vel']:+6.2f}"
                f"{debug_f['τ_force']:+6.2f}"
                f"{debug_f['τ_friction']:+6.2f}"
                f"{debug_f['τ_reaction']:+6.2f}"
                f"{debug_f['τ_measured']:+6.2f}"
                f"{tau_cmd_f[i]:+6.2f}"
            )

        lines.append("")
        lines.append("=" * 86)
        lines.append("TORQUE COMPONENT EXPLANATIONS:")
        lines.append("• Pos   (joint pos)  = Joint position in degrees")
        lines.append("• Grav  (gravity)    = Feed-forward gravity compensation")
        lines.append("• Inert (inertia)    = Feed-forward inertia compensation")
        lines.append("• Track (tracking)   = Position tracking control (Kp error)")
        lines.append("• Vel   (velocity)  = Velocity damping control (Kd error)")
        lines.append("• Force (bilateral) = Force reflection between robots (telepresence)")
        lines.append("• Frict (friction)  = Feed-forward friction compensation (transparency)")
        lines.append("• React (reaction)  = External forces (human interaction, contact)")
        lines.append("• Meas  (measured)  = Raw torque from motor current sensor")
        lines.append("• Cmd   (command)   = Final torque sent to motor")
        lines.append("-" * 86)
        lines.append("Cmd = Track + Vel + Force + Grav + Inert + Frict")
        lines.append(
            "React = Meas - Grav - Inert - Frict  (external forces)"
            if use_friction_comp
            else "React = Meas - Grav - Inert  (external forces)"
        )
        lines.append("Force = Kf × (reflect_other_robot - React)  (telepresence)")
        lines.append("Frict = b_visc×ω + f_coulomb×sign(ω)  (transparency)")
        lines.append(
            f"Control Gains: Kp=robot-class (position) | Kd=robot-class (velocity) | Kf={Kf} (force reflection)"
        )
        lines.append(
            f"Joint Gains: shoulder_pan Kp={Kp_gains['shoulder_pan']:.1f} | shoulder_lift Kp={Kp_gains['shoulder_lift']:.1f} | elbow_flex Kp={Kp_gains['elbow_flex']:.1f}"
        )
        lines.append(
            f"Friction Comp: {'ON' if use_friction_comp else 'OFF'} | Viscous: {friction_viscous['shoulder_pan']:.3f} | Coulomb: {friction_coulomb['shoulder_pan']:.3f} (robot-class)"
        )
        lines.append(
            f"Gravity Comp: {'ON' if use_gravity_comp else 'OFF'} | Inertia Comp: {'ON' if use_gravity_comp else 'OFF'} | (Both disabled for testing)"
        )

        block = "\n".join(lines)
        if first_print:
            sys.stdout.write(block + "\n")
            first_print = False
        else:
            sys.stdout.write(CURSOR_UP * len(lines) + ESC_CLR_EOL + block + "\n")
        sys.stdout.flush()

    busy_wait(max(0.0, 1.0 / FRQ - (time.perf_counter() - tic)))
