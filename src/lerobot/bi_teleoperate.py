import math
import sys
import time

from lerobot.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.robots.so101_follower_torque.so101_follower_t import SO101FollowerT
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FRQ = 100
PRINT_HZ = 10
RERUN_HZ = 100
ESC_CLR_EOL = "\x1b[K"
CURSOR_UP = "\x1b[F"

follower_cfg = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760431551",
    id="follower_arm_torque",
)

leader_cfg = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760428721",
    id="leader_arm_torque",
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

    # Joint-specific control gains for better tracking
    Kp_gains = follower.kp_gains
    Kd_gains = follower.kd_gains
    Kf_gains = follower.kf_gains

    # Compute torque commands in one line using list comprehension
    tau_cmd_f = [
        Kp_gains[j] * (pos_l[j] - pos_f[j])  # Position tracking
        + Kd_gains[j] * (vel_l[j] - vel_f[j])  # Velocity damping
        + Kf_gains[j] * (-tau_reaction_l[j] - tau_reaction_f[j])  # Force reflection
        for j in follower.bus.motors
    ]

    tau_cmd_l = [
        Kp_gains[j] * (pos_f[j] - pos_l[j])  # Position tracking
        + Kd_gains[j] * (vel_f[j] - vel_l[j])  # Velocity damping
        + Kf_gains[j] * (-tau_reaction_f[j] - tau_reaction_l[j])  # Force reflection
        for j in leader.bus.motors
    ]

    # Store interaction torques and debug info
    for i, j in enumerate(follower.bus.motors):
        # Store debug info
        debug_info_f[j] = {
            "τ_measured": tau_meas_f[j],
            "τ_reaction": tau_reaction_f[j],
            "τ_ref": tau_cmd_f[i],
            "θ_err": pos_l[j] - pos_f[j],
            "ω_err": vel_l[j] - vel_f[j],
            "τ_err": -tau_reaction_l[j] - tau_reaction_f[j],
        }
        debug_info_l[j] = {
            "τ_measured": tau_meas_l[j],
            "τ_reaction": tau_reaction_l[j],
            "τ_ref": tau_cmd_l[i],
            "θ_err": pos_f[j] - pos_l[j],
            "ω_err": vel_f[j] - vel_l[j],
            "τ_err": -tau_reaction_f[j] - tau_reaction_l[j],
        }

    # Send torques to both arms
    follower.send_action({f"{m}.effort": tau_cmd_f[i] for i, m in enumerate(follower.bus.motors)})
    leader.send_action({f"{m}.effort": tau_cmd_l[i] for i, m in enumerate(leader.bus.motors)})

    # Observation: follower side only (θ_f, ω_f, τ_ext)
    observation = {
        "follower_joint_angles": pos_f,  # θ_f: current angles
        "follower_angular_velocities": vel_f,  # ω_f: current velocities
        "follower_external_torques": tau_reaction_f,  # τ_ext: measured minus deterministic components
    }

    # Action: leader targets (θ_leader[τ], ω_leader[τ], τ_leader[τ])
    action = {
        "leader_target_angles": pos_l,  # θ_leader[τ]: absolute target angles
        "leader_target_velocities": vel_l,  # ω_leader[τ]: absolute target velocities
        "leader_interaction_torques": tau_reaction_l,  # τ_leader[τ]: cmd minus deterministic components
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
        lines.append(f"{'Joint':<13}{'Pos':>8}{'React':>6}{'Meas':>6}{'Cmd':>6}")
        lines.append(f"{'':13}{'(deg)':>8}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}")
        lines.append("-" * 86)

        for i, j in enumerate(leader.bus.motors):
            debug_l = debug_info_l[j]

            lines.append(
                f"{j:<13s}"
                f"{math.degrees(pos_l[j]):+8.1f}"
                f"{debug_l['τ_reaction']:+6.2f}"
                f"{debug_l['τ_measured']:+6.2f}"
                f"{tau_cmd_l[i]:+6.2f}"
            )

        lines.append("")
        lines.append("FOLLOWER ARM TORQUE ANALYSIS:")
        lines.append(f"{'Joint':<13}{'Pos':>8}{'React':>6}{'Meas':>6}{'Cmd':>6}")
        lines.append(f"{'':13}{'(deg)':>8}{'(Nm)':>6}{'(Nm)':>6}{'(Nm)':>6}")
        lines.append("-" * 86)

        for i, j in enumerate(follower.bus.motors):
            debug_f = debug_info_f[j]

            lines.append(
                f"{j:<13s}"
                f"{math.degrees(pos_f[j]):+8.1f}"
                f"{debug_f['τ_reaction']:+6.2f}"
                f"{debug_f['τ_measured']:+6.2f}"
                f"{tau_cmd_f[i]:+6.2f}"
            )

        lines.append("")
        lines.append("=" * 86)
        lines.append("TORQUE COMPONENT EXPLANATIONS:")
        lines.append("• Pos   (joint pos)  = Joint position in degrees")
        lines.append("• React (reaction)  = External forces (human interaction, contact)")
        lines.append("• Meas  (measured)  = Raw torque from motor current sensor")
        lines.append("• Cmd   (command)   = Final torque sent to motor")
        lines.append("-" * 86)
        lines.append(
            "Cmd = Track + Vel + Force + (Added as feedforward in send_action: Grav + Inert + Frict)"
        )
        lines.append("React = Meas - Grav - Inert - Frict  (external forces)")
        lines.append("Force = Kf × (reflect_other_robot - React)  (telepresence)")
        lines.append("Frict = b_visc×ω + f_coulomb×sign(ω)  (transparency)")
        lines.append(
            f"Joint Gains: shoulder_pan Kp={Kp_gains['shoulder_pan']:.1f} | shoulder_pan Kd={Kd_gains['shoulder_pan']:.1f} | shoulder_pan Kf={Kf_gains['shoulder_pan']:.1f}"
        )
        lines.append(
            f"Friction Comp, Viscous: {follower.friction_viscous['shoulder_pan']:.3f} | Coulomb: {follower.friction_coulomb['shoulder_pan']:.3f} (robot-class)"
        )

        block = "\n".join(lines)
        if first_print:
            sys.stdout.write(block + "\n")
            first_print = False
        else:
            sys.stdout.write(CURSOR_UP * len(lines) + ESC_CLR_EOL + block + "\n")
        sys.stdout.flush()

    busy_wait(max(0.0, 1.0 / FRQ - (time.perf_counter() - tic)))
