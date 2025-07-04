import math
import time

## Pseudo code for Impedance control and 4 channel bilateral tele-op
# 1. Read the leader and follower
# 2. Compute the follower's impedance torque
# 3. Compute the leader's torque
# 4. Set the leader and follower's torque
# # Read leader
# θ_ℓ, ω_ℓ, F_ℓ = leader.read_state()
# # Read the follower
# θ_f, ω_f, F_f = follower.read_state()
# # Compute the follower’s impedance torque
# τ_f = (
#     Kp*(θ_ℓ - θ_f)    # position coupling
#   + Kd*(ω_ℓ - ω_f)    # velocity coupling
#   + K_ff * F_ℓ        # force feed-forward from the leader
# )
# follower.set_torque(τ_f)
# # Compute the leader’s  torque
# τ_ℓ = (
#     K_mp*(θ_f   - θ_ℓ)   # motion-feedback
#   + K_md*(ω_f   - ω_ℓ)   # velocity-feedback
#   - K_mf * F_f          # force-feedback (negative)
# )
# leader.set_torque(τ_ℓ)
# Other details mentioned by Bi-ACT paper:
# The controller design adopted a control of position and force
# for each axis, as shown in Fig. 3. Angle information was
# obtained from encoders, and angular velocity was calculated
# by differentiating this information. The disturbance torque τdis
# was calculated using a disturbance observer (DOB) [29], and
# the torque response value τres was estimated using a force
# reaction observer (RFOB) [30].
from lerobot.common.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.common.robots.so101_follower_torque.so101_follower_t import SO101FollowerT

F_C = 0.2  # Nm, coulomb (static)
F_V = 0.05  # Nm·s rad⁻¹, residual viscous
Kt = 1.4043083213990768  # Nm/A
AMP_PER_LSB = 0.0065  # 6.5 mA/LSB
MAX_LSB = 2047
Kp, Kd, K_local = 15, 0.5, 0.01
DT = 0.002  # 500 Hz
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]  # "gripper"
TICKS_TO_RAD = 2.0 * math.pi / 4096.0  # Feetech HLS-3625: 4096 ticks = 1 rev

STEP = 0
SCALE_TRANSIENT = 0.2
SCALE_NORMAL = 1.0


def friction(omega: float) -> float:
    return F_C * math.tanh(omega / 0.05) + F_V * omega


def nm_to_lsb(tau: float) -> int:
    return int(max(-MAX_LSB, min(MAX_LSB, round((tau / Kt) / AMP_PER_LSB))))


def ticks_to_rad(ticks: int) -> float:
    return ticks * TICKS_TO_RAD


robot_config = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_awesome_follower_arm",
)

teleop_config = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760428721",
    id="my_awesome_leader_arm",
)

robot = SO101FollowerT(robot_config)

teleop_device = SO101FollowerT(teleop_config)

robot.connect()
teleop_device.connect()

print("Starting bilateral tele-op")
prev_l = dict.fromkeys(MOTORS, 0.0)
prev_f = dict.fromkeys(MOTORS, 0.0)

last_print = time.time()
loops = 0

while True:
    STEP += 1
    # first 1000 steps at 20%, then full effort
    scale = SCALE_TRANSIENT if STEP <= 1000 else SCALE_NORMAL

    tic = time.perf_counter()
    obs_l = robot.get_observation()
    obs_f = teleop_device.get_observation()

    tgt_l, tgt_f = {}, {}
    for j in MOTORS:
        ticks_l = obs_l[f"{j}.pos"]
        ticks_f = obs_f[f"{j}.pos"]

        th_l = ticks_to_rad(ticks_l)  # → radians
        th_f = ticks_to_rad(ticks_f)

        dth_l = (th_l - prev_l[j]) / DT
        dth_f = (th_f - prev_f[j]) / DT

        prev_l[j], prev_f[j] = th_l, th_f

        tau_l = -(Kp * (th_f - th_l) + Kd * (dth_f - dth_l) - K_local * dth_l) - friction(dth_l)
        tau_f = -(Kp * (th_l - th_f) + Kd * (dth_l - dth_f) - K_local * dth_f) - friction(dth_f)

        tau_l *= scale
        tau_f *= scale

        tgt_l[f"{j}.effort"] = nm_to_lsb(tau_l)
        tgt_f[f"{j}.effort"] = nm_to_lsb(tau_f)

    robot.send_action(tgt_l)
    teleop_device.send_action(tgt_f)

    leader_str = " | ".join(f"{j}:{tgt_l[f'{j}.effort']:+5d}" for j in MOTORS)
    follower_str = " | ".join(f"{j}:{tgt_f[f'{j}.effort']:+5d}" for j in MOTORS)
    print(f"Leader:   {leader_str}")
    print(f"Follower: {follower_str}")

    loops += 1
    elapsed = time.perf_counter() - tic
    to_sleep = DT - elapsed
    if to_sleep > 0:
        time.sleep(to_sleep)

    now = time.time()
    if now - last_print >= 1.0:
        print(f"{loops / (now - last_print):6.1f} Hz control loop")  # e.g. "500.2 Hz"
        loops = 0
        last_print = now
