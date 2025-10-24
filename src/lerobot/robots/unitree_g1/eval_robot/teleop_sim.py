# unitree_lerobot/eval_robot/teleop_sim_min.py
import time
import traceback
import numpy as np

from unitree_lerobot.lerobot.src.lerobot.teleoperators.homunculus import (
    HomunculusArm, HomunculusArmConfig
)
from unitree_lerobot.eval_robot.robot_control.robot_arm_test import G1_29_ArmController
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK

# ---- map teleop (5 chans in [-1,1]) -> joint targets (rad)
# Order: S_pitch, S_yaw, S_roll, Elbow_flex, Wrist_roll
def scale_to_joint_limits(u5: np.ndarray) -> np.ndarray:
    # Tweak these if your sim build expects the alternate set
    mins = np.array([-3.05,  0.00, -2.30, -1.00,  1.95], dtype=np.float32)
    maxs = np.array([ 2.65,  2.20,  2.30,  2.00, -1.00], dtype=np.float32)
    u = np.clip(u5.astype(np.float32), -1.0, 1.0)
    return mins + (u + 1.0) * 0.5 * (maxs - mins)

def main():
    # --- Teleop (Homunculus) ---
    EXO_PORT = "/dev/ttyACM0"     # change if needed
    EXO_ID   = "unitree_left"

    exo_cfg = HomunculusArmConfig(EXO_PORT, id=EXO_ID)
    exo = HomunculusArm(exo_cfg)
    exo.connect(calibrate=True)

    # --- Robot (Simulation) ---
    # No EvalRealConfig, no setup_robot_interface — direct construction
    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=True)

    # Determine DoF safely (fallback to 14 if not exposed)
    arm_dof = getattr(arm_ctrl, "arm_dof", 14)

    # Optional: neutral pose
    neutral = np.zeros(arm_dof, dtype=np.float32)
    try:
        tau0 = arm_ik.solve_tau(neutral)
        arm_ctrl.ctrl_dual_arm(neutral, tau0)
    except Exception:
        pass

    # Control loop
    HZ = 100.0
    DT = 1.0 / HZ

    # Simple smoothing to avoid jitter
    q_prev = neutral.copy()
    alpha = 0.35  # 0..1, higher = snappier

    try:
        while True:
            t0 = time.perf_counter()

            # 1) Read teleop dict (values ~ [-100, 100]); take first 5 channels
            teleop = exo.get_action()         # Ordered dict
            vals = list(teleop.values())[:5]

            # Keep first, negate others (matches your earlier convention)
            vals = [vals[0]] + [-v for v in vals[1:]]

            # Normalize to [-1, 1]
            norm = np.asarray(vals, dtype=np.float32) / 100.0

            # 2) Map to joint targets and assemble full arm command
            q5 = scale_to_joint_limits(norm)
            arm_cmd = np.zeros(arm_dof, dtype=np.float32)
            arm_cmd[:5] = q5

            # 3) Low-pass filter for smoothness
            q_smooth = alpha * arm_cmd + (1.0 - alpha) * q_prev
            q_prev = q_smooth

            # 4) Compute torques and send
            tau = arm_ik.solve_tau(q_smooth)
            arm_ctrl.ctrl_dual_arm(q_smooth, tau)

            # 5) rate control
            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        try:
            exo.disconnect()
        except Exception:
            pass
        # Optionally: arm_ctrl.ctrl_dual_arm_go_home()

if __name__ == "__main__":
    main()
