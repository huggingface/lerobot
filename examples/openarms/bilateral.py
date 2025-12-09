import time
import numpy as np
import pinocchio as pin
from os.path import dirname

from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig


same_direction = {"joint_4", "gripper"}

idx = {
    "joint_1": 0,
    "joint_2": 1,
    "joint_3": 2,
    "joint_4": 3,
    "joint_5": 4,
    "joint_6": 5,
    "joint_7": 6,
    "gripper": 7,
}

# joints to freeze
frozen = {"joint_6", "joint_7", "gripper"}
initial_pose = {}


def pos_deg(rob, obs):
    out = {}
    for side in ("left", "right"):
        for m in getattr(rob, f"bus_{side}").motors:
            k = f"{side}_{m}.pos"
            if k in obs:
                out[f"{side}_{m}"] = obs[k]
    return out


def vel_rad(rob, obs):
    out = {}
    for side in ("left", "right"):
        for m in getattr(rob, f"bus_{side}").motors:
            k = f"{side}_{m}.vel"
            out[f"{side}_{m}"] = np.deg2rad(obs.get(k, 0.0))
    return out


def main():
    cfg = OpenArmsLeaderConfig(
        port_left="can0",
        port_right="can1",
        can_interface="socketcan",
        id="openarms_bilateral",
        manual_control=False,
    )

    rob = OpenArmsLeader(cfg)
    rob.connect(calibrate=True)

    urdf = "/home/yope/Documents/lerobot_g1_integration/openarm_description/openarm_bimanual_pybullet.urdf"
    rob.pin_robot = pin.RobotWrapper.BuildFromURDF(urdf, dirname(urdf))
    rob.pin_robot.data = rob.pin_robot.model.createData()

    dt = 0.005
    grav = 1.0
    fric = 0.3

    # capture initial pose to freeze selected joints later
    obs0 = rob.get_action()
    for side in ("left", "right"):
        for m in getattr(rob, f"bus_{side}").motors:
            key = f"{side}_{m}.pos"
            if key in obs0 and m in frozen:
                initial_pose[f"{side}_{m}"] = obs0[key]

    try:
        while True:
            obs = rob.get_action()

            pdeg = pos_deg(rob, obs)
            prad = {k: np.deg2rad(v) for k, v in pdeg.items()}
            vrad = vel_rad(rob, obs)

            tau_g = rob._gravity_from_q(prad)
            tau_f = rob._friction_from_velocity(vrad, friction_scale=fric)

            # bilateral midpoint calculation
            cmd = {}
            for m in rob.bus_right.motors:
                kl = f"left_{m}.pos"
                kr = f"right_{m}.pos"
                if kl not in obs or kr not in obs:
                    continue

                ql = obs[kl]
                qr = obs[kr]

                if m in same_direction:
                    qmid = 0.5 * (ql + qr)
                else:
                    qmid = 0.5 * (ql - qr)

                # assign midpoint for both
                cmd[f"left_{m}"] = qmid
                cmd[f"right_{m}"] = qmid if m in same_direction else -qmid

            # override midpoint with frozen values
            for key, val in initial_pose.items():
                cmd[key] = val

            # single mit control call
            for side in ("left", "right"):
                bus = getattr(rob, f"bus_{side}")
                for m in bus.motors:
                    base_key = f"{side}_{m}"
                    kp = float(cfg.position_kp[idx[m]])
                    kd = float(cfg.position_kd[idx[m]])
                    torque = tau_g.get(base_key, 0.0) * grav + tau_f.get(base_key, 0.0)
                    pos_cmd = cmd.get(base_key, pdeg.get(base_key, 0.0))

                    bus._mit_control(
                        motor=m,
                        kp=kp,
                        kd=kd,
                        position_degrees=pos_cmd,
                        velocity_deg_per_sec=0.0,
                        torque=torque,
                    )

            time.sleep(dt)

    except KeyboardInterrupt:
        pass

    rob.bus_left.disable_torque()
    rob.bus_right.disable_torque()
    rob.disconnect()


if __name__ == "__main__":
    main()
