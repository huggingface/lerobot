from lerobot_robot_piper import Piper, PiperConfig

cfg = PiperConfig(
    can_interface="can0",
    joint_names=[f"joint_{i+1}" for i in range(6)],
    joint_signs=[-1, 1, -1, 1, -1, 1],
    include_gripper=True,
    cameras={},
)

arm = Piper(cfg)
arm.connect()

obs = arm.get_observation()
print({k: obs[k] for k in cfg.joint_names if k in obs})
if cfg.include_gripper:
    print("gripper:", obs.get("gripper.pos"))

cmd = {f"{name}.pos": 0.0 for name in cfg.joint_names}
if cfg.include_gripper:
    cmd["gripper.pos"] = 0.0
arm.send_action(cmd)

arm.disconnect()
