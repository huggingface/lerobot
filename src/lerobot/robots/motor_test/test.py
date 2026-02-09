import time

from lerobot.robots.motor_test import RobstrideTest, RobstrideTestConfig

# -------------------- User config --------------------
PORT = "can0"
MOTOR_CONFIG = {"joint_1": (0x01, 0x01, "ELO5")}

FORCE_FRESH_CALIBRATION = True
RUN_CALIBRATION_ON_CONNECT = True

# Motion test settings
RUN_MOTION_TEST = True
NUM_STEPS = 200
STEP_PERIOD_SEC = 0.05
POSITION_DEG_A = 20.0
POSITION_DEG_B = -20.0

# Bus API checks
RUN_BUS_GAIN_TEST = True
RUN_SYNC_READ_ALL_STATES_TEST = True
RUN_SEND_ACTION_TEST = True
RUN_SEND_NORMALIZED_ACTION_TEST = True
# -----------------------------------------------------


def force_fresh_calibration(robot: RobstrideTest) -> None:
    # Remove persisted calibration and clear loaded calibration from memory.
    if robot.calibration_fpath.is_file():
        robot.calibration_fpath.unlink()
    robot.calibration = {}
    robot.bus.calibration = {}


def run_basic_read_checks(robot: RobstrideTest) -> None:
    print("\n[CHECK] basic read paths")
    obs = robot.get_observation()
    print("observation:", obs)

    pos = robot.bus.read("Present_Position", "joint_1")
    vel = robot.bus.read("Present_Velocity", "joint_1")
    trq = robot.bus.read("Present_Torque", "joint_1")
    tmp = robot.bus.read("Temperature_MOS", "joint_1")
    print(f"single read joint_1: pos={pos:.3f}, vel={vel:.3f}, torque={trq:.3f}, temp={tmp:.1f}")

    sync_pos = robot.bus.sync_read("Present_Position")
    print("sync_read position:", sync_pos)


def run_gain_checks(robot: RobstrideTest) -> None:
    print("\n[CHECK] gain update paths")
    robot.bus.write("Kp", "joint_1", 12.0)
    robot.bus.write("Kd", "joint_1", 0.6)
    robot.bus.sync_write("Kp", {"joint_1": 15.0})
    robot.bus.sync_write("Kd", {"joint_1": 0.8})
    print("updated gains:", robot.bus._gains["joint_1"])


def run_state_dump_check(robot: RobstrideTest) -> None:
    if not hasattr(robot.bus, "sync_read_all_states"):
        print("\n[CHECK] sync_read_all_states not available on this bus")
        return

    print("\n[CHECK] sync_read_all_states")
    all_states = robot.bus.sync_read_all_states()
    print("all states:", all_states)


def run_action_checks(robot: RobstrideTest) -> None:
    print("\n[CHECK] action send paths")
    if RUN_SEND_ACTION_TEST:
        out_action = robot.send_action({"joint_1.pos": POSITION_DEG_A})
        print("send_action output:", out_action)
        time.sleep(0.2)

    if RUN_SEND_NORMALIZED_ACTION_TEST:
        out_norm_action = robot.send_normalized_action({"joint_1.pos": POSITION_DEG_B})
        print("send_normalized_action output:", out_norm_action)
        time.sleep(0.2)


def run_motion_test(robot: RobstrideTest) -> None:
    print("\n[CHECK] motion loop")
    for i in range(NUM_STEPS):
        tgt = POSITION_DEG_A if (i % 2 == 0) else POSITION_DEG_B
        robot.send_action({"joint_1.pos": tgt})
        obs = robot.get_observation()
        print(f"step={i:03d} target={tgt:>6.1f} obs={obs}")
        time.sleep(STEP_PERIOD_SEC)


def main() -> None:
    config = RobstrideTestConfig(
        port=PORT,
        motor_config=MOTOR_CONFIG,
        cameras={},
    )
    robot = RobstrideTest(config)

    if FORCE_FRESH_CALIBRATION:
        force_fresh_calibration(robot)

    print("[INFO] connecting...")
    robot.connect(calibrate=RUN_CALIBRATION_ON_CONNECT)
    print("[INFO] connected")

    try:
        run_basic_read_checks(robot)

        if RUN_BUS_GAIN_TEST:
            run_gain_checks(robot)

        if RUN_SYNC_READ_ALL_STATES_TEST:
            run_state_dump_check(robot)

        run_action_checks(robot)

        if RUN_MOTION_TEST:
            run_motion_test(robot)

    finally:
        print("\n[INFO] disconnecting...")
        robot.disconnect()
        print("[INFO] disconnected")


if __name__ == "__main__":
    main()
