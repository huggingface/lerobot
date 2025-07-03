import time
import traceback

from lerobot.robots.hope_jr import HopeJrArm, HopeJrArmConfig
from lerobot.teleoperators.homunculus import HomunculusArm, HomunculusArmConfig
from lerobot.utils.utils import move_cursor_up


def make_left_arm() -> tuple[HomunculusArm, HopeJrArm]:
    exo_arm_cfg = HomunculusArmConfig("/dev/tty.usbmodem11301", id="left")
    exo_arm = HomunculusArm(exo_arm_cfg)
    arm_cfg = HopeJrArmConfig("/dev/tty.usbserial-1120", id="left")
    arm = HopeJrArm(arm_cfg)
    return exo_arm, arm


def main():
    exo_arm, arm = make_left_arm()
    display_len = max(len(key) for key in exo_arm.action_features)
    exo_arm.connect()
    arm.connect()

    try:
        while True:
            start = time.perf_counter()

            arm_action = exo_arm.get_action()
            arm_action["shoulder_pitch.pos"] = -arm_action["shoulder_pitch.pos"]
            arm_action["wrist_yaw.pos"] = -arm_action["wrist_yaw.pos"]
            arm.send_action(arm_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'ARM':>7}")
            for joint, val in arm_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(arm_action) + 5)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        exo_arm.disconnect()
        arm.disconnect()


if __name__ == "__main__":
    main()
