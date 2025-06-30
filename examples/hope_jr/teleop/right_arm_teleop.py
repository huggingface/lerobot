import time
import traceback

from lerobot.common.robots.hope_jr import HopeJrArm, HopeJrArmConfig
from lerobot.common.teleoperators.homunculus import HomunculusArm, HomunculusArmConfig
from lerobot.common.utils.utils import move_cursor_up


def make_right_arm() -> tuple[HomunculusArm, HopeJrArm]:
    right_exo_arm_cfg = HomunculusArmConfig("/dev/tty.usbmodem2101", id="right")
    right_exo_arm = HomunculusArm(right_exo_arm_cfg)
    right_arm_cfg = HopeJrArmConfig("/dev/tty.usbserial-140", id="right")
    right_arm = HopeJrArm(right_arm_cfg)
    return right_exo_arm, right_arm


def main():
    right_exo_arm, right_arm = make_right_arm()
    display_len = max(len(key) for key in right_exo_arm.action_features)
    right_exo_arm.connect()
    right_arm.connect()

    try:
        while True:
            start = time.perf_counter()

            right_arm_action = right_exo_arm.get_action()
            right_arm_action["shoulder_pitch.pos"] = -right_arm_action["shoulder_pitch.pos"]
            right_arm_action["wrist_yaw.pos"] = -right_arm_action["wrist_yaw.pos"]
            right_arm.send_action(right_arm_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'ARM':>7}")
            for joint, val in right_arm_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(right_arm_action) + 5)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        right_exo_arm.disconnect()
        right_arm.disconnect()


if __name__ == "__main__":
    main()
