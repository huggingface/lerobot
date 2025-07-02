import time
import traceback

from lerobot.robots.hope_jr import HopeJrArm, HopeJrArmConfig
from lerobot.teleoperators.homunculus import HomunculusArm, HomunculusArmConfig
from lerobot.utils.utils import move_cursor_up


def make_left_arm() -> tuple[HomunculusArm, HopeJrArm]:
    left_exo_arm_cfg = HomunculusArmConfig("/dev/tty.usbmodem2101", id="left")
    left_exo_arm = HomunculusArm(left_exo_arm_cfg)
    left_arm_cfg = HopeJrArmConfig("/dev/tty.usbserial-140", id="left")
    left_arm = HopeJrArm(left_arm_cfg)
    return left_exo_arm, left_arm


def main():
    left_exo_arm, left_arm = make_left_arm()
    display_len = max(len(key) for key in left_exo_arm.action_features)
    left_exo_arm.connect()
    left_arm.connect()

    try:
        while True:
            start = time.perf_counter()

            left_arm_action = left_exo_arm.get_action()
            left_arm_action["shoulder_pitch.pos"] = -left_arm_action["shoulder_pitch.pos"]
            left_arm_action["wrist_yaw.pos"] = -left_arm_action["wrist_yaw.pos"]
            left_arm.send_action(left_arm_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'ARM':>7}")
            for joint, val in left_arm_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(left_arm_action) + 5)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        left_exo_arm.disconnect()
        left_arm.disconnect()


if __name__ == "__main__":
    main()
