import time
import traceback

from lerobot.common.robots.hope_jr import HopeJrHand, HopeJrHandConfig, homonculus_glove_to_hope_jr_hand
from lerobot.common.teleoperators.homonculus import HomonculusGlove, HomonculusGloveConfig
from lerobot.common.utils.utils import move_cursor_up


def make_left_hand() -> tuple[HomonculusGlove, HopeJrHand]:
    left_glove_cfg = HomonculusGloveConfig("/dev/tty.usbmodem11301", id="left", side="left")
    left_glove = HomonculusGlove(left_glove_cfg)
    left_hand_cfg = HopeJrHandConfig("/dev/cu.usbmodem58760432281", id="left", side="left")
    left_hand = HopeJrHand(left_hand_cfg)
    return left_glove, left_hand


def main():
    left_glove, left_hand = make_left_hand()
    left_glove.connect()
    left_hand.connect()
    display_len = max(len(key) for key in left_hand.action_features)

    try:
        while True:
            start = time.perf_counter()

            left_glove_action = left_glove.get_action()
            left_hand_action = homonculus_glove_to_hope_jr_hand(left_glove_action)
            left_hand.send_action(left_hand_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print(f"\n{'NAME':<{display_len}} | {'HAND':>7}")
            for joint, val in left_hand_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(left_hand_action) + 4)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        left_glove.disconnect()
        left_hand.disconnect()


if __name__ == "__main__":
    main()
