import time
import traceback

from lerobot.robots.hope_jr import HopeJrHand, HopeJrHandConfig, homunculus_glove_to_hope_jr_hand
from lerobot.teleoperators.homunculus import HomunculusGlove, HomunculusGloveConfig
from lerobot.utils.utils import move_cursor_up


def make_right_hand() -> tuple[HomunculusGlove, HopeJrHand]:
    right_glove_cfg = HomunculusGloveConfig("/dev/tty.usbmodem2101", id="right", side="right")
    right_glove = HomunculusGlove(right_glove_cfg)
    right_hand_cfg = HopeJrHandConfig("/dev/tty.usbserial-140", id="right", side="right")
    right_hand = HopeJrHand(right_hand_cfg)
    return right_glove, right_hand


def main():
    right_glove, right_hand = make_right_hand()
    right_glove.connect()
    right_hand.connect()
    display_len = max(len(key) for key in right_hand.action_features)

    try:
        while True:
            start = time.perf_counter()

            right_glove_action = right_glove.get_action()
            right_hand_action = homunculus_glove_to_hope_jr_hand(right_glove_action)
            right_hand.send_action(right_hand_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print(f"\n{'NAME':<{display_len}} | {'HAND':>7}")
            for joint, val in right_hand_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(right_hand_action) + 4)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        right_glove.disconnect()
        right_hand.disconnect()


if __name__ == "__main__":
    main()
