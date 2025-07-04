import time
import traceback

from lerobot.robots.hope_jr import HopeJrHand, HopeJrHandConfig, homunculus_glove_to_hope_jr_hand
from lerobot.teleoperators.homunculus import (
    HomunculusGlove,
    HomunculusGloveConfig,
    homunculus_glove_to_hope_jr_hand,
)
from lerobot.utils.utils import move_cursor_up


def make_hand() -> tuple[HomunculusGlove, HopeJrHand]:
    glove_cfg = HomunculusGloveConfig("/dev/tty.usbmodem1201", id="right", side="right")
    glove = HomunculusGlove(glove_cfg)
    hand_cfg = HopeJrHandConfig("/dev/tty.usbmodem58760432281", id="right", side="right")
    hand = HopeJrHand(hand_cfg)
    return glove, hand


def main():
    glove, hand = make_hand()
    glove.connect()
    hand.connect()
    display_len = max(len(key) for key in hand.action_features)

    try:
        while True:
            start = time.perf_counter()

            glove_action = glove.get_action()
            hand_action = homunculus_glove_to_hope_jr_hand(glove_action)
            hand.send_action(hand_action)

            time.sleep(0.01)
            loop_s = time.perf_counter() - start

            print(f"\n{'NAME':<{display_len}} | {'HAND':>7}")
            for joint, val in hand_action.items():
                print(f"{joint:<{display_len}} | {val:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(hand_action) + 4)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        glove.disconnect()
        hand.disconnect()


if __name__ == "__main__":
    main()
