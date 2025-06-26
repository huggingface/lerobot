import time
import traceback

from lerobot.common.teleoperators.homonculus import HomonculusGlove, HomonculusGloveConfig
from lerobot.common.utils.utils import move_cursor_up

config = HomonculusGloveConfig("/dev/tty.usbmodem11401", side="left", id="left")
glove = HomonculusGlove(config)
glove.connect()

display_len = max(len(key) for key in glove.action_features)
glove.calibrate()
try:
    while True:
        start = time.perf_counter()
        raw_action = glove._read(normalize=False)
        norm_action = glove._normalize(raw_action)
        loop_s = time.perf_counter() - start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'RAW':>7} | {'NORM':>7}")
        for joint in glove.joints:
            print(f"{joint:<{display_len}} | {raw_action[joint]:>7} | {norm_action[joint]:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(len(norm_action) + 5)

except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    glove.disconnect()
