import time
import traceback

from lerobot.common.teleoperators.homonculus import HomonculusArm, HomonculusArmConfig
from lerobot.common.utils.utils import move_cursor_up

# cfg = HomonculusArmConfig("/dev/tty.usbmodem2101", id="left")
cfg = HomonculusArmConfig("/dev/tty.usbmodem11401", id="left")
arm = HomonculusArm(cfg)
display_len = max(len(key) for key in arm.action_features)

arm.connect()
arm.calibrate()
try:
    while True:
        start = time.perf_counter()
        raw_action = arm._read(normalize=False)
        norm_action = arm._normalize(raw_action)
        loop_s = time.perf_counter() - start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'RAW':>7} | {'NORM':>7}")
        for joint in arm.joints:
            print(f"{joint:<{display_len}} | {raw_action[joint]:>7} | {norm_action[joint]:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(len(norm_action) + 5)

except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    arm.disconnect()
