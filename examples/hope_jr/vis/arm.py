import time
import traceback

from lerobot.robots.hope_jr import HopeJrArm, HopeJrArmConfig
from lerobot.utils.utils import move_cursor_up

# cfg = HopeJrArmConfig("/dev/tty.usbserial-140", id="right")
cfg = HopeJrArmConfig("/dev/tty.usbserial-1120", id="left", side="left")
arm = HopeJrArm(cfg)
display_len = max(len(key) for key in arm.action_features)

arm.connect()
arm.calibrate()
arm.bus.disable_torque()
try:
    while True:
        start = time.perf_counter()
        raw_obs = arm.bus.sync_read("Present_Position", normalize=False)
        norm_obs = {arm.bus.motors[name].id: val for name, val in raw_obs.items()}
        norm_obs = arm.bus._normalize(norm_obs)
        norm_obs = {arm.bus._id_to_name(id_): val for id_, val in norm_obs.items()}
        loop_s = time.perf_counter() - start

        print("\n----------------------------------")
        print(f"{'NAME':<15} | {'RAW':>7} | {'NORM':>7}")
        for motor in arm.bus.motors:
            print(f"{motor:<15} | {raw_obs[motor]:>7} | {norm_obs[motor]:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(len(arm.bus.motors) + 5)

except KeyboardInterrupt:
    pass
except Exception as e:
    traceback.print_exc(e)
finally:
    arm.disconnect()
