import time
import math
import numpy as np
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig


def main():
    cfg = OpenArmsFollowerConfig(
        port_left="can0",
        port_right="can1",
        can_interface="socketcan",
        id="openarms_test",
        manual_control=True,  # direct position control
    )

    print('connecting...')
    rob = OpenArmsFollower(cfg)
    rob.connect(calibrate=True)

    # disable left torque fully — keep it still
    rob.bus_left.disable_torque()

    # desired angular sweep = 1/4 of current joint range
    sweep_deg = 20.0  # tweak if you want bigger movement

    # frequency of movement
    hz = 100.0
    dt = 1.0 / hz
    move_time = 1.0  # seconds per joint

    print('starting right–arm joint test…')
    print('support the arm and keep clear')

    time.sleep(1.0)

    # iterate motors except gripper
    for motor in rob.bus_right.motors:
        if motor == 'gripper':
            continue

        print(f'testing {motor} on right arm...')
        start = time.time()

        # read current position as center
        obs = rob.get_action()
        key = f'right_{motor}.pos'
        center = obs.get(key, 0.0)

        t = 0.0
        while time.time() - start < move_time:
            offset = sweep_deg * math.sin(2 * math.pi * t)
            pos_cmd = center + offset

            rob.bus_right._mit_control(
                motor=motor,
                kp=3.0,    # some stiffness so it tracks well
                kd=0.2,
                position_degrees=pos_cmd,
                velocity_deg_per_sec=0.0,
                torque=0.0
            )

            t += dt
            time.sleep(dt)

        print(f'done {motor}')

    print('\nall right–arm joints tested')
    print('disabling torque…')
    rob.bus_right.disable_torque()
    rob.disconnect()


if __name__ == '__main__':
    main()
