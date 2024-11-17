"""
Description: Simple script that shows how the right arm R mimics the angles and position
of the left arm L, which is set to manual mode and can be moved around, for 20 seconds.

It also monitors the digital output of ionum 2 and prints a message when it detects a value of 1.
Additionally, it monitors the digital input of ionum 2 and prints a message when it detects a value of 1.
"""

import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from xarm.wrapper import XArmAPI

# Global variables for caching speed and acceleration limits
MAX_SPEED_LIMIT = None
MAX_ACC_LIMIT = None


def initialize_limits(arm):
    global MAX_SPEED_LIMIT, MAX_ACC_LIMIT
    MAX_SPEED_LIMIT = max(arm.joint_speed_limit) / 3
    MAX_ACC_LIMIT = max(arm.joint_acc_limit) / 3


def mimic_arm(arm_source, arm_target, stop_event):
    start_time = time.time()
    iteration_count = 0

    while not stop_event.is_set():
        code, angles = arm_source.get_servo_angle()
        code_gripper, pos_gripper = arm_source.get_gripper_position()
        if code == 0:
            # Command the target arm to move to the source arm's joint angles
            arm_target.set_servo_angle_j(angles=angles, is_radian=False, wait=False)
        else:
            print(f"Failed to get angles from source arm: code {code}")

        if code_gripper == 0:
            # command the gripper to follow
            arm_target.set_gripper_position(pos=pos_gripper, wait=False)

        iteration_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:  # Print frequency every second
            frequency = iteration_count / elapsed_time
            print(f"Current teleoperation frequency: {frequency:.2f} Hz")
            start_time = time.time()
            iteration_count = 0

        # Small sleep to prevent CPU overuse
        # time.sleep(0.004)  # 250 Hz -> very smooth
        # time.sleep(0.02)  # 50 Hz -> smooth
        # time.sleep(0.033)  # 30 Hz -> barely smooth
        # time.sleep(0.05)   # 20 Hz  -> jerky motion
        # time.sleep(0.1)    # 10 Hz  -> jerky motion, makes the whole robot tremble
        # time.sleep(0.2)  # 5 Hz


def monitor_digital_output(arm, stop_event):
    while not stop_event.is_set():
        code, value = arm.get_tgpio_output_digital(ionum=2)
        if code == 0 and value == 1:
            print("Digital output 2 is HIGH")
        time.sleep(0.1)  # Check every 100ms


def monitor_digital_input(arm, stop_event):
    single_click_time = 0.2
    double_click_time = 0.5
    long_click_time = 1.0

    last_press_time = 0
    last_click_time = 0
    long_click_detected = False
    click_count = 0
    long_click_state = True  # starts in manual mode

    while not stop_event.is_set():
        code, value = arm.get_tgpio_digital(ionum=2)
        if code == 0:
            current_time = time.time()

            if value == 1:  # Button pressed
                if last_press_time == 0:
                    last_press_time = current_time
                elif not long_click_detected and current_time - last_press_time >= long_click_time:
                    print("Long click detected -> Switching manual mode")
                    long_click_detected = True
                    long_click_state = not long_click_state
                    if long_click_state:
                        arm.set_tgpio_digital(ionum=2, value=1)
                        # manual mode
                        arm.clean_error()
                        arm.set_mode(2)
                        arm.set_state(0)
                    else:
                        arm.set_tgpio_digital(ionum=2, value=0)
                        # disable manual mode
                        arm.clean_error()
                        arm.set_mode(0)
                        arm.set_state(0)
            else:  # Button released
                if last_press_time != 0:
                    press_duration = current_time - last_press_time

                    if not long_click_detected:
                        if press_duration < single_click_time:
                            click_count += 1
                            if click_count == 1:
                                last_click_time = current_time
                            elif click_count == 2:
                                if current_time - last_click_time < double_click_time:
                                    print("Double click detected -> Open gripper")
                                    arm.set_gripper_position(pos=600, wait=False)  # Open gripper
                                    click_count = 0
                                else:
                                    print("Single click detected -> Close gripper")
                                    arm.set_gripper_position(pos=50, wait=False)  # Close gripper
                                    click_count = 1
                                    last_click_time = current_time
                        else:
                            print("Single click detected -> Close gripper")
                            arm.set_gripper_position(pos=50, wait=False)  # Close gripper
                            click_count = 0

                    last_press_time = 0
                    long_click_detected = False

            # Reset click count if too much time has passed since last click
            if click_count == 1 and current_time - last_click_time >= double_click_time:
                print("Single click detected -> Close gripper")
                arm.set_gripper_position(pos=50, wait=False)  # Close gripper
                click_count = 0

        time.sleep(0.01)  # Check every 10ms for more precise detection


# IP addresses of the arms
ip_left = "192.168.1.236"
ip_right = "192.168.1.218"

# Initialize both arms
arm_left = XArmAPI(ip_left)
arm_right = XArmAPI(ip_right)

# Enable both arms, and grippers
## L
arm_left.motion_enable(enable=True)
arm_left.clean_error()
arm_left.set_mode(0)
arm_left.set_state(state=0)
#
arm_left.set_gripper_mode(0)
arm_left.set_gripper_enable(True)
arm_left.set_gripper_speed(5000)  # default speed, as there's no way to fetch gripper speed from API


## R
arm_right.motion_enable(enable=True)
arm_right.clean_error()
arm_right.set_mode(1)
arm_right.set_state(state=0)
#
arm_right.set_gripper_mode(0)
arm_right.set_gripper_enable(True)
arm_right.set_gripper_speed(5000)  # default speed, as there's no way to fetch gripper speed from API
# According to User Manual, should range in 1000-5000
#
# see https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf

# Initialize the global speed and acceleration limits
initialize_limits(arm_right)

# Set left arm to manual mode
arm_left.set_mode(2)
arm_left.set_state(0)
# Light up the digital output 2 (button), to signal manual mode
arm_left.set_tgpio_digital(ionum=2, value=1)

# Create a stop event for synchronization
stop_event = threading.Event()

# Create and start the mimic thread
mimic_thread = threading.Thread(target=mimic_arm, args=(arm_left, arm_right, stop_event))
mimic_thread.start()

# # Create and start the digital output monitoring thread
# monitor_output_thread = threading.Thread(target=monitor_digital_output, args=(arm_left, stop_event))
# monitor_output_thread.start()

# Create and start the digital input monitoring thread
monitor_input_thread = threading.Thread(target=monitor_digital_input, args=(arm_left, stop_event))
monitor_input_thread.start()

# Run
print("Starting mimic operation, digital output monitoring, and digital input monitoring...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Operation interrupted by user.")

# Stop the mimic operation and monitoring
stop_event.set()
mimic_thread.join()
# monitor_output_thread.join()
monitor_input_thread.join()

print("Mimic operation, digital output monitoring, and digital input monitoring completed.")

# Turn off manual mode after recording
arm_left.set_mode(0)
arm_left.set_state(0)
# Light down the digital output 2 (button), to signal manual mode
arm_left.set_tgpio_digital(ionum=2, value=0)

# Disconnect both arms
arm_left.disconnect()
arm_right.disconnect()
