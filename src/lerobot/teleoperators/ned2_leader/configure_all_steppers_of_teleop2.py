"""
Python Calibration process using Dynamixel SDK

This script is used to configure all steppers on the robot.
It is used to set the velocity profile, the stall threshold, the direction, the delay, the motor ratio, the limit position min, the limit position max, and the default home position.

It is used to configure the steppers for the calibration process.

Script written by:
- Pierre HANTSON
"""



import time
import math
import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else : 
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

from dynamixel_sdk import *


STEPPERS = [
    {
        'id': 2,
        'v_start': 0.0,
        'a_1': 0.785,
        'v_1': 0.314,
        'a_max': 2.094,
        'v_max': 0.4,
        'd_max': 2.094,
        'd_1': 0.785,
        'v_stop': 0.0209,
        'stall_threshold': 7,
        'direction': -1,
        'delay': 200,
        'motor_ratio': 0.0872,
        'limit_position_min': -2.90,
        'limit_position_max': 2.90,
        'default_home_position': 0.0
    },
    {
        'id': 3,
        'v_start': 0.0,
        'a_1': 0.524,
        'v_1': 0.314,
        'a_max': 1.047,
        'v_max': 0.3,
        'd_max': 1.047,
        'd_1': 0.524,
        'v_stop': 0.0209,
        'stall_threshold': 7,
        'direction': 1,
        'delay': 1000,
        'motor_ratio': 0.0872,
        'limit_position_min': -2.09,
        'limit_position_max': 0.61,
        'default_home_position': 0.0
    },
    {
        'id': 4,
        'v_start': 0.0,
        'a_1': 0.785,
        'v_1': 0.314,
        'a_max': 1.309,
        'v_max': 0.3,
        'd_max': 1.309,
        'd_1': 0.785,
        'v_stop': 0.0209,
        'stall_threshold': 6,
        'direction': -1,
        'delay': 1000,
        'motor_ratio': 0.0872,
        'limit_position_min': -1.34,
        'limit_position_max': 1.54,
        'default_home_position': 0.0
    }
]

RADIAN_PER_SECONDS_TO_RPM = 9.549296586
RADIAN_PER_SECONDS_SQ_TO_RPM_SQ = 91.1887
STEPPERS_MOTOR_STEPS_PER_REVOLUTION = 200.0
MICRO_STEPS = 8.0
ADDR_VSTART = 1024
ADDR_A1 = 1028
ADDR_V1 = 1032
ADDR_AMAX = 1036
ADDR_VMAX = 1040
ADDR_DMAX = 1044
ADDR_D1 = 1048
ADDR_VSTOP = 1052
ADDR_HOMING_DIRECTION = 149
ADDR_HOMING_STALL_THRESHOLD = 150
ADDR_COMMAND = 147
ADDR_HOMING_STATUS = 148
ADDR_PRESENT_POSITION = 132
ADDR_HOMING_ABS_POSITION = 151
ADDR_TORQUE_ENABLE = 64
ADDR_HW_ERROR_STATUS = 70

def write_register(packetHandler, portHandler, motor_id, address, value, byte_length):
    if byte_length == 1:
        value = value & 0xFF
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, address, value)
    elif byte_length == 2:
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, motor_id, address, value)
    elif byte_length == 4:
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, motor_id, address, value)
    else:
        raise ValueError("unsupported byte_length")
    if dxl_comm_result != 0 or dxl_error != 0:
        raise Exception(f"error writing register {address}: comm={dxl_comm_result}, err={dxl_error}")

def read_register(packetHandler, portHandler, motor_id, address, byte_length):
    if byte_length == 1:
        value, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, address)
    elif byte_length == 2:
        value, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, motor_id, address)
    elif byte_length == 4:
        value, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, motor_id, address)
    else:
        raise ValueError("unsupported byte_length")
    if dxl_comm_result != 0 or dxl_error != 0:
        raise Exception(f"error reading register {address}: comm={dxl_comm_result}, err={dxl_error}")
    return value

def calibrate_all_steppers(port_name="/tmp/ttyVIRTUAL1", baudrate=1000000, timeout=30):
    portHandler = PortHandler(port_name)
    packetHandler = PacketHandler(2.0)

    if not portHandler.openPort():
        raise Exception("master got error failed to open port")
    if not portHandler.setBaudRate(baudrate):
        raise Exception("master got error failed to set baudrate")
    try:
        for stepper in STEPPERS:
            motor_id = stepper['id']
            hw_error_status = read_register(packetHandler, portHandler, motor_id, ADDR_HW_ERROR_STATUS, 1)
            if hw_error_status != 0:
                raise Exception(f"master got error hw_error_status={hw_error_status} for motor {motor_id}")

            velocity_profile = {
                ADDR_VSTART: int(stepper['v_start'] * RADIAN_PER_SECONDS_TO_RPM * 100),
                ADDR_A1: int(stepper['a_1'] * RADIAN_PER_SECONDS_SQ_TO_RPM_SQ),
                ADDR_V1: int(stepper['v_1'] * RADIAN_PER_SECONDS_TO_RPM * 100),
                ADDR_AMAX: int(stepper['a_max'] * RADIAN_PER_SECONDS_SQ_TO_RPM_SQ),
                ADDR_VMAX: int(stepper['v_max'] * RADIAN_PER_SECONDS_TO_RPM * 100),
                ADDR_DMAX: int(stepper['d_max'] * RADIAN_PER_SECONDS_SQ_TO_RPM_SQ),
                ADDR_D1: int(stepper['d_1'] * RADIAN_PER_SECONDS_SQ_TO_RPM_SQ),
                ADDR_VSTOP: int(stepper['v_stop'] * RADIAN_PER_SECONDS_TO_RPM * 100)
            }
            for addr, value in velocity_profile.items():
                write_register(packetHandler, portHandler, motor_id, addr, value, 4)
                time.sleep(0.01)
            
            direction = stepper['direction']
            ttl_direction = 0 if direction < 0 else 1

            write_register(packetHandler, portHandler, motor_id, ADDR_HOMING_DIRECTION, ttl_direction, 1)
            write_register(packetHandler, portHandler, motor_id, ADDR_HOMING_STALL_THRESHOLD, stepper['stall_threshold'], 1)
            write_register(packetHandler, portHandler, motor_id, ADDR_COMMAND, 0, 1)
            time.sleep(0.01)

            start_time = time.time()
            while time.time() - start_time < timeout:
                homing_status = read_register(packetHandler, portHandler, motor_id, ADDR_HOMING_STATUS, 1)
                if homing_status == 2 : 
                    present_position = read_register(packetHandler, portHandler, motor_id, ADDR_PRESENT_POSITION, 4)
                    write_register(packetHandler, portHandler, motor_id, ADDR_HOMING_ABS_POSITION, present_position, 4)
                    write_register(packetHandler, portHandler, motor_id, ADDR_TORQUE_ENABLE, 0, 1)
                    break
                elif homing_status == 3:
                    raise Exception(f"master got error homing_status={homing_status} for motor {motor_id}")
                time.sleep(0.01)
            else:
                raise Exception(f"master got error timeout={timeout} for motor {motor_id}")
        portHandler.closePort()
        return True, "master got configured"
    except Exception as e:
        portHandler.closePort()
        return False, str(e)
    


if __name__ == "__main__":
    result, info = calibrate_all_steppers()
    print(result, info)
