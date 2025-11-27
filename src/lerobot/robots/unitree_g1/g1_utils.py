from enum import IntEnum

# ruff: noqa: N801, N815

# =============================================================================
# Serialization Keys for ZMQ Communication Protocol
# =============================================================================
# These constants define the JSON keys used for robot state/command serialization.
# They are shared between robot_server.py (robot-side) and unitree_sdk2_socket.py (client-side).

# Top-level message keys
TOPIC = "topic"
DATA = "data"

# Motor state keys
MOTOR_STATE = "motor_state"
MOTOR_Q = "q"  # position
MOTOR_DQ = "dq"  # velocity
MOTOR_TAU_EST = "tau_est"  # estimated torque
MOTOR_TEMPERATURE = "temperature"

# Motor command keys
MOTOR_CMD = "motor_cmd"
MOTOR_MODE = "mode"
MOTOR_KP = "kp"
MOTOR_KD = "kd"
MOTOR_TAU = "tau"

# IMU state keys
IMU_STATE = "imu_state"
IMU_QUATERNION = "quaternion"
IMU_GYROSCOPE = "gyroscope"
IMU_ACCELEROMETER = "accelerometer"
IMU_RPY = "rpy"
IMU_TEMPERATURE = "temperature"

# Other state keys
WIRELESS_REMOTE = "wireless_remote"
MODE_MACHINE = "mode_machine"
MODE_PR = "mode_pr"

# Number of motors
NUM_MOTORS = 35


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34
