"""Debug helpers for Robstride motors over CAN (private, CANopen, and MIT protocols)."""

import struct
import time
from typing import Dict, Optional, Tuple, Union

import can
import numpy as np

from lerobot.motors.robstride.tables import CAN_CMD_CLEAR_FAULT, CommMode

# -----------------------------------------------------------------------------
# CAN ID helpers (private protocol, 29-bit extended frames)
# -----------------------------------------------------------------------------

def make_ext_id(comm_type: int, host_id: int, target_id: int) -> int:
    """
    Build a 29-bit extended CAN ID for the Robstride private protocol.

    Layout:
        bits 28–24 : communication type (5 bits)
        bits 23–8  : host CAN ID (16 bits)
        bits 7–0   : target CAN ID (8 bits)
    """
    return ((comm_type & 0x1F) << 24) | ((host_id & 0xFFFF) << 8) | (target_id & 0xFF)


def parse_reply_id(arbitration_id: int) -> Tuple[int, int, int]:
    """
    Parse a 29-bit ID that follows the Robstride private layout.

    Returns:
        (comm_type, host_field, target_field)
    """
    comm_type = (arbitration_id >> 24) & 0x1F
    host_field = (arbitration_id >> 8) & 0xFFFF
    target_field = arbitration_id & 0xFF
    return comm_type, host_field, target_field


# -----------------------------------------------------------------------------
# Ping helpers
# -----------------------------------------------------------------------------


def ping_private(bus: can.Bus, motor_id: int) -> bool:
    """
    Ping a motor using the private protocol.

    :param bus: CAN bus object
    :param motor_id: ID of the motor to ping
    :return: True if motor responds, False otherwise
    """
    # Build ping frame for the Robstride private protocol
    identifier = make_ext_id(1, motor_id, motor_id)
    ping_msg = can.Message(
        arbitration_id=identifier,
        data=[0x00] * 8,
        is_extended_id=True,
    )
    try:
        bus.send(ping_msg)
        # Wait for response
        response = bus.recv(timeout=0.1)
        if response:
            host_field, _target_field = parse_reply_id(response.arbitration_id)[1:]
            print("Received response from motor ID:", host_field)
            return True
        return False
    except can.CanError:
        print("CAN bus error while pinging motor ID:", motor_id)
        return False


def ping_canopen(bus: can.Bus, motor_id: int, timeout: float = 0.05) -> bool:
    """
    Check if a motor responds as a CANopen node by reading its device type.
    """

    # SDO upload request for 0x1000:00 (Device Type)
    # CCS=0x40 → initiate upload
    data = [
        0x40,       # SDO upload request
        0x00, 0x10, # Index 0x1000
        0x00,       # Subindex
        0x00, 0x00, 0x00, 0x00
    ]

    req_id = 0x600 + motor_id
    resp_id = 0x580 + motor_id

    msg = can.Message(arbitration_id=req_id, data=data, is_extended_id=False)
    bus.send(msg)

    start = time.time()
    while time.time() - start < timeout:
        resp = bus.recv(timeout=0.005)
        if resp is None:
            continue

        if resp:
            # Any valid SDO response means CANopen is alive
            print(resp)
            return True

    return False
def ping_mit(bus: can.Bus, motor_id: int) -> bool:
    """
    Ping a motor using the MIT protocol.

    :param bus: CAN bus object
    :param motor_id: ID of the motor to ping
    :return: True if motor responds, False otherwise
    """
    # Build ping frame for the MIT protocol
    data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)

    bus.send(msg)
    # Wait for response
    response = bus.recv(timeout=0.1)
    if response:
        print(msg)
        return True
    return False


# -----------------------------------------------------------------------------
# Protocol switching
# -----------------------------------------------------------------------------


def switch_canopen_to_private(bus: can.Bus, motor_id: int, host_id: int = 0x01) -> bool:
    """
    Switch a motor from CANopen back to the Robstride private protocol.

    Motor needs to be manually rebooted after receiving this frame.
    """

    COMM_TYPE = 25  # protocol switch command

    arb_id = (
        (COMM_TYPE & 0x1F) << 24
        | (host_id & 0xFFFF) << 8
        | (motor_id & 0xFF)
    )

    # 0 = private protocol
    data = [0] * 8

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)

    bus.send(msg)

    msg = bus.recv(timeout=0.5)
    if msg:
        print("success")
        return True

    print("failed")
    return False


def switch_mit_to_private(bus: can.Bus, motor_id: int) -> bool:
    """
    Switch a motor from MIT protocol to the Robstride private protocol.

    Motor needs to be manually rebooted after receiving this frame.
    """


    arb_id = motor_id

    # 0 = private protocol
    data = [0xFF] * 8
    data[7] = 0xFD
    data[6] = CommMode.PrivateProtocole.value

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    bus.send(msg)

    # Motor reboots → allow time for restart
    msg = bus.recv(timeout=0.5)
    if msg:
        print("success")
        return True

    print("failed")
    return False


def switch_private_to_mit(bus: can.Bus, motor_id: int, host_id: int = 0x01) -> bool:
    """
    Switch a motor from the private protocol to MIT protocol.

    Motor needs to be manually rebooted after receiving this frame.
    """
    arb_id = make_ext_id(0x19, host_id, motor_id)

    data = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x02,  # F_CMD = 2 → MIT
        0x00
    ]

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
    bus.send(msg)

    msg = bus.recv(timeout=0.5)
    if msg:
        print("success")
        return True

    print("failed")
    return False


# -----------------------------------------------------------------------------
# Private protocol: single parameter read/write + save
# -----------------------------------------------------------------------------


def single_parameter_read(
    bus: can.Bus,
    motor_id: int,
    param_index: int,
    host_id: int = 0x01,
) -> Optional[Dict[str, int]]:
    """
    Read a single parameter from a motor using the private protocol.
    """


    arb_id = make_ext_id(0x11, host_id, motor_id)

    data = [
        param_index & 0xFF,
        (param_index >> 8) & 0xFF,
        0,
        0,
        0,
        0,
        0,
        0x00
    ]

    msg = can.Message(
        arbitration_id=arb_id,
        data=data,
        is_extended_id=True,
    )

    bus.send(msg)

    time.sleep(0.02)
    msg = bus.recv(0.2)
    if msg:
        dic = decode_single_param_reply(msg)
        print(dic)
        return dic
    print("No response received for parameter read.")
    return None

def decode_single_param_reply(msg: can.Message) -> Dict[str, int]:
    """Decode a single-parameter read reply into a dictionary of fields."""
    arb_id = msg.arbitration_id

    comm_type = (arb_id >> 24) & 0x1F
    status = (arb_id >> 16) & 0xFF
    host_id = (arb_id >> 8) & 0xFF
    motor_id = arb_id & 0xFF

    if comm_type != 0x11:
        raise ValueError(f"Unexpected comm_type: 0x{comm_type:02X}")

    data = msg.data
    index = data[0] | (data[1] << 8)  # little-endian index

    # For uint32 parameter (like canTimeout):
    value = struct.unpack_from("<I", bytes(data[4:8]))[0]  # little-endian uint32

    return {
        "status": status,
        "host_id": host_id,
        "motor_id": motor_id,
        "index": index,
        "value": value,
    }

def single_parameter_write(
    bus: can.Bus,
    motor_id: int,
    param_index: int,
    param_value: int,
    size: int,
    host_id: int = 0x01,
) -> bool:
    """
    Write a single parameter to a motor using the private protocol.
    """

    msg = make_single_param_write_msg(
        motor_id=motor_id,
        index=param_index,
        value=param_value,
        size=size,  # assuming 4-byte parameter
        host_id=host_id,
    )
    bus.send(msg)
    msg = bus.recv(0.02)
    if msg:
        return True
    return False


def make_single_param_write_msg(
    motor_id: int,
    index: int,
    value: int,
    size: int = 4,      # 1, 2 or 4 bytes, depending on parameter type
    host_id: int = 0x01,
) -> can.Message:
    """
    Build a private-protocol frame (comm type 18) to write a single parameter.

    index : parameter index (e.g. 0x7028 for canTimeout)
    value : integer value to write
    size  : number of bytes used by the parameter (1 / 2 / 4)
    """
    if size not in (1, 2, 4):
        raise ValueError("size must be 1, 2 or 4 bytes")

    # 29-bit arbitration ID
    arb_id = make_ext_id(comm_type=0x12, target_id=motor_id, host_id=host_id)

    # Byte0–1: index, little-endian
    data = bytearray(8)
    data[0] = index & 0xFF
    data[1] = (index >> 8) & 0xFF

    # Byte2–3: 0x00 already (data[2] and data[3] = 0)

    # Byte4–7: value, little-endian, truncated/padded to 'size'
    value_bytes = value.to_bytes(size, byteorder="little", signed=False)
    data[4 : 4 + size] = value_bytes  # remaining bytes stay 0

    return can.Message(
        arbitration_id=arb_id,
        data=bytes(data),
        is_extended_id=True,
    )


def save_frame(bus: can.Bus, motor_id: int, host_id: int = 0x01) -> bool:
    """Ask the motor to persist its current configuration (private protocol)."""
    arb_id = make_ext_id(comm_type=0x12, target_id=motor_id, host_id=host_id)

    # Spec shows 01 02 03 04 05 06 07 08; likely ignored by the firmware.
    # You can also safely use [0]*8 if you prefer.
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    msg = can.Message(
        arbitration_id=arb_id,
        data=data,
        is_extended_id=True,
    )
    bus.send(msg)
    msg = bus.recv(0.2)
    if msg:
        return True
    return False


# -----------------------------------------------------------------------------
# MIT helpers: enable/disable/change ID/control + decode
# -----------------------------------------------------------------------------

def change_motor_id(bus: can.Bus, old_motor_id: int, new_motor_id: int) -> bool:
    """
    Change a motor ID using the MIT protocol.

    Motor needs to be manually rebooted after the ID change.
    """

    arb_id = old_motor_id

    data = [0xFF] * 8
    data[6] = new_motor_id & 0xFF  # New motor ID in byte 6
    data[7] = 0xFA
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
    bus.send(msg)
    msg = bus.recv(0.5)
    if msg:
        return True
    return False

def parse_data(data: bytes) -> Tuple[float, float, float, int]:
    """Decode MIT feedback frame to human-readable units."""
    motor_id = data[0]
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    t_mos = (data[6] << 8) | data[7]

    # Get motor limits
    pmax, vmax, tmax = 12.57, 50, 6

    # Decode to physical values (radians)
    position_rad = _uint_to_float(q_uint, -pmax, pmax, 16)
    velocity_rad_per_sec = _uint_to_float(dq_uint, -vmax, vmax, 12)
    torque = _uint_to_float(tau_uint, -tmax, tmax, 12)

    # Convert to degrees
    position_degrees = np.degrees(position_rad)
    velocity_deg_per_sec = np.degrees(velocity_rad_per_sec)

    return position_degrees, velocity_deg_per_sec, torque, t_mos


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Convert float to unsigned integer for CAN transmission."""
    x = max(x_min, min(x_max, x))  # Clamp to range
    span = x_max - x_min
    data_norm = (x - x_min) / span
    return int(data_norm * ((1 << bits) - 1))

def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    """Convert unsigned integer from CAN to float."""
    span = x_max - x_min
    data_norm = float(x) / ((1 << bits) - 1)
    return data_norm * span + x_min


def enable(bus: can.Bus, motor_id: int) -> bool:
    """
    Enable a motor using the MIT protocol.
    """

    arb_id = motor_id

    data = [0xFF] * 8
    data[7] = 0xFC  # Enable command

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    bus.send(msg)
    msg = bus.recv(0.02)
    if msg:
        return True
    return False


def disable(bus: can.Bus, motor_id: int) -> bool:
    """
    Disable a motor using the MIT protocol.
    """

    arb_id = motor_id

    data = [0xFF] * 8
    data[7] = 0xFD  # Disable command

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    bus.send(msg)
    msg = bus.recv(0.02)
    if msg:
        return True
    return False


def mit_control(
    bus: can.Bus,
    motor_id: int,
    kp: float,
    kd: float,
    position_degrees: float,
    velocity_deg_per_sec: float,
    torque: float,
    pmax: float = 12.57,
    vmax: float = 50,
    tmax: float = 6,
) -> Union[can.Message, bool]:
    """
    Send a MIT-style position/velocity/torque command to a motor.

    Args:
        motor_id: Target motor ID
        kp: Position gain
        kd: Velocity gain
        position_degrees: Target position (degrees)
        velocity_deg_per_sec: Target velocity (degrees/s)
        torque: Target torque (N·m)
    """
    position_rad = np.radians(position_degrees)
    velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

    kp_uint = _float_to_uint(kp, 0, 500, 12)
    kd_uint = _float_to_uint(kd, 0, 5, 12)
    q_uint = _float_to_uint(position_rad, -pmax, pmax, 16)
    dq_uint = _float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
    tau_uint = _float_to_uint(torque, -tmax, tmax, 12)

    # Pack data
    data = [0] * 8
    data[0] = (q_uint >> 8) & 0xFF
    data[1] = q_uint & 0xFF
    data[2] = dq_uint >> 4
    data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
    data[4] = kp_uint & 0xFF
    data[5] = kd_uint >> 4
    data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
    data[7] = tau_uint & 0xFF

    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    bus.send(msg)
    msg = bus.recv(0.02)
    if msg:
        return msg
    return False
