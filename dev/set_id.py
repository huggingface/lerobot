#!/usr/bin/env python3
"""
Motor ID and Baudrate Configuration Tool

This script provides functions to set motor ID and baudrate for Feetech motors.
"""

from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors.motors_bus import Motor, MotorNormMode, get_address


def set_motor_id_and_baudrate(port: str, motor_model: str, current_id: int, new_id: int, 
                               current_baudrate: int, new_baudrate: int):
    """Set motor ID and baudrate for a Feetech motor."""
    
    temp_motor = Motor(id=current_id, model=motor_model, norm_mode=MotorNormMode.RANGE_0_100)
    motors = {"temp_motor": temp_motor}
    
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus._connect(handshake=False)
    bus.set_baudrate(current_baudrate)
    bus._disable_torque(current_id, motor_model)
    
    # Set new ID
    addr, length = get_address(bus.model_ctrl_table, motor_model, "ID")
    bus._write(addr, length, current_id, new_id)
    
    # Set new baudrate
    addr, length = get_address(bus.model_ctrl_table, motor_model, "Baud_Rate")
    baudrate_value = bus.model_baudrate_table[motor_model][new_baudrate]
    bus._write(addr, length, new_id, baudrate_value)
    
    bus.set_baudrate(new_baudrate)
    bus.disconnect()


def scan_motors(port: str, baudrates: list[int]):
    """Scan for motors on the given port."""
    
    temp_motor = Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
    motors = {"temp_motor": temp_motor}
    found_motors = {}
    
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus._connect(handshake=False)
    
    for baudrate in baudrates:
        bus.set_baudrate(baudrate)
        id_model = bus.broadcast_ping()
        found_motors[baudrate] = id_model
    
    bus.disconnect()
    return found_motors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set motor ID and baudrate")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyACM0)")
    parser.add_argument("--scan", action="store_true", help="Scan for motors")
    parser.add_argument("--model", default="sts3215", help="Motor model")
    parser.add_argument("--current-id", type=int, help="Current motor ID")
    parser.add_argument("--new-id", type=int, help="New motor ID")
    parser.add_argument("--current-baudrate", type=int, help="Current baudrate")
    parser.add_argument("--new-baudrate", type=int, default=1_000_000, help="New baudrate")
    
    args = parser.parse_args()
    
    if args.scan:
        found = scan_motors(args.port, [1_000_000, 500_000, 250_000, 115_200, 57_600])
        print("Scan complete:", found)
    elif args.current_id is not None and args.new_id is not None:
        set_motor_id_and_baudrate(
            args.port,
            args.model,
            args.current_id,
            args.new_id,
            args.current_baudrate,
            args.new_baudrate
        )
        print("Success")
    else:
        print("Use --scan to scan for motors, or provide --current-id and --new-id to set motor parameters")
