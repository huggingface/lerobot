# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script configure a single motor at a time to a given ID and baudrate.

Example of usage:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem585A0080521 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --id 1
```
"""

import argparse
import time

from lerobot.common.motors.dynamixel.dynamixel import MODEL_RESOLUTION as DXL_MODEL_RESOLUTION
from lerobot.common.motors.feetech.feetech import MODEL_RESOLUTION as FTCH_MODEL_RESOLUTION


def configure_motor(port, brand, model, target_motor_idx, target_baudrate):
    if brand == "feetech":
        from lerobot.common.motors.feetech.feetech import FeetechMotorsBus

        motor_bus = FeetechMotorsBus(port=port, motors={"motor": (target_motor_idx, model)})

    elif brand == "dynamixel":
        from lerobot.common.motors.dynamixel.dynamixel import DynamixelMotorsBus

        motor_bus = DynamixelMotorsBus(port=port, motors={"motor": (target_motor_idx, model)})

    motor_bus.connect()

    # Motor bus is connected, proceed with the rest of the operations
    try:
        print("Scanning all baudrates and motor indices")
        model_baudrates = list(motor_bus.model_baudrate_table[model].values())
        motor_index = -1  # Set the motor index to an out-of-range value.

        for baudrate in model_baudrates:
            motor_bus.set_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices(list(range(1, 10)))
            if len(present_ids) > 1:
                raise ValueError(
                    "Error: More than one motor ID detected. This script is designed to only handle one motor at a time. Please disconnect all but one motor."
                )

            if len(present_ids) == 1:
                if motor_index != -1:
                    raise ValueError(
                        "Error: More than one motor ID detected. This script is designed to only handle one motor at a time. Please disconnect all but one motor."
                    )
                motor_index = present_ids[0]
                break

        if motor_index == -1:
            raise ValueError("No motors detected. Please ensure you have one motor connected.")

        print(f"Motor index found at: {motor_index}")

        if brand == "feetech":
            # Allows ID and BAUDRATE to be written in memory
            motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Lock", 0)

        if baudrate != target_baudrate:
            print(f"Setting its baudrate to {target_baudrate}")
            baudrate_idx = model_baudrates.index(target_baudrate)

            # The write can fail, so we allow retries
            motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Baud_Rate", baudrate_idx)
            time.sleep(0.5)
            motor_bus.set_bus_baudrate(target_baudrate)
            present_baudrate_idx = motor_bus.read_with_motor_ids(
                motor_bus.motor_models, motor_index, "Baud_Rate", num_retry=2
            )

            if present_baudrate_idx != baudrate_idx:
                raise OSError("Failed to write baudrate.")

        print(f"Setting its index to desired index {target_motor_idx}")
        if brand == "feetech":
            motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "Lock", 0)
        motor_bus.write_with_motor_ids(motor_bus.motor_models, motor_index, "ID", target_motor_idx)

        present_idx = motor_bus.read_with_motor_ids(
            motor_bus.motor_models, target_motor_idx, "ID", num_retry=2
        )
        if present_idx != target_motor_idx:
            raise OSError("Failed to write index.")

        if brand == "feetech":
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            motor_bus.write("Lock", 0)
            motor_bus.write("Maximum_Acceleration", 254)
            motor_bus.write("Max_Angle_Limit", 4095)  # default 4095
            motor_bus.write("Min_Angle_Limit", 0)  # default 0
            motor_bus.write("Offset", 0)
            motor_bus.write("Mode", 0)
            motor_bus.write("Goal_Position", 2048)
            motor_bus.write("Lock", 1)
            print("Offset", motor_bus.read("Offset"))

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        if motor_bus.is_connected:
            motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    model_choices = [*FTCH_MODEL_RESOLUTION.keys(), *DXL_MODEL_RESOLUTION.keys()]
    brand_choices = ["feetech", "dynamixel"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port")
    parser.add_argument("--brand", type=str, required=True, choices=brand_choices, help="Motor brand")
    parser.add_argument("--model", type=str, required=True, choices=model_choices, help="Motor model")
    parser.add_argument("--id", type=int, required=True, help="Desired ID of the current motor (e.g. 1,2,3)")
    parser.add_argument(
        "--baudrate", type=int, default=1_000_000, help="Desired baudrate for the motor (default: 1_000_000)"
    )
    args = parser.parse_args()

    configure_motor(args.port, args.brand, args.model, args.id, args.baudrate)
