# Biwheel Base

This folder provides the two-wheel differential-drive base used by `xlerobot`.
It includes a shared kinematics layer plus two hardware choices:

- `biwheel_feetech` for Feetech servo-driven wheels
- `biwheel_odrive` for ODrive-controlled wheels

For Feetech wheel installation, refer to the
[documentation](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble_2wheel.html).
For the ODrive BLDC version, see [Hardware Installation](#hardware-installation).

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/assembled.png"
    alt="ODrive hoverboard wheels with XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

## Files

```text
src/lerobot/robots/xlerobot/sub_robots/biwheel_base/
|-- README.md
|-- __init__.py
|-- config_biwheel_base.py
|-- biwheel_base.py
|-- biwheel_feetech.py
`-- biwheel_odrive.py
```

## Variants

`biwheel_base.py` contains the shared differential-drive math and the common action/observation schema.
It is not the hardware-specific implementation most users should configure directly.

Supported user-facing base types:

- `biwheel_feetech`: Feetech wheel motors on a shared bus
- `biwheel_odrive`: ODrive velocity-controlled wheel motors on a dedicated connection
- `biwheel_base`: generic alias resolved by `xlerobot`; it uses Feetech by default, or ODrive if `driver: odrive`

## Observations and actions

The base uses the same keys for observations and actions:

- `x.vel`: linear velocity in m/s
- `theta.vel`: yaw velocity in deg/s

The shared kinematics layer also supports:

- `invert_left_motor` / `invert_right_motor` for motor wiring direction fixes
- `reverse_front_direction` to flip only forward/backward direction while preserving turning sign

## Configuration

Common configuration fields:

- `wheel_radius`: wheel radius in meters
- `wheel_base`: left-to-right wheel separation in meters
- `invert_left_motor`: invert left wheel direction
- `invert_right_motor`: invert right wheel direction
- `reverse_front_direction`: treat the opposite chassis side as the front

Feetech-only fields:

- `port`: serial device for the Feetech bus
- `base_motor_ids`: left/right motor IDs
- `max_wheel_raw`: raw wheel speed clamp
- `handshake_on_connect`: whether to handshake on connect
- `disable_torque_on_disconnect`: whether to disable torque on disconnect

ODrive-only fields:

- `odrive_serial`: optional ODrive serial number
- `axis_left`: ODrive axis index for the left wheel
- `axis_right`: ODrive axis index for the right wheel
- `odrive_timeout_s`: discovery timeout
- `disable_watchdog`: disable the ODrive watchdog on connect
- `request_closed_loop`: request closed-loop control on connect

## Shared bus behavior

`biwheel_feetech` supports shared-bus mode and is intended to be used with `xlerobot` shared bus assembly.
`biwheel_odrive` does NOT use shared buses and should stay on its own dedicated ODrive connection.

If you use the generic `biwheel_base` type inside `xlerobot`, the builder selects:

- Feetech when `driver` is omitted
- ODrive when `driver: odrive`

## Example configs

Base-only ODrive example:

```json
"base": {
  "type": "biwheel_odrive",
  "wheel_radius": 0.05,
  "wheel_base": 0.25,
  "reverse_front_direction": false,
  "invert_left_motor": true,
  "invert_right_motor": false,
  "axis_left": 1,
  "axis_right": 0
}
```

This is available in:

- `src/lerobot/robots/xlerobot/configs/base_only.json`

Shared-bus Feetech example:

```json
"base": {
  "type": "biwheel_feetech",
  "wheel_radius": 0.05,
  "wheel_base": 0.25,
  "reverse_front_direction": false,
  "base_motor_ids": [9, 10],
  "invert_left_motor": true,
  "invert_right_motor": false
}
```

This is available in:

- `src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_feetech.json`

## Example runs

Base-only keyboard teleop with ODrive:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_biwheel.json
```

Full xlerobot example with Feetech biwheel base:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_feetech.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_default_composite_lekiwi.json
```

## Notes

- `BiwheelFeetech` reads and writes wheel velocity through `Present_Velocity` and `Goal_Velocity`.
- `BiwheelODrive` uses encoder velocity feedback and writes `controller.input_vel` in turns per second.
- Feetech motors can be configured interactively through `setup_motors()`.
- ODrive connect configures velocity-control passthrough and can request closed-loop mode automatically.

## Hardware Installation

### Feetech Motors

For the Feetech wheel version, follow the
[XLeRobot assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble_2wheel.html).

### BLDC Motors

Use
[`hardware/odrive/odrive_motor_holder.step`](https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/odrive/odrive_motor_holder.step)
from [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) to
mount the BLDC wheel holders to the IKEA cart. Secure them with M8 screws.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/wheel_holder.png"
    alt="Hoverboard wheels holder"
    style="max-height: 420px; width: auto;"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/wheels.png"
    alt="Hoverboard wheels"
    style="max-height: 420px; width: auto;"
  />
</p>

If the chassis tilts, add washers on the front or rear between the IKEA wheel
shaft and the cart to level the opposite side. Choose the side based on the
driving direction you want to treat as the front.

### ODrive v3.6 Board

Use M3x15 mm nylon standoff screws to secure the ODrive v3.6 board inside the
enclosure defined by
[`hardware/odrive/odrive_cap.step`](https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/odrive/odrive_cap.step)
and
[`hardware/odrive/odrive_housing.step`](https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/odrive/odrive_housing.step)
in
[Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot).

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/board.png"
    alt="ODrive board"
    style="max-height: 420px; width: auto;"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/board_w_case.png"
    alt="ODrive board with case"
    style="max-height: 420px; width: auto;"
  />
</p>

Use M4 screws to secure the board case to the basket mesh through the mounting
holes on the back of the case cap.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/board_case_holes.png"
    alt="ODrive board with case"
    style="max-height: 420px; width: auto;"
  />
</p>

### Wiring Schematic

TODO

## ODrive Setup and Calibration (Hoverboard Motors)

This guide configures an ODrive for two hoverboard motors using hall sensors.

Reference material:
- https://stijnsprojects.github.io/Balancing-robot/
- https://github.com/BracketBotCapstone/quickstart/blob/main/setup/calibrate_drive.py

### 1. Install tooling and udev rules

```bash
pip install --upgrade odrive pygame
sudo apt install curl
sudo bash -c "curl https://cdn.odriverobotics.com/files/odrive-udev-rules.rules > /etc/udev/rules.d/91-odrive.rules && udevadm control --reload-rules && udevadm trigger"
sudo reboot
```

After reboot:

```bash
odrivetool
```

### 2. Apply motor, encoder, and controller config

Run the following in the `odrivetool` interactive shell:

```python
# Motor setup
odrv0.axis0.motor.config.pole_pairs = 15
odrv0.axis1.motor.config.pole_pairs = 15

odrv0.axis0.motor.config.resistance_calib_max_voltage = 4
odrv0.axis1.motor.config.resistance_calib_max_voltage = 4

odrv0.axis0.motor.config.requested_current_range = 25
odrv0.axis1.motor.config.requested_current_range = 25

odrv0.axis0.motor.config.current_control_bandwidth = 100
odrv0.axis1.motor.config.current_control_bandwidth = 100

odrv0.axis0.motor.config.torque_constant = 8.27 / 16
odrv0.axis1.motor.config.torque_constant = 8.27 / 16

odrv0.axis0.motor.config.calibration_current = 5
odrv0.axis1.motor.config.calibration_current = 5

# Encoder setup (hall sensors)
odrv0.axis0.encoder.config.mode = ENCODER_MODE_HALL
odrv0.axis1.encoder.config.mode = ENCODER_MODE_HALL

odrv0.axis0.encoder.config.cpr = 90
odrv0.axis1.encoder.config.cpr = 90

odrv0.axis0.encoder.config.calib_scan_distance = 150
odrv0.axis1.encoder.config.calib_scan_distance = 150

odrv0.axis0.encoder.config.bandwidth = 100
odrv0.axis1.encoder.config.bandwidth = 100

# Controller gains and limits
odrv0.axis0.controller.config.pos_gain = 1
odrv0.axis1.controller.config.pos_gain = 1

odrv0.axis0.controller.config.vel_gain = 0.02 * odrv0.axis0.motor.config.torque_constant * odrv0.axis0.encoder.config.cpr
odrv0.axis1.controller.config.vel_gain = 0.02 * odrv0.axis1.motor.config.torque_constant * odrv0.axis1.encoder.config.cpr

odrv0.axis0.controller.config.vel_integrator_gain = 0.1 * odrv0.axis0.motor.config.torque_constant * odrv0.axis0.encoder.config.cpr
odrv0.axis1.controller.config.vel_integrator_gain = 0.1 * odrv0.axis1.motor.config.torque_constant * odrv0.axis1.encoder.config.cpr

odrv0.axis0.controller.config.vel_limit = 10
odrv0.axis1.controller.config.vel_limit = 10

odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
odrv0.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
```

Save and reboot:

```python
odrv0.save_configuration()
odrv0.reboot()
```

### 3. Run full calibration

Reconnect with `odrivetool`, then run:

> Safety: During calibration, make sure both wheels do not touch the ground.

```python
odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
odrv0.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
```

### 4. Persist calibration

When calibration is successful, mark motor and encoder as pre-calibrated:

```python
odrv0.axis0.motor.config.pre_calibrated = True
odrv0.axis0.encoder.config.pre_calibrated = True
odrv0.axis1.motor.config.pre_calibrated = True
odrv0.axis1.encoder.config.pre_calibrated = True

odrv0.save_configuration()
odrv0.reboot()
```

### 5. Run `test_connection` and `odrive_controller`

From the repository root:

```bash
cd software/examples/odrive
python3 test_connection.py
```

After the connection test succeeds:

```bash
python3 odrive_controller.py
```

`odrive_controller.py` will prompt you to choose a mode:
- `1`: automatic basic motion test
- `2`: interactive command control
- `3`: pygame real-time keyboard control (recommended)
