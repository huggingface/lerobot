# ODrive Setup and Calibration (Hoverboard Motors)

This guide configures an ODrive for two hoverboard motors using hall sensors.

Reference material:
- https://stijnsprojects.github.io/Balancing-robot/
- https://github.com/BracketBotCapstone/quickstart/blob/main/setup/calibrate_drive.py

## 1. Install tooling and udev rules

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

## 2. Apply motor, encoder, and controller config

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

## 3. Run full calibration

Reconnect with `odrivetool`, then run:

> Safety: During calibration, make sure both wheels do not touch the ground.

```python
odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
odrv0.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
```

## 4. Persist calibration

When calibration is successful, mark motor and encoder as pre-calibrated:

```python
odrv0.axis0.motor.config.pre_calibrated = True
odrv0.axis0.encoder.config.pre_calibrated = True
odrv0.axis1.motor.config.pre_calibrated = True
odrv0.axis1.encoder.config.pre_calibrated = True

odrv0.save_configuration()
odrv0.reboot()
```

## 5. Run `test_connection` and `odrive_controller`

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
