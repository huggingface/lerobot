# Simple Motor Control Interfaces

This directory contains simple interfaces for controlling Feetech motors with basic left/right movement commands.

## Files

- `simple_motor_control.py` - Real-time keyboard control (requires raw terminal input)
- `motor_control_simple.py` - Command-based control (uses input prompts - **recommended**)

## Setup

### 1. Configure Your Motors

Before running either script, you need to modify the motor configuration in the `main()` function:

```python
# CONFIGURATION - Modify these settings for your setup
PORT = "/dev/ttyUSB0"  # Change this to your motor port

MOTORS = {
    "motor1": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    # Add more motors as needed:
    # "motor2": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}
```

### 2. Find Your Port

To find the correct port for your motors, you can use:

```bash
python -m lerobot.find_port
```

Or check available ports with:
```bash
ls /dev/tty*
```

Common ports:
- `/dev/ttyUSB0`, `/dev/ttyUSB1` (Linux)
- `/dev/tty.usbmodem*` (macOS)
- `COM1`, `COM2`, etc. (Windows)

## Usage

### Recommended: Command-based Interface

```bash
python dev/motor_control_simple.py
```

**Controls:**
- `a` or `left` - Move motors left (decrease position)
- `d` or `right` - Move motors right (increase position)
- `s` or `status` - Show current motor positions
- `step` - Change movement step size
- `goto` - Move to specific position
- `h` or `help` - Show help
- `q` or `quit` - Quit

### Alternative: Real-time Keyboard Interface

```bash
python dev/simple_motor_control.py
```

**Controls:**
- `a` - Move left
- `d` - Move right
- `s` - Show positions
- `+`/`-` - Adjust step size
- `q` - Quit

## Motor Configuration Examples

### Single Motor
```python
MOTORS = {
    "base": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}
```

### Multiple Motors
```python
MOTORS = {
    "shoulder": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "elbow": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "wrist": Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}
```

### Supported Motor Models

Based on the Feetech implementation, supported models include:
- `sts3215`
- `sts3032`
- `scscl`
- And others defined in the Feetech tables

### Normalization Modes

Available normalization modes:
- `MotorNormMode.RANGE_0_100` - 0 to 100 range
- `MotorNormMode.RANGE_M100_100` - -100 to 100 range  
- `MotorNormMode.DEGREES` - Degree-based positioning

## Safety Notes

⚠️ **Important Safety Considerations:**

1. **Start with small step sizes** - Default is 50 motor steps
2. **Know your motor limits** - Check min/max position limits
3. **Emergency stop** - Use Ctrl+C to stop the program
4. **Power off** - Turn off motor power if something goes wrong
5. **Test carefully** - Start with single motor before using multiple motors

## Troubleshooting

### Connection Issues
- Check if the correct port is specified
- Ensure motor power is on
- Verify motor ID matches configuration
- Check USB cable connection

### Permission Issues (Linux)
```bash
sudo chmod 666 /dev/ttyUSB0
# or add user to dialout group:
sudo usermod -a -G dialout $USER
```

### Motor Not Responding
- Check motor ID with `bus.broadcast_ping()`
- Verify baud rate settings
- Ensure motor firmware is compatible

## How It Works

The interfaces use the LeRobot Feetech motor bus implementation:

1. **Initialize** - Create `FeetechMotorsBus` with port and motor configuration
2. **Connect** - Establish serial communication with motors
3. **Configure** - Set up motor parameters (delays, acceleration, etc.)
4. **Enable Torque** - Allow motors to move
5. **Control Loop** - Read current positions and write new goal positions
6. **Cleanup** - Disable torque and disconnect safely

### Key Operations

- **Read Position**: `bus.sync_read("Present_Position", normalize=False)`
- **Write Position**: `bus.sync_write("Goal_Position", new_positions, normalize=False)`
- **Enable/Disable**: `bus.enable_torque()` / `bus.disable_torque()`

The interfaces move all configured motors simultaneously by the same step size. For individual motor control, use the `goto` command in the command-based interface. 