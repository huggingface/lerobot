# Damiao Motors and CAN Bus

This guide covers setup and usage of Damiao motors with LeRobot via CAN bus communication.

Currently, only Linux is supported, as the OpenArms CAN adapter only has drivers for Linux.

## Linux CAN Setup

Before using Damiao motors, you need to set up the CAN interface on your Linux system.

### Install CAN Utilities

```bash
sudo apt-get install can-utils
```

### Configure CAN Interface (Manual)

For standard CAN FD (recommended for OpenArms):

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

For standard CAN (without FD):

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### Configure CAN Interface (Using LeRobot)

LeRobot provides a utility script to setup and test CAN interfaces:

```bash
# Setup multiple interfaces (e.g., OpenArms Followers with 2 CAN buses)
lerobot-setup-can --mode=setup --interfaces=can0,can1
```

## Debugging CAN Communication

Use the built-in debug tools to test motor communication:

```bash
# Test motors on all interfaces
lerobot-setup-can --mode=test --interfaces=can0,can1

# Run speed/latency test
lerobot-setup-can --mode=speed --interfaces=can0
```

The test mode will scan for motors (IDs 0x01-0x08) and report which ones respond. Example output:

```
can0: UP (CAN FD)
  Motor 0x01 (joint_1): ✓ FOUND
    → Response 0x11 [FD]: 00112233...
  Motor 0x02 (joint_2): ✓ FOUND
  Motor 0x03 (joint_3): ✗ No response
  ...
  Summary: 2/8 motors found
```

## Usage

### Basic Setup

```python
from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus

# Define your motors with send/receive CAN IDs
motors = {
    "joint_1": Motor(id=0x01, motor_type_str="dm8009", recv_id=0x11),
    "joint_2": Motor(id=0x02, motor_type_str="dm4340", recv_id=0x12),
    "joint_3": Motor(id=0x03, motor_type_str="dm4310", recv_id=0x13),
}

# Create the bus
bus = DamiaoMotorsBus(
    port="can0",  # Linux socketcan interface
    motors=motors,
)

# Connect
bus.connect()
```

### Reading Motor States

```python
# Read single motor position (degrees)
position = bus.read("Present_Position", "joint_1")

# Read from multiple motors
positions = bus.sync_read("Present_Position")  # All motors
positions = bus.sync_read("Present_Position", ["joint_1", "joint_2"])

# Read all states at once (position, velocity, torque)
states = bus.sync_read_all_states()
# Returns: {'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5}, ...}
```

### Writing Motor Commands

```python
# Enable torque
bus.enable_torque()

# Set goal position (degrees)
bus.write("Goal_Position", "joint_1", 45.0)

# Set positions for multiple motors
bus.sync_write("Goal_Position", {
    "joint_1": 45.0,
    "joint_2": -30.0,
    "joint_3": 90.0,
})

# Disable torque
bus.disable_torque()
```

## Configuration Options

| Parameter      | Default   | Description                                                 |
| -------------- | --------- | ----------------------------------------------------------- |
| `port`         | -         | CAN interface (`can0`) or serial port (`/dev/cu.usbmodem*`) |
| `use_can_fd`   | `True`    | Enable CAN FD for higher data rates                         |
| `bitrate`      | `1000000` | Nominal bitrate (1 Mbps)                                    |
| `data_bitrate` | `5000000` | CAN FD data bitrate (5 Mbps)                                |

## Motor Configuration

Each motor requires:

- `id`: CAN ID for sending commands
- `motor_type`: One of the supported motor types (e.g., `"dm8009"`, `"dm4340"`)
- `recv_id`: CAN ID for receiving responses

OpenArms default IDs follow the pattern: send ID `0x0N`, receive ID `0x1N` where N is the joint number.

## Troubleshooting

### No Response from Motors

1. **Check power**
2. **Verify CAN wiring**: Check CAN-H, CAN-L, and GND connections
3. **Check motor IDs**: Use Damiao Debugging Tools to verify/configure IDs
4. **Test CAN interface**: Run `candump can0` to see if messages are being received
5. **Run diagnostics**: `lerobot-setup-can --mode=test --interfaces=can0`

### Motor Timeout Parameter

If motors were configured with timeout=0, they won't respond to commands. Use Damiao Debugging Tools to set a non-zero timeout value.

### Verify CAN FD Status

```bash
ip -d link show can0 | grep fd
```
