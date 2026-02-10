# Bimanual Yam Arms with LeRobot

This guide explains how to use bimanual Yam arms with LeRobot for data collection.

## Overview

The bimanual Yam setup consists of:

- **2 Follower Arms**: Controlled by LeRobot to execute actions
- **2 Leader Arms**: With teaching handles for teleoperation
- **4 CAN Interfaces**: For communication with the arms

## Hardware Setup

### Required CAN Interfaces

You need to set up 4 CAN interfaces with the following names:

- `can_follower_r`: Right follower arm
- `can_follower_l`: Left follower arm
- `can_leader_r`: Right leader arm (with teaching handle)
- `can_leader_l`: Left leader arm (with teaching handle)

### CAN Interface Configuration

For details on setting up persistent CAN interface names, see:

- `i2rt/doc/set_persist_id_socket_can.md`

Make sure all CAN interfaces are UP and accessible:

```bash
ip link show can_follower_r
ip link show can_follower_l
ip link show can_leader_r
ip link show can_leader_l
```

### Initialize CAN Interfaces

Before starting the Yam arm servers, you need to initialize all CAN interfaces with the correct bitrate (1000000).

A Python script is provided to automatically detect and reset all CAN interfaces:

```bash
python src/lerobot/robots/bi_yam_follower/reset_can_interfaces.py
```

This script will:

- Detect all CAN interfaces on your system
- Reset each interface with bitrate 1000000
- Provide clear feedback about the configuration status

**Alternative: Manual reset with bash**

If you prefer a bash script, you can create this file and run it:

```bash
#!/bin/bash

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Function to reset a CAN interface
reset_can_interface() {
    local iface=$1
    echo "Resetting CAN interface: $iface"
    $SUDO ip link set "$iface" down
    $SUDO ip link set "$iface" up type can bitrate 1000000
}

# Get all CAN interfaces
can_interfaces=$(ip link show | grep -oP '(?<=: )(can\w+)')

# Check if any CAN interfaces were found
if [[ -z "$can_interfaces" ]]; then
    echo "No CAN interfaces found."
    exit 1
fi

# Reset each CAN interface
echo "Detected CAN interfaces: $can_interfaces"
for iface in $can_interfaces; do
    reset_can_interface "$iface"
done

echo "All CAN interfaces have been reset with bitrate 1000000."
```

Save as `reset_can.sh`, make it executable with `chmod +x reset_can.sh`, and run with `./reset_can.sh`.

## Software Setup

### Platform Support

**Note:** Bimanual Yam arms require Linux for hardware operation due to:

- CAN interface support (SocketCAN on Linux)
- Hardware control libraries designed for Linux

### Install Dependencies

**Install LeRobot with Yam support:**

For Yam arms with Intel RealSense camera support (recommended):

```bash
pip install -e ".[yam,intelrealsense]"
```

For basic Yam arm functionality with OpenCV cameras:

```bash
pip install -e ".[yam]"
```

This will install:

- `portal` - RPC framework for client-server communication (Linux only)
- `i2rt` - Robotics library providing the server infrastructure ([i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt))
- `pyrealsense2` - Intel RealSense camera support (if using intelrealsense extra)

**Verify installation:**

```bash
python -c "import i2rt, portal; print('Dependencies OK')"
```

## Running the System

### Step 1: Start the Unified Server

The easiest way to start all 4 arm servers is using the unified server script. In one terminal:

```bash
python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py
```

This single command starts all 4 servers (2 followers + 2 leaders) with default settings:

- Right follower: port 1234 (CAN: `can_follower_r`)
- Left follower: port 1235 (CAN: `can_follower_l`)
- Right leader: port 5001 (CAN: `can_leader_r`)
- Left leader: port 5002 (CAN: `can_leader_l`)

**Customize CAN interfaces and ports:**

```bash
python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py \
  --right_follower_can can0 \
  --left_follower_can can1 \
  --right_leader_can can2 \
  --left_leader_can can3
```

**Run follower-only mode (without teaching handles):**

```bash
python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py \
  --mode follower_only
```

Leave this terminal running while recording data.

### Step 2: Testing and Setup

#### Step 2.1: Test Teleoperator (In another terminal)

Before recording, test that the teleoperator connection works:

```bash
lerobot-teleoperate \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --display_data=true
```

#### Step 2.2: Find Camera

Identify available cameras on your system:

```bash
lerobot-find-cameras realsense
```

or for OpenCV cameras:

```bash
lerobot-find-cameras opencv
```

For the following steps, adjust the camera `index_or_path` based on the output from this command.

#### Step 2.3: Test Camera + Teleoperator

Note: Replace the `index_or_path` values with the camera indices found in the previous step. If you have different cameras or want to use different settings, adjust the camera configuration accordingly.

**Using Intel RealSense cameras:**

```bash
lerobot-teleoperate \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --display_data=true \
  --robot.cameras='{
    left: {"type": "intelrealsense", "serial_number_or_name": "335122271633", "width": 1280, "height": 720, "fps": 30},
    right: {"type": "intelrealsense", "serial_number_or_name": "323622271837", "width": 1280, "height": 720, "fps": 30}
  }'
```

**Using OpenCV cameras:**

```bash
lerobot-teleoperate \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --display_data=true \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 2, "width": 1280, "height": 720, "fps": 30},
    right: {"type": "opencv", "index_or_path": 10, "width": 1280, "height": 720, "fps": 30}
  }'
```

#### Step 2.4: Login to HuggingFace

Before recording, log in to HuggingFace:

```bash
uvx hf auth login
```

### Step 3: Record Data with LeRobot

In a new terminal, use `lerobot-record` to collect data:

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    left: {"type": "intelrealsense", "serial_number_or_name": "335122271633", "width": 1280, "height": 720, "fps": 30},
    right: {"type": "intelrealsense", "serial_number_or_name": "323622271837", "width": 1280, "height": 720, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/bimanual-yam-demo \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick and place the object" \
  --display_data=true
```

### Configuration Parameters

#### Robot Configuration (`bi_yam_follower`)

- `robot.type`: Set to `bi_yam_follower`
- `robot.left_arm_port`: Server port for left follower arm (default: 1235)
- `robot.right_arm_port`: Server port for right follower arm (default: 1234)
- `robot.server_host`: Server hostname (default: "localhost")
- `robot.cameras`: Camera configurations (same as other robots)
- `robot.left_arm_max_relative_target`: Optional safety limit for left arm
- `robot.right_arm_max_relative_target`: Optional safety limit for right arm

#### Teleoperator Configuration (`bi_yam_leader`)

- `teleop.type`: Set to `bi_yam_leader`
- `teleop.left_arm_port`: Server port for left leader arm (default: 5002)
- `teleop.right_arm_port`: Server port for right leader arm (default: 5001)
- `teleop.server_host`: Server hostname (default: "localhost")

## Gripper Control with Teaching Handles

The teaching handles don't have physical grippers, but they have an **encoder button** (or digital input) that is used to command the follower gripper:

- **Press the encoder button**: Toggles the gripper between fully closed (0.0) and fully open (1.0)
- The leader handle input is read by the `bi_yam_leader` server and exposed as a binary gripper state
- The follower grippers mirror this open/close command in real-time
- Note: Currently only binary control is supported (no intermediate continuous positions)

## Architecture

### Data Flow

```
┌─────────────────┐         ┌─────────────────┐
│  Leader Arms    │         │  Follower Arms  │
│  (Teaching      │         │  (Execution)    │
│   Handles)      │         │                 │
└────────┬────────┘         └────────▲────────┘
         │                           │
         │ Read State                │ Send Actions
         │                           │
    ┌────▼────┐              ┌───────┴─────┐
    │ Leader  │              │  Follower   │
    │ Servers │              │  Servers    │
    │ (5001,  │              │  (1234,     │
    │  5002)  │              │   1235)     │
    └────┬────┘              └───────▲─────┘
         │                           │
         │                           │
    ┌────▼───────────────────────────┴──────┐
    │         LeRobot Recording             │
    │  - bi_yam_leader (teleoperator)       │
    │  - bi_yam_follower (robot)            │
    │  - Cameras                            │
    │  - Dataset writer                     │
    └───────────────────────────────────────┘
```

### Server Process Details

The bimanual Yam setup uses server processes that wrap i2rt functionality:

**Unified Server (`run_bimanual_yam_server.py`)**:

- Starts all 4 arms (2 followers + 2 leaders) in a single process
- Easiest setup - just one command
- Recommended for most users

**Individual Server (`run_yam_server.py`)**:

- Runs a single arm server
- Useful for debugging or custom setups
- Requires running 4 separate instances

Each server:

1. Connects to a Yam arm via CAN
2. Provides gravity compensation
3. Exposes the robot state via a portal RPC server
4. Accepts position commands (for follower arms) or reads state (for leader arms)

## Troubleshooting

### CAN Interface Issues

If you get errors about missing CAN interfaces:

```bash
# Check if interfaces exist
ip link show | grep can

# Bring up an interface if it's down
sudo ip link set can_follower_r up
```

### Port Already in Use

If you get "address already in use" errors:

```bash
# Find and kill processes using the ports
lsof -ti:1234 | xargs kill -9
lsof -ti:1235 | xargs kill -9
lsof -ti:5001 | xargs kill -9
lsof -ti:5002 | xargs kill -9
```

### Connection Timeouts

If LeRobot can't connect to the servers:

1. Make sure all 4 i2rt server processes are running
2. Check that the servers started successfully without errors
3. Verify the port numbers match in both scripts

### Slow Control Loop

If you see warnings about slow control frequency:

- This usually means the system is overloaded
- Try reducing camera resolution or FPS
- Check CPU usage and close unnecessary applications

## Advanced Usage

### Running Individual Server Processes

If you need more control or want to run servers separately, you can use the individual server script:

```bash
# Terminal 1: Right follower
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --can_channel can_follower_r \
  --gripper v3 \
  --mode follower \
  --server_port 1234

# Terminal 2: Left follower
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --can_channel can_follower_l \
  --gripper v3 \
  --mode follower \
  --server_port 1235

# Terminal 3: Right leader
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --can_channel can_leader_r \
  --gripper yam_teaching_handle \
  --mode leader \
  --server_port 5001

# Terminal 4: Left leader
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --can_channel can_leader_l \
  --gripper yam_teaching_handle \
  --mode leader \
  --server_port 5002
```

### Visualizer Mode

You can visualize arm movements using MuJoCo:

```bash
# Local visualization (directly connected to hardware)
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --can_channel can_follower_r \
  --gripper v3 \
  --mode visualizer_local

# Remote visualization (connect to running server)
python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
  --server_host localhost \
  --server_port 1234 \
  --mode visualizer_remote
```

### Without Teleoperation

You can also use the bimanual Yam follower arms with a trained policy (without teleop):

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{...}' \
  --policy.path=path/to/trained/policy \
  --dataset.repo_id=${HF_USER}/bimanual-yam-eval \
  --dataset.num_episodes=5
```

## References

- **i2rt library**: Python library for controlling Yam arm hardware (install via `pip install -e '.[yam]'`)
  - GitHub: [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt)
  - Used internally by `run_yam_server.py` for hardware communication
- **portal**: RPC framework for client-server communication (installed with yam dependencies)
- **LeRobot documentation**: See main docs for training and evaluation workflows
