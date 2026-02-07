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

## Software Setup

### Platform Support

**Note:** Bimanual Yam arms require Linux for hardware operation due to:

- CAN interface support (SocketCAN on Linux)
- Hardware control libraries designed for Linux

### Install Dependencies

**Install LeRobot with Yam support:**

```bash
pip install -e ".[yam]"
```

This will install:

- `portal` - RPC framework for client-server communication (Linux only)
- `i2rt` - Robotics library providing the server infrastructure ([i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt))

**Verify installation:**

```bash
python -c "import i2rt, portal; print('Dependencies OK')"
```

## Running the System

### Step 1: Start the Server Processes

Before using LeRobot, you need to start server processes for each arm using i2rt. Run these commands in separate terminals:

**Right Follower Arm:**

```bash
python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_r \
  --gripper v3 \
  --mode follower \
  --server_port 1234
```

**Left Follower Arm:**

```bash
python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_l \
  --gripper v3 \
  --mode follower \
  --server_port 1235
```

**Right Leader Arm (Teaching Handle):**

```bash
python -m i2rt.scripts.minimum_gello \
  --can_channel can_leader_r \
  --gripper v3 \
  --mode leader \
  --server_port 5001
```

**Left Leader Arm (Teaching Handle):**

```bash
python -m i2rt.scripts.minimum_gello \
  --can_channel can_leader_l \
  --gripper v3 \
  --mode leader \
  --server_port 5002
```

These servers provide portal RPC interfaces on the specified ports:

- Follower servers (1234, 1235): Accept position commands from LeRobot
- Leader servers (5001, 5002): Provide teaching handle positions to LeRobot

Leave all 4 terminal windows running.

### Step 2: Record Data with LeRobot

In a new terminal, use `lerobot-record` to collect data:

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
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

The teaching handles don't have physical grippers, but they have an **encoder knob** that controls the follower gripper:

- **Turn the encoder knob**: Controls gripper position (0 = closed, 1 = open)
- The encoder position is read by the i2rt `minimum_gello.py` server when running in "leader" mode
- The follower grippers will mirror your encoder movements in real-time

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
    ┌────▼────────────────────────────┴─────┐
    │         LeRobot Recording              │
    │  - bi_yam_leader (teleoperator)       │
    │  - bi_yam_follower (robot)            │
    │  - Cameras                             │
    │  - Dataset writer                      │
    └───────────────────────────────────────┘
```

### Server Process Details

Each server process runs an instance of the i2rt `minimum_gello.py` script. This script:

1. Connects to the Yam arm via CAN
2. Provides gravity compensation
3. Exposes the robot state via a portal RPC server
4. Accepts position commands (for follower arms) or just reads state (for leader arms)

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

### Custom Server Ports

You can run the servers on different ports by changing the `--server_port` argument when running the i2rt `minimum_gello.py` script for each arm:

```bash
# Find the i2rt script location
python -c "import i2rt, os; print(os.path.dirname(i2rt.__file__))"

# Right follower
python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_r \
  --gripper linear_4310 \
  --mode follower \
  --server_port 1234 &

# Left follower
python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_l \
  --gripper linear_4310 \
  --mode follower \
  --server_port 1235 &

# Right leader
python -m i2rt.scripts.minimum_gello \
  --can_channel can_leader_r \
  --gripper yam_teaching_handle \
  --mode follower \
  --server_port 5001 &

# Left leader
python -m i2rt.scripts.minimum_gello \
  --can_channel can_leader_l \
  --gripper yam_teaching_handle \
  --mode follower \
  --server_port 5002 &
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

- i2rt library: Python library for controlling Yam arm hardware (install via `pip install -e '.[yam]'`)
- LeRobot documentation: See main docs for training and evaluation workflows
