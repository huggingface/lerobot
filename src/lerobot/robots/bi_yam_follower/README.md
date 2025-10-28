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

### Install Dependencies

Install LeRobot with Yam support:
```bash
pip install -e ".[yam]"
```

This will install:
- `portal` library for client-server communication
- `i2rt` library for low-level motor control of Yam arms

**Note**: If `i2rt` is not available on PyPI, you may need to install it separately from source:
```bash
pip install git+https://github.com/your-org/i2rt.git
# OR from a local clone:
# cd /path/to/i2rt && pip install -e .
```

## Running the System

### Step 1: Start the Server Processes

Before using LeRobot, you need to start the server processes that manage the Yam arms:

```bash
python -m lerobot.scripts.setup_bi_yam_servers
```

This will start 4 server processes:
- Right follower arm server: `localhost:1234`
- Left follower arm server: `localhost:1235`
- Right leader arm server: `localhost:5001`
- Left leader arm server: `localhost:5002`

Leave this running in a terminal window.

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

Each server process (started by `setup_bi_yam_servers.py`) runs a separate instance of the i2rt `minimum_gello.py` script in "follower" mode. This mode:
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
1. Make sure `setup_bi_yam_servers.py` is running
2. Check that all 4 servers started successfully
3. Verify the port numbers match in both scripts

### Slow Control Loop

If you see warnings about slow control frequency:
- This usually means the system is overloaded
- Try reducing camera resolution or FPS
- Check CPU usage and close unnecessary applications

## Advanced Usage

### Custom Server Ports

You can run the servers on different ports by modifying the `setup_bi_yam_servers.py` script or running the i2rt `minimum_gello.py` script directly for each arm:

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

