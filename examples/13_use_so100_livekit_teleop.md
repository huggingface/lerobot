# LeRobot Remote Teleoperation with LiveKit

This guide explains how to use LiveKit to enable LeRobot teleoperation over the internet between any devices.

## Overview

The Livekit integration allows you to operate a robot remotely where:
- **Leader side**: Controls the robot by reading positions from leader arms and streaming them over Livekit, display any cameras from the follower side.
- **Follower side**: Receives commands via LiveKit and executes them on the follower robot, publish camera streams to LiveKit.

This setup enables:
- Remote robot tele-operation anywhere in the world.
- WebRTC-based real-time communication which automatically handles NAT traversal with zero effort.
- Lowest latency possible when using LiveKit Cloud.

## Prerequisites

1. **Livekit Server**: You need access to a LiveKit server, either cloud or self hosted.
   - Cloud service: [Livekit Cloud](https://cloud.livekit.io/)
   - Self-hosted: [Livekit Server](https://github.com/livekit/livekit)

2. **Python Dependencies**: Ensure you have the Livekit Python SDK installed your Python environment.
   ```bash
   pip install livekit
   ```

3. **Robot Hardware**: Currently compatible with any robot config that's based off the Manipulator class.

## Setup

### 1. Livekit Server Configuration

If using Livekit Cloud:
1. Sign up at [cloud.livekit.io](https://cloud.livekit.io/)
2. Create a new project
3. Note your server URL and API credentials

If self-hosting:
1. Follow the [Running LiveKit locally](https://docs.livekit.io/home/self-hosting/local/)
2. Start your local LiveKit development server

### 2. Generate Livekit Tokens

You need to generate separate tokens for the leader and follower robots. To do so, you will need [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup/).

```bash
# Install Livekit CLI
curl -sSL https://get.livekit.io/cli | bash

# Generate tokens (replace with your actual API key, secret, room name)
lk token create \
  --api-key <api key> \
  --api-secret <api secret> \
  --room <room name> \
  --identity leader \
  --valid-for 7200h

lk token create \
  --api-key apikey \
  --api-secret <api secret> \
  --room <room name> \
  --identity follower \
  --valid-for 7200h
```

### 3. Environment Configuration

Create environment files for your LiveKit url & credentials:

**`.env`**:
```bash
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_LEADER_TOKEN=livekit-token-for-leader
LIVEKIT_FOLLOWER_TOKEN=livekit-token-for-follower
```

**Important**: Place these files in the root directory of your lerobot workspace.

## Usage

### Configuring your robot

In `lerobot/common/robot_devices/robots/configs.py` there are examples for regular and bimanual versions of leader & follower configs.  You will need to modify them the same way with the correct serial port & calibration routine.

**Run the leader**:
   ```bash
   python lerobot/scripts/control_robot.py \
       --robot.type=so100bimanual_livekit_leader \
       --control.type=teleoperate --control.fps=30
   ```

**Run the follower**:
   ```bash
   python lerobot/scripts/control_robot.py \
       --robot.type=so100bimanual_livekit_follower \
       --control.type=teleoperate --control.fps=1

       # note: the control.fps here is not important, the follower will send the commands to the robot as fast as it receives from the leader.
   ```

## Robot Configuration Details

### Pre-configured Robot Types

The system includes pre-configured robot types for SO100 bimanual setups:

- `so100bimanual_livekit_leader`: For the leader side (no cameras, only leader arms)
- `so100bimanual_livekit_follower`: For the follower side (includes cameras and follower arms)

### Custom Configuration

You can create custom configurations by extending the `LivekitManipulatorRobotConfig`:

```python
from lerobot.common.robot_devices.robots.configs import LivekitManipulatorRobotConfig

@RobotConfig.register_subclass("my_custom_livekit_robot")
@dataclass
class MyCustomLivekitRobotConfig(LivekitManipulatorRobotConfig):
    # Your custom configuration here
    is_leader: bool = False

    # Override leader_arms, follower_arms, cameras as needed
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})
```
