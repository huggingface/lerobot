# ROS2 Distributed Teleoperation

This tutorial explains how to set up a distributed teleoperation system using ROS2, allowing a leader robot on one computer to control a follower robot on another computer over a network.

## Prerequisites

1. ROS2 (Foxy or newer) installed on both computers
2. Network connectivity between the leader and follower computers
3. lerobot package installed on both computers

## Installation

Ensure you have the required ROS2 packages for sensor messages:

```bash
pip install sensor_msgs
```

## Setup

### 1. Configure ROS2 Domain ID (optional but recommended)

To isolate your ROS2 network from others, set the same domain ID on both computers:

```bash
# On both leader and follower computers
export ROS_DOMAIN_ID=42  # Choose any number between 0-232
```

### 2. Network Configuration

Ensure both computers can communicate over the network:

1. Connect both computers to the same network
2. Verify they can ping each other
3. Make sure any firewalls allow ROS2 communication (default ports)

## Running the Distributed Teleoperation

### 1. Leader Robot Computer

On the computer connected to the leader robot:

```bash
python lerobot/scripts/run_ros2_leader.py \
  --robot.type=koch \
  --robot.leader_arms.main.port=/dev/ttyUSB0 \
  --ros2_config.node_name=lerobot_teleop \
  --ros2_config.topic_name=joint_states \
  --ros2_config.publish_rate_hz=200.0
```

Replace `/dev/ttyUSB0` with the actual port of your leader robot.

### 2. Follower Robot Computer

On the computer connected to the follower robot:

```bash
python lerobot/scripts/run_ros2_follower.py \
  --robot.type=koch \
  --robot.follower_arms.main.port=/dev/ttyUSB0 \
  --ros2_config.node_name=lerobot_teleop \
  --ros2_config.topic_name=joint_states
```

Replace `/dev/ttyUSB0` with the actual port of your follower robot.

### Note on Topic Names

Make sure both leader and follower use the same topic name (default is `joint_states`).

## Troubleshooting

### 1. Check ROS2 Discovery

Verify nodes can discover each other:

```bash
# On either computer
ros2 topic list
```

You should see `/joint_states` in the list.

### 2. Monitor Message Flow

Check if messages are being published:

```bash
ros2 topic echo /joint_states
```

### 3. Network Issues

If nodes can't discover each other:
- Ensure both computers are on the same network
- Check if ROS_DOMAIN_ID is set to the same value on both computers
- Verify firewall settings
- Try using the ROS_LOCALHOST_ONLY=0 environment variable

### 4. Debug Node Status

```bash
ros2 node list
ros2 node info /lerobot_teleop_leader  # or follower
```

## Advanced Configuration

### Quality of Service Settings

For teleoperation over unreliable networks, you can adjust QoS settings:

```bash
# Use reliable QoS instead of best effort (might increase latency but improve reliability)
--ros2_config.use_best_effort_qos=false
```

### Publishing Rate

Adjust the rate at which joint states are published:

```bash
# On leader robot, set a lower rate for unreliable networks
--ros2_config.publish_rate_hz=100.0
```

## Examples

### Using with myCobot 280

For MyCobot robots, ensure you've set up the correct motor types:

Leader:
```bash
python lerobot/scripts/run_ros2_leader.py \
  --robot.type=mycobot \
  --robot.leader_arms.main.port=/dev/ttyUSB0 \
  --ros2_config.publish_rate_hz=100.0
```

Follower:
```bash
python lerobot/scripts/run_ros2_follower.py \
  --robot.type=mycobot \
  --robot.follower_arms.main.port=/dev/ttyUSB0
```
