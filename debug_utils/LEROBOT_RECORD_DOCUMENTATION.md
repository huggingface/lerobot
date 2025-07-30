# LeRobot Record: Complete Inference & Control Documentation

## 📋 **Table of Contents**
1. [Basic Usage](#basic-usage)
2. [Core Arguments](#core-arguments)
3. [Policy Configuration](#policy-configuration)
4. [Dataset Configuration](#dataset-configuration)
5. [Robot Configuration](#robot-configuration)
6. [Teleoperator Configuration](#teleoperator-configuration)
7. [Inference Logging](#inference-logging)
8. [Real-time Visualization](#real-time-visualization)
9. [Interactive Controls](#interactive-controls)
10. [Performance Optimization](#performance-optimization)
11. [Advanced Features](#advanced-features)
12. [Examples](#examples)

---

## **Basic Usage**

```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760434091 \
  --dataset.repo_id=username/dataset_name \
  --dataset.single_task="Task description" \
  --policy.path=username/model_name
```

---

## **Core Arguments**

### **Control Modes**
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--policy.path` | `str` | Load pretrained policy for autonomous control | `None` |
| `--teleop.type` | `str` | Enable teleoperation (`so100_leader`, `gamepad`, `keyboard`) | `None` |
| `--display_data` | `bool` | Enable real-time Rerun visualization | `false` |
| `--log` | `bool` | **Enable comprehensive inference logging** | `false` |
| `--play_sounds` | `bool` | Use vocal synthesis for status updates | `true` |
| `--resume` | `bool` | Resume recording on existing dataset | `false` |

---

## **Policy Configuration**

### **Loading Models**
```bash
# From Hugging Face Hub
--policy.path=username/model_name

# Local path
--policy.path=/path/to/local/model

# With specific configuration overrides
--policy.device=cuda:0
--policy.use_amp=true
```

### **Policy Arguments**
| Argument | Type | Description |
|----------|------|-------------|
| `--policy.path` | `str` | Model path (HF Hub or local) |
| `--policy.device` | `str` | Device (`cpu`, `cuda`, `cuda:0`, etc.) |
| `--policy.use_amp` | `bool` | Use automatic mixed precision |

---

## **Dataset Configuration**

### **Basic Dataset Settings**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset.repo_id` | `str` | **Required** | Dataset identifier (`username/dataset_name`) |
| `--dataset.single_task` | `str` | **Required** | Task description |
| `--dataset.fps` | `int` | `30` | **Control loop frequency (Hz)** |
| `--dataset.episode_time_s` | `float` | `60` | Episode duration (seconds) |
| `--dataset.reset_time_s` | `float` | `60` | Reset phase duration (seconds) |
| `--dataset.num_episodes` | `int` | `50` | Number of episodes to record |

### **Storage & Upload Settings**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset.root` | `str` | `None` | Local storage directory |
| `--dataset.video` | `bool` | `true` | Encode frames as video |
| `--dataset.push_to_hub` | `bool` | `true` | Upload to Hugging Face Hub |
| `--dataset.private` | `bool` | `false` | Create private repository |
| `--dataset.tags` | `list[str]` | `None` | Add tags (e.g., `'["manipulation", "sim"]'`) |

### **Performance Settings**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset.num_image_writer_processes` | `int` | `0` | Number of image writer processes |
| `--dataset.num_image_writer_threads_per_camera` | `int` | `4` | Threads per camera for image saving |
| `--dataset.video_encoding_batch_size` | `int` | `1` | Episodes per video encoding batch |

---

## **Robot Configuration**

### **Basic Robot Setup**
```bash
--robot.type=so100_follower \
--robot.id=robot_name \
--robot.port=/dev/tty.usbmodem58760434091
```

### **Camera Configuration**
```bash
--robot.cameras='{
  "gripper": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},
  "top":     {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 25}
}'
```

### **Camera Parameters**
| Parameter | Type | Description | Recommended |
|-----------|------|-------------|-------------|
| `type` | `str` | Camera type (`opencv`, `realsense`) | `opencv` |
| `index_or_path` | `int/str` | Camera index or device path | `0`, `1`, `2` |
| `width` | `int` | Image width (pixels) | `640-1280` |
| `height` | `int` | Image height (pixels) | `480-720` |
| `fps` | `int` | Camera FPS | `25-30` |

---

## **Teleoperator Configuration**

### **Arm Teleoperators**
```bash
# SO-100 Leader Arm
--teleop.type=so100_leader \
--teleop.port=/dev/tty.usbmodem58760434092

# Gamepad Control
--teleop.type=gamepad

# Keyboard Control  
--teleop.type=keyboard
```

### **Multi-Device Setup (LeKiwi)**
```bash
--teleop='[
  {"type": "so100_leader", "port": "/dev/tty.leader"},
  {"type": "keyboard"}
]'
```

---

## **Inference Logging**

### **Enable Comprehensive Logging**
```bash
--log=true
```

### **What Gets Logged**
When `--log=true` is enabled, creates detailed CSV files in `inference_logs/`:

#### **1. Robot State Logging**
- **File**: `{robot_name}_robot_state.csv`
- **Data**: 
  - Timestamp, step count
  - Motor positions for all joints
  - Robot connection status
  - Motor loads, temperatures, voltages

#### **2. Policy Inference Logging**  
- **File**: `{robot_name}_policy_inference.csv`
- **Data**:
  - Inference timing (ms)
  - Policy input observations
  - Policy output actions
  - Raw policy tensor statistics (mean, std, min, max)
  - Task description

#### **3. Trajectory Logging**
- **File**: `{robot_name}_trajectory.csv`  
- **Data**:
  - Complete trajectory waypoints
  - Action sequences over time
  - Episode metadata

### **Log Directory Structure**
```
inference_logs/
└── dataset_name/
    └── 20250728_143052/
        ├── so100_follower_robot_state.csv
        ├── so100_follower_policy_inference.csv
        └── so100_follower_trajectory.csv
```

### **Console Output**
```
📊 INFERENCE STEP 1284 @ 59.21s
============================================================
🔧 SERVO POSITIONS:
   shoulder_pan.pos:    10.54
   shoulder_lift.pos:   -77.77
   elbow_flex.pos :    90.34
   wrist_flex.pos :   -15.65
   wrist_roll.pos :    39.49
   gripper.pos    :     0.88
🎯 POLICY OUTPUT:
   shoulder_pan.pos:    12.63
   shoulder_lift.pos:   -73.23
   elbow_flex.pos :    84.02
   wrist_flex.pos :   -17.12
   wrist_roll.pos :    38.28
   gripper.pos    :     0.62
⏱️  TIMING: Inference took 14.3ms
📋 TASK: Put the plastic part in the cup
============================================================
```

---

## **Real-time Visualization**

### **Enable Rerun Display**
```bash
--display_data=true
```

### **What You'll See**
- **Camera Feeds**: All configured cameras in real-time
- **Robot State**: Joint positions, loads, temperatures
- **Policy Actions**: Target positions and commands
- **Timing Graphs**: Inference timing, control frequency
- **Episode Progress**: Current episode, timing, completion

### **Rerun Interface**
- **Timeline**: Scrub through recorded data
- **Multi-view**: Multiple camera angles simultaneously  
- **Data Plots**: Real-time graphs of all scalar data
- **3D Visualization**: Robot pose visualization (if available)

---

## **Interactive Controls**

### **Keyboard Controls During Recording**

| Key | Action | Description |
|-----|--------|-------------|
| **`→` (Right Arrow)** | Exit Early | End current episode early |
| **`←` (Left Arrow)** | Re-record | Exit and re-record current episode |
| **`Esc`** | Stop Recording | Stop entire recording session |

### **Voice Announcements**
With `--play_sounds=true` (default):
- **Episode Start**: "Recording episode {N}"
- **Reset Phase**: "Reset the environment"  
- **Re-record**: "Re-record episode"

### **Episode Management**
- **Automatic Reset**: Built-in reset phase between episodes
- **Manual Reset**: Use reset time to manually prepare environment
- **Episode Validation**: Re-record failed episodes with `←` key

---

## **Performance Optimization**

### **Control Frequency**
```bash
# Standard 30 Hz control
--dataset.fps=30

# High-frequency control  
--dataset.fps=60

# Lower frequency for slow systems
--dataset.fps=20
```

### **Camera Optimization**
```bash
# Reduce resolution for better performance
--robot.cameras='{
  "gripper": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 25}
}'

# Multiple cameras with different settings
--robot.cameras='{
  "gripper": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30},
  "top":     {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 20}
}'
```

### **Image Writer Performance**
```bash
# For unstable FPS, try processes
--dataset.num_image_writer_processes=1 \
--dataset.num_image_writer_threads_per_camera=2

# For stable FPS, use threads only (default)
--dataset.num_image_writer_processes=0 \
--dataset.num_image_writer_threads_per_camera=4
```

### **Timing Analysis**
Use the inference logging to analyze performance:
```bash
--log=true
```
Monitor console output for timing bottlenecks:
- **Inference time**: Policy computation speed
- **Camera read time**: Image capture speed  
- **Total loop time**: Overall control frequency

---

## **Advanced Features**

### **Batch Video Encoding**
```bash
# Encode videos in batches (less frequent disk I/O)
--dataset.video_encoding_batch_size=5
```

### **Custom Storage Location**
```bash
--dataset.root=/path/to/custom/storage
```

### **Headless Operation**
```bash
# Disable visualization and sounds for server deployment
--display_data=false \
--play_sounds=false
```

### **Development Mode**
```bash
# Quick testing with minimal episodes
--dataset.num_episodes=2 \
--dataset.episode_time_s=10 \
--dataset.reset_time_s=5 \
--dataset.push_to_hub=false
```

---

## **Examples**

### **1. Autonomous Recording with Logging**
```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=follower_arm \
  --robot.port=/dev/tty.usbmodem58760434091 \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},
    "top":     {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 25}
  }' \
  --dataset.repo_id=username/pick_place_eval \
  --dataset.single_task="Pick up the object and place it in the box" \
  --dataset.fps=30 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --policy.path=username/trained_policy \
  --log=true \
  --display_data=true
```

### **2. Teleoperation Recording**
```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.follower \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.leader \
  --robot.cameras='{"cam": {"type": "opencv", "index_or_path": 0}}' \
  --dataset.repo_id=username/teleop_demos \
  --dataset.single_task="Demonstration of pick and place" \
  --dataset.num_episodes=50 \
  --display_data=true
```

### **3. High-Performance Recording**
```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.follower \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 60}
  }' \
  --dataset.repo_id=username/high_freq_data \
  --dataset.single_task="High frequency control task" \
  --dataset.fps=60 \
  --dataset.num_image_writer_processes=2 \
  --dataset.num_image_writer_threads_per_camera=2 \
  --policy.path=username/fast_policy \
  --log=true
```

### **4. Development/Debug Mode**
```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.follower \
  --robot.cameras='{"cam": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240}}' \
  --dataset.repo_id=username/debug_test \
  --dataset.single_task="Debug test run" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --dataset.video=false \
  --policy.path=username/test_policy \
  --log=true \
  --display_data=true
```

### **5. Multi-Modal Recording (LeKiwi)**
```bash
python -m lerobot.record \
  --robot.type=lekiwi_client \
  --teleop='[
    {"type": "so100_leader", "port": "/dev/tty.leader"},
    {"type": "keyboard"}
  ]' \
  --robot.cameras='{
    "head": {"type": "opencv", "index_or_path": 0},
    "wrist": {"type": "opencv", "index_or_path": 1}
  }' \
  --dataset.repo_id=username/mobile_manip \
  --dataset.single_task="Mobile manipulation task" \
  --display_data=true
```

---

## **🔧 Troubleshooting**

### **Performance Issues**
- **Low FPS**: Reduce camera resolution, increase image writer threads
- **High latency**: Check `--dataset.fps` setting, monitor inference timing
- **Memory issues**: Reduce batch size, use video encoding

### **Connection Issues**  
- **Robot disconnection**: Check USB connections, motor power
- **Camera errors**: Verify camera indices with `python -m lerobot.find_cameras`
- **Policy loading**: Ensure model path exists and is accessible

### **Logging Issues**
- **No inference logs**: Ensure `--log=true` and policy is loaded
- **CSV errors**: Check disk space and write permissions
- **Console spam**: Normal during reset phase (policy=None is expected)

---

## **📊 Data Analysis**

After recording with `--log=true`, analyze your data:

```python
import pandas as pd

# Load logged data
robot_state = pd.read_csv('inference_logs/dataset_name/timestamp/robot_state.csv')
policy_data = pd.read_csv('inference_logs/dataset_name/timestamp/policy_inference.csv')

# Analyze timing
print(f"Average inference time: {policy_data['inference_time_ms'].mean():.1f}ms")
print(f"Control frequency: {1000/policy_data['inference_time_ms'].mean():.1f} Hz")

# Plot trajectories
import matplotlib.pyplot as plt
plt.plot(robot_state['timestamp'], robot_state['shoulder_pan.pos'])
plt.title('Joint Trajectory')
plt.show()
```

This documentation covers all major aspects of `lerobot.record` for inference, control, logging, and visualization! 🚀 