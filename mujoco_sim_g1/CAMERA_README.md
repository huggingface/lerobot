# Camera System for MuJoCo G1 Simulator

## Overview

The simulator has two cameras defined:

### 1. **`head_camera`** - Robot Ego-View
- **Location**: Attached to `torso_link` body
- **Position**: `[0.06, 0.0, 0.45]` relative to torso (6cm forward, 45cm up)
- **Orientation**: `euler="0 -0.8 -1.57"` (facing forward, slightly tilted down)
- **FOV**: 90 degrees
- **Purpose**: First-person view from the robot's perspective (like a head-mounted camera)

### 2. **`global_view`** - Third-Person View  
- **Location**: Fixed in world coordinates
- **Position**: `[2.910, -5.040, 3.860]` (behind and above the robot)
- **Purpose**: External observer view for visualization

## How Camera Publishing Works

The camera system uses a **zero-copy architecture** with three components:

```
┌─────────────────────┐
│   MuJoCo Simulator  │
│  (Main Process)     │
│                     │
│  1. Render cameras  │
│  2. Copy to shmem   │──┐
└─────────────────────┘  │
                         │ Shared Memory
                         │ (fast IPC)
                    ┌────▼────────────────┐
                    │  Image Publisher    │
                    │  (Subprocess)       │
                    │                     │
                    │  3. Encode images   │
                    │  4. ZMQ publish     │
                    └─────────┬───────────┘
                              │
                              │ TCP (ZMQ)
                              │ port 5555
                              ▼
                    ┌─────────────────────┐
                    │  Your Policy/Client │
                    │  (Subscribe)        │
                    └─────────────────────┘
```

### Key Technologies:
- **MuJoCo Renderer**: Captures RGB images from virtual cameras
- **Shared Memory (`multiprocessing.shared_memory`)**: Zero-copy transfer between processes
- **ZMQ (ZeroMQ)**: Network socket for publishing images (TCP)
- **No ROS2 required!** Pure Python multiprocessing

## Usage

### Basic Simulation (No Camera Publishing)
```bash
python run_sim.py
```

### With Camera Publishing
```bash
# Publish head camera on default port 5555
python run_sim.py --publish-images

# Publish multiple cameras
python run_sim.py --publish-images --cameras head_camera global_view

# Custom port
python run_sim.py --publish-images --camera-port 6000
```

### Viewing Camera Streams

In a **separate terminal**, run the camera viewer:

```bash
# Basic usage (default: localhost:5555)
python view_cameras.py

# Custom host/port
python view_cameras.py --host 192.168.1.100 --port 6000

# Save images to directory
python view_cameras.py --save ./camera_recordings

# Adjust display rate
python view_cameras.py --fps 60
```

**Keyboard Controls:**
- `q`: Quit viewer
- `s`: Save snapshot of current frame

**Example Workflow:**
```bash
# Terminal 1: Start simulator with camera publishing
python run_sim.py --publish-images

# Terminal 2: View the camera feed
python view_cameras.py
```

### Receiving Images in Your Code

```python
import zmq
import numpy as np
from gr00t_wbc.control.sensor.sensor_server import ImageMessageSchema

# Connect to camera publisher
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

while True:
    # Receive serialized image data
    data = socket.recv_pyobj()
    
    # Decode images
    if "head_camera" in data:
        # Decode image (returns numpy array HxWx3 uint8)
        img = decode_image(data["head_camera"])
        
        # Use image for your policy
        process_observation(img)
```

## Camera Configuration

Edit `config.yaml` to change camera settings:

```yaml
IMAGE_DT: 0.033333  # Publishing rate (30 Hz)
ENABLE_OFFSCREEN: false  # Enable for camera rendering
MP_START_METHOD: "spawn"  # Multiprocessing method
```

Or programmatically in `run_sim.py`:

```python
camera_configs = {
    "head_camera": {
        "height": 480,
        "width": 640
    },
    "custom_camera": {
        "height": 224,
        "width": 224
    }
}
```

## Adding Custom Cameras

Edit `assets/g1_29dof_with_hand.xml` or `assets/scene_43dof.xml`:

```xml
<!-- Camera attached to robot body -->
<body name="torso_link" pos="0 0 0.019">
  <camera name="my_camera" pos="0.1 0.0 0.5" euler="0 0 0" fovy="60"/>
</body>

<!-- Camera in world coordinates -->
<worldbody>
  <camera name="side_view" pos="0 -3.0 1.5" xyaxes="1 0 0 0 0.5 0.866"/>
</worldbody>
```

Then publish it:
```bash
python run_sim.py --publish-images --cameras my_camera
```

## Performance Notes

- **Rendering overhead**: ~5-10ms per camera per frame @ 640x480
- **Publishing overhead**: ~2-3ms for encoding + network
- Image publishing runs in **separate subprocess** to not block simulation
- Uses **shared memory** for fast inter-process image transfer
- Target: 30 FPS camera publishing while maintaining 500 Hz simulation

## Troubleshooting

### No images received?
1. Check if offscreen rendering is enabled (`--publish-images` flag)
2. Verify ZMQ port is not blocked
3. Check camera exists in scene XML

### Images are delayed?
- Reduce `IMAGE_DT` in config
- Lower camera resolution
- Use fewer cameras

### "Camera not found" error?
- Verify camera name in XML matches config
- Check XML syntax is valid
- Ensure MuJoCo model loads successfully

## Quick Reference

### File Structure
```
mujoco_sim_g1/
├── run_sim.py              # Simulator launcher
├── view_cameras.py         # Camera viewer (this file!)
├── config.yaml             # Simulator config
├── assets/
│   ├── scene_43dof.xml     # Scene with global_view camera
│   └── g1_29dof_with_hand.xml  # Robot model with head_camera
└── sim/
    ├── base_sim.py         # MuJoCo environment
    ├── sensor_utils.py     # ZMQ camera server/client
    └── image_publish_utils.py  # Multiprocessing image publisher
```

### Camera Definitions

Edit these files to modify cameras:

**`assets/g1_29dof_with_hand.xml`** - Robot-attached cameras:
```xml
<body name="torso_link" pos="0 0 0.019">
  <camera name="head_camera" pos="0.06 0.0 0.45" euler="0 -0.8 -1.57" fovy="90"/>
</body>
```

**`assets/scene_43dof.xml`** - World-frame cameras:
```xml
<worldbody>
  <camera name="global_view" pos="2.910 -5.040 3.860" xyaxes="0.866 0.500 0.000 -0.250 0.433 0.866" fovy="45"/>
</worldbody>
```

### Complete Example

```bash
# Terminal 1: Start simulator with camera publishing
cd mujoco_sim_g1
python run_sim.py --publish-images --cameras head_camera global_view

# Terminal 2: View cameras in real-time
python view_cameras.py

# Terminal 3: Use in your policy (Python code)
from sim.sensor_utils import SensorClient, ImageUtils
client = SensorClient()
client.start_client("localhost", 5555)
data = client.receive_message()
img = ImageUtils.decode_image(data["head_camera"])
# img is now numpy array (H, W, 3) in BGR format
```

