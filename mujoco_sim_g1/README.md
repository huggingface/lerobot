# Standalone MuJoCo Simulator for Unitree G1

This is a standalone MuJoCo physics simulator for the Unitree G1 robot, extracted from the GR00T-WholeBodyControl repository.

## Features

- **Physics Simulation**: Runs G1 robot in MuJoCo at 500Hz (2ms timestep)
- **DDS Communication**: Uses Unitree SDK2 DDS for robot communication
- **Compatible**: Works with existing `unitree_g1.py` control code via DDS
- **Visualization**: Real-time 3D visualization of robot motion

## Directory Structure

```
mujoco_sim_g1/
├── requirements.txt          # Python dependencies
├── run_sim.py               # Main launcher script
├── config.yaml              # Simulation configuration
├── sim/                     # Simulation modules
│   ├── base_sim.py
│   ├── simulator_factory.py
│   ├── unitree_sdk2py_bridge.py
│   └── ...
└── assets/                  # Robot models
    ├── g1_29dof_with_hand.xml
    └── meshes/*.STL
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
cd mujoco_sim_g1
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `rclpy`, you can comment out the ROS2 imports in `sim/base_sim.py` (lines 11, 609-617) if you don't need camera publishing.

## Usage

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Run the simulator
python run_sim.py
```

The simulator will:
1. Load the G1 robot model
2. Initialize DDS communication on domain 0
3. Open a MuJoCo visualization window
4. Start listening for motor commands via DDS

### Running with Your Robot Control Code

Once the simulator is running, you can control it from another terminal:

```bash
# In another terminal
cd /home/yope/Documents/lerobot
conda activate unitree_lerobot

# Run your existing control code
python test_locomotion_minimal.py
```

Your `unitree_g1.py` code will automatically connect to the simulator via DDS!

## Configuration

Edit `config.yaml` to customize:

- **SIMULATE_DT**: Simulation timestep (default: 0.002s = 500Hz)
- **DOMAIN_ID**: DDS domain ID (default: 0)
- **ENABLE_ONSCREEN**: Show visualization (default: true)
- **USE_JOYSTICK**: Enable gamepad control (default: false)
- **ROBOT_SCENE**: Path to MuJoCo XML scene

### PD Gains

The simulator uses the following PD gains (matching NVIDIA GR00T):

**Legs (indices 0-11):**
- Hip joints: KP=150, KD=2
- Knee joints: KP=300, KD=4
- Ankle joints: KP=40, KD=2

**Waist (indices 12-14):**
- All waist joints: KP=250, KD=5

**Arms (indices 15-28):**
- Shoulders: KP=100, KD=2-5
- Elbows/Wrists: KP=20-40, KD=1-2

## Troubleshooting

### ImportError: cannot import name 'ChannelFactoryInitialize'

```bash
pip install --upgrade unitree-sdk2py
```

### ROS2/rclpy errors

If you don't need camera publishing, edit `sim/base_sim.py`:
- Comment out line 11: `import rclpy`
- Comment out lines 609-617 (ROS2 initialization)

### Meshes not found

Make sure mesh paths in `assets/g1_29dof_with_hand.xml` are relative:
```xml
<mesh file="meshes/pelvis.STL"/>  <!-- Not absolute path -->
```

### Robot falls immediately

Check that:
1. PD gains match NVIDIA's values (see config.yaml)
2. Velocity command scaling is correct (ang_vel_scale=0.25, cmd_scale=[2.0, 2.0, 0.25])
3. Observations for missing joints are zeroed out (indices 12, 14, 20, 21, 27, 28)

## Technical Details

### Communication

The simulator publishes:
- **`rt/lowstate`**: Robot state (joint positions, velocities, IMU, etc.)
- **`rt/wirelesscontroller`**: Wireless remote controller state (if joystick enabled)

The simulator subscribes to:
- **`rt/lowcmd`**: Motor commands (position, velocity, torque, KP, KD)

### Coordinate Frames

- **World frame**: Z-up
- **Joint ordering**: 29 DOF (12 legs + 3 waist + 14 arms)
- **IMU**: Quaternion in [w, x, y, z] format

### Performance

- Simulation runs at ~500Hz (2ms timestep)
- Viewer updates at ~50Hz (20ms)
- Typical CPU usage: 20-40% on single core

## Files from GR00T-WholeBodyControl

This standalone simulator was extracted from:
- `gr00t_wbc/control/envs/g1/sim/` (simulation modules)
- `gr00t_wbc/control/robot_model/model_data/g1/` (robot model files)
- `gr00t_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml` (configuration)

## License

Follows the license of the original GR00T-WholeBodyControl repository.

