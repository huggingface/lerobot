# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
- `pip install -e .` - Install LeRobot in development mode
- `pip install -e ".[aloha, pusht]"` - Install with simulation environment extras
- `conda create -y -n lerobot python=3.10 && conda activate lerobot` - Create virtual environment
- `conda install ffmpeg -c conda-forge` - Install ffmpeg dependency
- `wandb login` - Setup Weights & Biases for experiment tracking

### Testing
- `pytest` - Run all tests
- `pytest tests/path/to/specific_test.py` - Run specific test file
- `pytest -v` - Run with verbose output
- `pytest --timeout=300` - Run with timeout
- `make test-end-to-end` - Run end-to-end tests with all policies
- `make DEVICE=cuda test-end-to-end` - Run tests on GPU

### Code Quality
- `ruff check .` - Run linter
- `ruff format .` - Format code
- `bandit -r src/` - Security checks
- `pre-commit run --all-files` - Run pre-commit hooks

### Training and Evaluation
- `python -m lerobot.scripts.train --config_path=lerobot/diffusion_pusht` - Train using existing config
- `python -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht` - Evaluate policy
- `python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0` - Visualize dataset

### Robot Operations
- `lerobot-find-cameras` - Find available cameras
- `lerobot-find-port` - Find available serial ports
- `lerobot-calibrate` - Calibrate robot motors
- `lerobot-record` - Record demonstrations
- `lerobot-replay` - Replay recorded episodes
- `lerobot-teleoperate` - Teleoperate robot

## Teleoperation Commands
- `python -m lerobot.teleoperate --robot.type=bi_so101_follower --robot.left_arm_port=COM4 --robot.right_arm_port=COM9 --robot.id=my_bimanual --teleop.type=bi_so101_leader --teleop.left_arm_port=COM11 --teleop.right_arm_port=COM3 --teleop.id=my_bimanual_leader --display_data=true` - Bimanual SO101 teleoperation command with specific port configurations

## Architecture Overview

LeRobot is a comprehensive robotics library with the following key components:

### Core Structure
- **`src/lerobot/`** - Main source code directory
- **`datasets/`** - Dataset handling with LeRobotDataset format
- **`policies/`** - ML policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA, PI0)
- **`robots/`** - Robot implementations (Koch, ALOHA, SO100, SO101, etc.)
- **`cameras/`** - Camera integrations (OpenCV, Intel RealSense)
- **`motors/`** - Motor drivers (Dynamixel, Feetech)
- **`envs/`** - Simulation environments integration

### Policy Types
- **ACT** - Action Chunking Transformer for manipulation tasks
- **Diffusion** - Diffusion-based policy learning
- **TDMPC** - Temporal Difference Model Predictive Control
- **VQ-BeT** - Vector Quantized Behavior Transformer
- **SmolVLA** - Compact Vision-Language-Action model
- **PI0** - Transformer-based imitation learning

### Data Format
LeRobotDataset uses:
- **Hugging Face datasets** for metadata (parquet/arrow)
- **MP4 videos** for efficient image storage
- **Temporal indexing** with delta_timestamps for multi-frame observations
- **Episode-based organization** with start/end indices

### Robot Hardware Support
- **Koch arms** - 6DOF manipulation arms
- **ALOHA** - Bimanual manipulation setup
- **SO100/SO101** - Low-cost educational arms (€114)
- **HopeJR** - Humanoid arm with hand
- **LeKiwi** - Mobile base for SO-101
- **Stretch3** - Hello Robot mobile manipulator

### Key Configuration Files
- **`pyproject.toml`** - Main project configuration and dependencies
- **`Makefile`** - Test automation and development workflows
- **`src/lerobot/__init__.py`** - Available robots, policies, datasets registry

### Training Pipeline
1. Load datasets using LeRobotDataset
2. Configure policy with appropriate config class
3. Train using `lerobot.scripts.train` with Hydra configuration
4. Evaluate with `lerobot.scripts.eval`
5. Upload models to Hugging Face Hub

### Development Workflow
1. Follow existing code patterns in each module
2. Update availability lists in `__init__.py` for new components  
3. Add tests following existing structure in `tests/`
4. Use pre-commit hooks for code quality
5. Check security with bandit before commits

### Camera Integration

#### RealSense Camera Support
- **Intel RealSense D405** cameras are supported via `intelrealsense` type
- **Configuration**: Use `RealSenseCameraConfig` with serial number identification
- **Image format**: Returns numpy arrays with shape `(height, width, 3)` and `dtype=uint8`
- **Multiple cameras**: Support for multiple cameras using unique serial numbers

#### Camera Configuration Example
```python
# In teleoperation script
--robot.cameras='{
  "left_cam": {"type": "intelrealsense", "serial_number_or_name": "218622270973", "width": 848, "height": 480, "fps": 30},
  "right_cam": {"type": "intelrealsense", "serial_number_or_name": "218622278797", "width": 848, "height": 480, "fps": 30}
}'
```

#### Camera Troubleshooting
- **Find cameras**: `python -m lerobot.find_cameras` to detect available cameras and their serial numbers
- **Test individual camera**: Create simple test script to verify camera connectivity
- **Common issues**: 
  - Camera busy from previous sessions - ensure proper disconnect
  - USB bandwidth limitations with multiple cameras
  - Driver updates may be needed for RealSense SDK

#### Rerun Visualization
- **Real-time visualization** of robot teleoperation data using Rerun (rerun.io)
- **Camera feeds**: Automatically logged as `observation.left_cam`, `observation.right_cam`
- **Motor data**: Joint positions logged as scalars for each motor
- **Navigation**: In Rerun viewer, expand "observation" tree in left panel to find camera streams
- **Image logging**: Camera data logged via `rr.Image()` in `log_rerun_data()` function
- **Troubleshooting**: 
  - Rerun cache clearing: `pip cache purge` then `pip install rerun-sdk --no-cache-dir --force-reinstall`
  - Test Rerun independently with simple image logging to verify functionality

#### Camera Data Pipeline
1. **Bimanual robot** calls `get_observation()` during teleoperation
2. **Individual cameras** read frames via `camera.async_read()` 
3. **Data format**: Returns `(480, 848, 3)` numpy arrays for RealSense D405
4. **Rerun logging**: Images logged via `rr.log(f"observation.{cam_name}", rr.Image(image_array))`
5. **Real-time display**: ~30Hz camera streaming with motor position data

### Important Notes
- Requires Python 3.10+
- GPU recommended for training (CUDA support)
- Windows compatibility with MINGW64
- Integration with Hugging Face ecosystem for model/dataset sharing
- Real-time robot control capabilities with motor drivers
- Extensive simulation environment support
- **RealSense integration** requires `pip install -e ".[intelrealsense]"`
- **Rerun visualization** automatically enabled with `--display_data=true`