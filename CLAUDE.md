# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is Hugging Face's state-of-the-art machine learning library for real-world robotics in PyTorch. It provides pre-trained models, datasets, and tools for robotic learning.

## Key Commands

### Installation

```bash
# Basic installation
pip install -e .

# With specific hardware support
pip install -e ".[aloha,pusht,xarm]"     # Simulation environments
pip install -e ".[intelrealsense]"       # RealSense cameras
pip install -e ".[dynamixel,feetech]"    # Motor control
pip install -e ".[dev,test]"             # Development tools
```

### Development Commands

```bash
# Run tests
pytest tests -vv                         # All tests
pytest tests/cameras/                    # Camera tests only
pytest tests/policies/                   # Policy tests only
pytest -k "test_name"                    # Run specific test

# End-to-end testing (via Makefile)
make DEVICE=cpu test-end-to-end
make DEVICE=cuda test-end-to-end
make test-act-ete-train                  # Test ACT policy training
make test-diffusion-ete-train            # Test Diffusion policy training

# Code quality
ruff check                               # Linting
ruff format                              # Formatting
pre-commit run --all-files               # Run all pre-commit hooks

# CLI Tools (installed as scripts)
lerobot-find-cameras                     # Discover cameras
lerobot-calibrate                        # Camera calibration
lerobot-record                           # Record dataset
lerobot-train                            # Train policy
lerobot-eval                             # Evaluate policy
lerobot-teleoperate                      # Manual control
```

## Architecture Overview

### Core Package Structure (`src/lerobot/`)

**Hardware Abstraction Layer**:

- `cameras/` - Camera drivers (OpenCV, RealSense, Kinect v2)
- `robots/` - Robot implementations (ALOHA, SO-100/101, Koch)
- `motors/` - Motor control (Dynamixel, Feetech)
- `teleoperators/` - Control interfaces (gamepad, keyboard)

**ML/AI Components**:

- `policies/` - Policy implementations (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA)
- `datasets/` - Dataset handling and processing
- `utils/` - Shared utilities

**Entry Points**:

- `scripts/` - Main CLI scripts for training, evaluation, visualization

### Key Architecture Patterns

1. **Plugin-Based Architecture**: Components are dynamically loaded via factory patterns
   - Look for `factory.py` files in each module
   - Register new components with `@register_*` decorators
   - Configuration-driven instantiation

2. **Configuration Management**: Dataclass-based configurations
   - Each component has a `configuration_*.py` file
   - Hierarchical config system with validation
   - Example: `cameras/opencv/configuration_opencv.py`

3. **Async/Threading**: For real-time performance
   - Camera capture runs in background threads
   - Thread-safe frame buffering
   - Async inference pipelines

4. **Hardware Abstraction**: Consistent interfaces
   - Base classes define contracts (e.g., `BaseCamera`, `BaseRobot`)
   - Mock implementations for testing (`tests/mocks/`)
   - Graceful fallback and error handling

### Adding New Components

**Adding a Camera**:

1. Create implementation in `cameras/your_camera/camera_your_camera.py`
2. Add configuration in `cameras/your_camera/configuration_your_camera.py`
3. Register in factory with `@register_camera`
4. Update `cameras/__init__.py` availability check
5. Add tests with mocks in `tests/cameras/`

**Adding a Policy**:

1. Implement in `policies/your_policy/modeling_your_policy.py`
2. Add configuration dataclass
3. Register in policy factory
4. Include training example in `scripts/`
5. Add tests for training/inference

**Adding a Robot**:

1. Create in `robots/your_robot.py`
2. Implement motor control interface
3. Add teleoperation support
4. Include calibration routines
5. Provide example usage

### Testing Strategy

**Test Organization**:

- Unit tests per module in `tests/`
- Mock hardware in `tests/mocks/`
- Fixtures in `tests/fixtures/`
- End-to-end tests in Makefile

**Running Tests**:

```bash
# Quick test for specific module
pytest tests/cameras/test_opencv.py -v

# Test with coverage
pytest tests/ --cov=lerobot --cov-report=html

# Test specific markers
pytest -m "not slow"
```

### Kinect v2 Integration

The codebase includes comprehensive Kinect v2 support with GPU acceleration:

**Key Files**:

- `src/lerobot/cameras/kinect/camera_kinect.py` - Main implementation
- `src/lerobot/cameras/kinect/configuration_kinect.py` - Configuration
- `kinect-toolbox-py310/` - High-level Python wrapper
- `pylibfreenect2-py310/` - Python bindings

**Pipeline Priority** (automatic selection):

1. CUDA - ~25-35 FPS (NVIDIA GPUs)
2. OpenCL - ~20-30 FPS (All GPUs)
3. OpenGL - ~15-25 FPS (Graphics cards)
4. CPU - ~5-15 FPS (Fallback)

**Environment Variables**:

```bash
LIBFREENECT2_INSTALL_PREFIX=C:\path\to\libfreenect2
LIBFREENECT2_PIPELINE=cuda  # Force specific pipeline
```

### Common Development Tasks

**Training a Policy**:

```bash
# Train ACT policy on ALOHA dataset
lerobot-train policy=act env=aloha dataset_repo_id=lerobot/aloha_sim_insertion_human

# Resume training from checkpoint
lerobot-train --resume

# Custom configuration
lerobot-train policy=diffusion env=pusht training.batch_size=32
```

**Recording Data**:

```bash
# Record teleoperated dataset
lerobot-record --robot-path robots/so100.py --fps 30 --root data/

# With specific camera
lerobot-record --robot-path robots/so100.py --camera-type kinect
```

**Evaluating Models**:

```bash
# Evaluate trained policy
lerobot-eval --policy-path outputs/train/act/ --robot-path robots/so100.py

# Record evaluation videos
lerobot-eval --policy-path outputs/train/act/ --record
```

### Code Style and Conventions

- Use type hints for all function signatures
- Follow existing patterns for new components
- Document with docstrings (Google style)
- Add unit tests for new functionality
- Run pre-commit hooks before committing

### Important Notes

- Always check hardware availability before using (`available_*` functions)
- Use mocks for hardware testing
- Configuration dataclasses should be frozen
- Factory registration is automatic via decorators
- Thread safety is critical for real-time components

## Camera System Deep Analysis

### Camera Architecture Overview

The camera system in LeRobot follows a consistent architecture pattern across all implementations:

```
Camera (ABC)                    # Abstract base class defining interface
├── OpenCVCamera               # Generic USB/webcam support
├── RealSenseCamera            # Intel RealSense depth cameras
└── KinectCamera               # Microsoft Kinect v2 with GPU acceleration
```

### Core Components

**1. Abstract Base Class (`camera.py`)**

- Defines the camera interface contract that all implementations must follow
- Key abstract methods:
  - `is_connected` - Connection status property
  - `find_cameras()` - Static method for device discovery
  - `connect(warmup=True)` - Establish connection with optional warmup
  - `read(color_mode=None)` - Synchronous frame capture
  - `async_read(timeout_ms)` - Asynchronous frame capture
  - `disconnect()` - Clean resource release

**2. Configuration System (`configs.py`)**

- Base `CameraConfig` using `draccus.ChoiceRegistry` for automatic registration
- Common configuration fields: `fps`, `width`, `height`
- `ColorMode` enum: RGB or BGR output format
- `Cv2Rotation` enum: 0°, 90°, 180°, 270° rotation support
- Each camera type extends with specific configuration needs

**3. Factory Pattern (`utils.py`)**

- `make_cameras_from_configs()` - Creates camera instances from config dictionary
- Dynamic import based on camera type to avoid unnecessary dependencies
- Maps config types to implementation classes:
  - `"opencv"` → `OpenCVCamera`
  - `"intelrealsense"` → `RealSenseCamera`
  - `"kinect"` → `KinectCamera`
- Platform-specific backend selection for OpenCV (Windows: MSMF, Others: ANY)

### Implementation Patterns

**1. Thread-Safe Async Capture**
All camera implementations follow this pattern:

```python
# Background thread continuously captures frames
self.thread = Thread(target=self._capture_loop)
self.frame_lock = Lock()          # Thread-safe frame access
self.latest_frame = None          # Most recent frame
self.new_frame_event = Event()    # Signals new frame availability
```

**2. Connection Lifecycle**

```python
camera.connect(warmup=True)       # Initialize hardware, optional warmup
# ... use camera ...
camera.disconnect()               # Always cleanup resources
```

**3. Multi-Stream Support**
RealSense and Kinect support multiple data streams:

- Color (RGB/BGR)
- Depth (millimeters as uint16 or float32)
- Infrared (grayscale intensity)

### Camera-Specific Details

**OpenCVCamera (`opencv/camera_opencv.py`)**

- Uses cv2.VideoCapture with platform-specific backends
- Supports both device indices (0, 1, 2...) and paths (/dev/video0)
- Max search index: 60 (accounts for non-sequential indices on Linux)
- MSMF hardware transform disabled on Windows for compatibility
- Rotation handled pre-capture for efficiency

**RealSenseCamera (`realsense/camera_realsense.py`)**

- Identifies cameras by serial number (stable) or name (if unique)
- Native support for aligned depth-to-color registration
- Configurable depth filtering and colorization
- Hardware-accelerated depth processing
- Temporal filtering for depth stability

**KinectCamera (`kinect/camera_kinect.py`)**

- GPU-accelerated pipelines with automatic fallback:
  1. CUDA (~25-35 FPS) - NVIDIA GPUs only
  2. OpenCL (~20-30 FPS) - All GPU vendors
  3. OpenGL (~15-25 FPS) - Graphics pipeline
  4. CPU (~5-15 FPS) - Software fallback
- Fixed resolutions: Color 1920x1080, Depth/IR 512x424
- Advanced depth filtering: bilateral and edge-aware
- BGRX to BGR conversion for OpenCV compatibility
- Proper resource cleanup with garbage collection

### Common Usage Patterns

**1. Camera Discovery**

```python
# Find all available cameras of a type
cameras = OpenCVCamera.find_cameras()
cameras = RealSenseCamera.find_cameras()
cameras = KinectCamera.find_cameras()
```

**2. Basic Capture Loop**

```python
config = OpenCVCameraConfig(index_or_path=0, fps=30)
camera = OpenCVCamera(config)
camera.connect()

try:
    while True:
        frame = camera.read()  # or camera.async_read()
        # Process frame...
finally:
    camera.disconnect()
```

**3. Multi-Camera Setup**

```python
configs = {
    "camera_front": OpenCVCameraConfig(index_or_path=0),
    "camera_side": RealSenseCameraConfig(serial_number="123456"),
    "camera_top": KinectCameraConfig(device_index=0)
}
cameras = make_cameras_from_configs(configs)
```

### Performance Considerations

1. **Async vs Sync Reads**:
   - `read()` - Blocks until frame captured (simple but can cause delays)
   - `async_read()` - Returns latest buffered frame (preferred for real-time)

2. **Resolution Trade-offs**:
   - Higher resolution = better quality but lower FPS
   - Kinect v2 has fixed hardware resolutions
   - Consider rotation impact on capture dimensions

3. **Pipeline Selection** (Kinect):
   - Set `LIBFREENECT2_PIPELINE` environment variable to force specific pipeline
   - CUDA requires NVIDIA GPU with CUDA toolkit
   - OpenCL works on AMD/Intel/NVIDIA GPUs
   - CPU fallback always available but slow

### Error Handling

Common exceptions to handle:

- `DeviceNotConnectedError` - Camera not connected when operation attempted
- `DeviceAlreadyConnectedError` - Trying to connect already connected camera
- `TimeoutError` - Frame capture timeout in async_read
- `ImportError` - Optional dependencies not installed (pyrealsense2, pylibfreenect2)

### Testing Cameras

```bash
# Test camera discovery and basic functionality
python -m lerobot.find_cameras opencv
python -m lerobot.find_cameras realsense
python -m lerobot.find_cameras kinect

# Test specific camera capture
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
config = OpenCVCameraConfig(index_or_path=0)
camera = OpenCVCamera(config)
camera.connect()
frame = camera.read()
print(f"Captured frame shape: {frame.shape}")
camera.disconnect()
```

### Adding a New Camera Type

1. Create new module in `cameras/your_camera/`
2. Implement camera class inheriting from `Camera` ABC
3. Create configuration class inheriting from `CameraConfig`
4. Register with draccus by importing in `cameras/__init__.py`
5. Add to factory in `utils.py`
6. Include comprehensive tests with hardware mocks
7. Document specific setup requirements
