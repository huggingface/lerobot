# Grid HIL SERL Environment

This example demonstrates a **simplified HIL-SERL setup** for computer vision-based grid position prediction. Instead of complex robotic manipulation, the algorithm learns to predict which of the 64 grid cells contains a red cube based on camera images, with human feedback during training. Episodes are single prediction attempts: if the guess is correct, the agent receives reward 1; otherwise reward 0.

## Overview

The environment consists of:
- An 8x8 grid world with high-definition visual rendering
- A red cube that randomly spawns at grid cell centers
- Top-left origin coordinate system (0,0) = top-left corner
- Automatic high-definition image capture (1920x1080)

## Files

- `grid_scene.xml` - Mujoco scene definition with 8x8 grid
- `grid_cube_randomizer.py` - Main script for randomizing cube positions
- `README.md` - This documentation

## Usage

### 1. Test the Environment
```bash
cd examples/grid_hil_serl
python grid_cube_randomizer.py
```

### 2. Record Demonstrations
```bash
# Record training data automatically (single step episodes)
python examples/grid_hil_serl/record_grid_demo.py --config_path examples/grid_hil_serl/record_grid_position_lerobot.json

# Or use LeRobot's recording script (standard dataset format)
python -m lerobot.scripts.rl.gym_manipulator --config_path record_grid_position_lerobot.json
```

### 3. Train HIL-SERL Policy
```bash
# Terminal 1: Start learner
python -m lerobot.scripts.rl.learner --config_path train_grid_position.json

# Terminal 2: Start actor (with human feedback)
python -m lerobot.scripts.rl.actor --config_path train_grid_position.json
```

### Command Line Options
```bash
# Environment testing
python grid_cube_randomizer.py --interval 2.0 --no-save

# Recording options (edit the JSON config to change episodes/root)
python examples/grid_hil_serl/record_grid_demo.py --config_path examples/grid_hil_serl/record_grid_position_lerobot.json
```

## Features

### Grid System
- **8x8 grid**: 64 total cells
- **Coordinate system**: (0,0) = top-left, (7,7) = bottom-right
- **Cell centers**: Cube spawns at precise grid cell centers
- **High-definition**: 32x32 texture with 256x256 resolution

### Cube Positioning
- **Random placement**: Uniform random distribution across all 64 cells
- **Precise positioning**: Cube lands exactly at grid cell centers
- **Physics compliant**: Proper velocity reset for instant teleportation
- **Visual feedback**: Clear console output of cell coordinates

### Image Capture
- **HD resolution**: 1920x1080 (Full HD)
- **Automatic saving**: Images saved after each cube repositioning
- **Professional quality**: Suitable for datasets and documentation
- **Top-down view**: Camera positioned for complete grid visibility

## Coordinate System

```
(0,0) → (-3.5, 3.5)   (7,0) → (3.5, 3.5)
      ↘                     ↙
(0,7) → (-3.5, -3.5)  (7,7) → (3.5, -3.5)
```

## HIL-SERL Workflow

This simplified setup demonstrates the core HIL-SERL concept with minimal complexity:

### Training Phase (Offline)
1. **Automatic Data Collection**: Environment randomly places cube in different grid positions
2. **Supervised Learning**: Algorithm learns to predict grid position from images
3. **Ground Truth Labels**: Exact grid coordinates provided for each image

### Human-in-the-Loop Phase (Online)
1. **Algorithm Prediction**: Model predicts cube position from camera images
2. **Binary Feedback**: Human (or auto-supervision) accepts or corrects the guess
3. **Iterative Learning**: Model improves based on the accepted/corrected outcome

### Key Simplifications
- **No Robot Control**: Focus purely on computer vision prediction
- **Single-Step Episodes**: One prediction per episode with immediate success/failure reward
- **Discrete Predictions**: 64 possible outputs (one per grid cell)
- **Perfect Ground Truth**: Exact position labels available
- **Visual Task Only**: No complex motor control or physics

## Integration with LeRobot

The environment integrates with LeRobot's HIL-SERL framework through:

1. **Custom Gym Environment**: `GridPositionPrediction-v0` registered with gymnasium
2. **LeRobot-Compatible Interface**: Proper observation/action space formatting
3. **Config Files**: `record_grid_position.json` and `train_grid_position.json`
4. **Dataset Collection**: Automated recording of image-position pairs

## Technical Details

- **Physics**: Mujoco physics engine with proper joint control
- **Rendering**: Offscreen rendering with PIL for image saving
- **Randomization**: NumPy-based random number generation
- **Threading**: Proper event handling for viewer controls

## Example Output

```
Loading scene: grid_scene.xml

==================================================
8x8 Grid Cube Randomizer
==================================================
This scene shows an 8x8 grid with a randomly positioned cube.
Cube position randomizes every 3.0 seconds.

Controls:
  R: Manually randomize cube position
  S: Save current camera view to img.jpg
  Space: Pause/unpause
  Esc: Exit
  Camera: Mouse controls for rotation/zoom
==================================================
Spawning cube at grid cell (3, 5) -> position (-0.5, -1.5)
Camera view saved to: img.jpg
Spawning cube at grid cell (1, 2) -> position (-2.5, 1.5)
Camera view saved to: img.jpg
```

## Dependencies

- mujoco
- numpy
- PIL (Pillow)
- gymnasium (optional, for integration)

## Related Examples

- `hil_serl_simulation_training/` - Full HIL-SERL training examples
- `lekiwi/` - Real robot integration examples
