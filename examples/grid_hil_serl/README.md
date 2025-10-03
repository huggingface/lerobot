# Grid HIL SERL Environment

This example demonstrates a custom Mujoco environment with an 8x8 grid world for robotic manipulation tasks using HIL (Human-in-the-Loop) SERL (Sample-Efficient Reinforcement Learning).

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

### Basic Usage
```bash
cd examples/grid_hil_serl
python grid_cube_randomizer.py
```

### Command Line Options
```bash
# Disable automatic image saving
python grid_cube_randomizer.py --no-save

# Change randomization interval (seconds)
python grid_cube_randomizer.py --interval 2.0

# Custom scene file
python grid_cube_randomizer.py --xml custom_scene.xml
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

## Integration with LeRobot

This environment can be integrated with LeRobot's HIL-SERL framework by:

1. Creating a custom environment class that inherits from `FrankaGymEnv`
2. Implementing the observation and action spaces
3. Adding reward functions for manipulation tasks
4. Using the grid positioning system for task generation

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
