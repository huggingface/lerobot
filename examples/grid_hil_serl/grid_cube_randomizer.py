#!/usr/bin/env python

"""
Random Grid Cube Spawner

This script loads the 8x8 grid scene and randomly positions a cube
in one of the 64 grid cells. The cube spawns at integer coordinates
within the grid boundaries.
"""

import numpy as np
import mujoco
import mujoco.viewer
import argparse
import time
from PIL import Image


def save_camera_view(model, data, filename="img.jpg"):
    """
    Save the current camera view to a JPEG image file.

    Args:
        model: Mujoco model
        data: Mujoco data
        filename: Output filename (default: img.jpg)
    """
    try:
        # Create a high-definition renderer for the current camera
        renderer = mujoco.Renderer(model, height=1080, width=1920)

        # Update the scene and render
        renderer.update_scene(data, camera="grid_camera")
        img = renderer.render()

        if img is not None:
            # Convert to PIL Image and save
            image = Image.fromarray(img)
            image.save(filename)
            print(f"Camera view saved to: {filename}")
        else:
            print("Warning: Could not capture camera view")

        # Clean up renderer (if close method exists)
        if hasattr(renderer, 'close'):
            renderer.close()

    except Exception as e:
        print(f"Error saving image: {e}")


def randomize_cube_position(model, data, grid_size=8):
    """
    Randomly position the cube in one of the grid cells.

    Args:
        model: Mujoco model
        data: Mujoco data
        grid_size: Size of the grid (8x8)
    """
    # For 8x8 grid: generate random cell indices from 0-7 for both x and y
    # This gives us coordinates for each of the 64 grid cells
    x_cell = np.random.randint(0, 8)  # 0 to 7 inclusive
    y_cell = np.random.randint(0, 8)  # 0 to 7 inclusive

    # Convert cell indices to center positions (offset by 0.5 from grid lines)
    # X: left(0) = -3.5, right(7) = 3.5
    x_pos = (x_cell - grid_size // 2) + 0.5
    # Y: top(0) = 3.5, bottom(7) = -3.5 (flipped coordinate system)
    y_pos = (grid_size // 2 - y_cell) - 0.5

    print(f"Spawning cube at grid cell ({x_cell}, {y_cell}) -> position ({x_pos}, {y_pos})")

    # Set the cube position and velocity (free joint has 6 DOF: 3 pos + 3 vel)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")

    # Set position (x, y, z) - keep rotation as identity (0, 0, 0)
    data.qpos[model.jnt_qposadr[cube_joint_id]:model.jnt_qposadr[cube_joint_id] + 6] = [x_pos, y_pos, 0.5, 0, 0, 0]

    # Reset velocity to zero (linear and angular velocities)
    data.qvel[model.jnt_dofadr[cube_joint_id]:model.jnt_dofadr[cube_joint_id] + 6] = [0, 0, 0, 0, 0, 0]

    return x_pos, y_pos


def run_grid_viewer(xml_path, randomize_interval=2.0, auto_save=True):
    """
    Run the grid viewer with random cube positioning.

    Args:
        xml_path: Path to the XML scene file
        randomize_interval: How often to randomize cube position (seconds)
        auto_save: Whether to automatically save camera view after each repositioning
    """
    print(f"Loading scene: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("\n" + "="*50)
    print("8x8 Grid Cube Randomizer")
    print("="*50)
    print("This scene shows an 8x8 grid with a randomly positioned cube.")
    print(f"Cube position randomizes every {randomize_interval} seconds.")
    print()
    print("Controls:")
    print("  R: Manually randomize cube position")
    print("  S: Save current camera view to img.jpg")
    print("  Space: Pause/unpause")
    print("  Esc: Exit")
    print("  Camera: Mouse controls for rotation/zoom")
    print("="*50)

    last_randomize_time = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial randomization
        x, y = randomize_cube_position(model, data)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            current_time = time.time()

            # Auto-randomize every few seconds
            if current_time - last_randomize_time > randomize_interval:
                x, y = randomize_cube_position(model, data)
                mujoco.mj_forward(model, data)
                # Force viewer to update the scene
                viewer.sync()
                # Save the current camera view if auto_save is enabled
                if auto_save:
                    save_camera_view(model, data, "img.jpg")
                last_randomize_time = current_time

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="8x8 Grid Cube Randomizer")
    parser.add_argument("--xml", type=str, default="grid_scene.xml",
                       help="Path to XML scene file")
    parser.add_argument("--interval", type=float, default=3.0,
                       help="Randomization interval in seconds")
    parser.add_argument("--no-save", action="store_true",
                       help="Disable automatic saving of camera views")

    args = parser.parse_args()

    try:
        run_grid_viewer(args.xml, args.interval, not args.no_save)
    except FileNotFoundError:
        print(f"Error: Could not find XML file '{args.xml}'")
        print("Make sure the XML file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
