#!/usr/bin/env python

"""
Test SO-101 MuJoCo robot with SO101KeyboardTeleop.
This mimics how lerobot_record will use the robot + teleop.
Uses manual GLFW rendering like orient_down.py for full keyboard control.
"""

import logging
import time
from pathlib import Path

import glfw
import mujoco as mj
from lerobot.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from lerobot.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.keyboard.teleop_so101_keyboard import SO101KeyboardTeleop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test robot with keyboard teleop (like lerobot_record does)."""
    # Create robot config
    robot_config = SO101MujocoConfig(
        id="so101_test",
        xml_path=Path("gym-hil/gym_hil/assets/SO101/pick_scene.xml"),
    )

    # Create teleop config (no parameters needed)
    teleop_config = KeyboardTeleopConfig()

    # Create robot and teleop
    robot = SO101MujocoRobot(robot_config)
    teleop = SO101KeyboardTeleop(teleop_config)

    # Connect both
    logger.info("Connecting robot...")
    robot.connect()
    logger.info("Connecting teleop...")
    teleop.connect()

    logger.info("\n" + "="*60)
    logger.info("SO-101 Robot + Keyboard Teleop Test")
    logger.info("Controls (matching orient_down.py):")
    logger.info("  Arrow Up/Down: Move +X / -X (world frame)")
    logger.info("  Arrow Left/Right: Move -Y / +Y (world frame)")
    logger.info("  Shift / Shift+R: Move +Z / -Z")
    logger.info("  [ / ] or Z / X: Wrist roll left/right")
    logger.info("  , / .: Gripper open/close")
    logger.info("  ESC: Exit (or close window)")
    logger.info("="*60 + "\n")

    # Close robot's renderer to avoid conflicts with GLFW
    # (Robot uses offscreen renderer for camera images, GLFW needs OpenGL context)
    robot._renderer.close()

    # Initialize GLFW for manual rendering (like orient_down.py)
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    # Create window
    window_width, window_height = 1280, 720
    window = glfw.create_window(
        window_width, window_height,
        "SO-101 Robot + Keyboard Teleop Test",
        None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Set up MuJoCo rendering
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    cam.distance = 1.3
    cam.azimuth = 140
    cam.elevation = -20

    scene = mj.MjvScene(robot.model, maxgeom=10000)
    ctx = mj.MjrContext(robot.model, mj.mjtFontScale.mjFONTSCALE_150)

    # Control loop
    step_count = 0
    last_print_time = time.time()

    try:
        while not glfw.window_should_close(window) and teleop.is_connected:
            step_start = time.time()

            # Get keyboard input from teleop (mimics lerobot_record line 324)
            keyboard_action = teleop.get_action()

            # Convert to base action (mimics lerobot_record line 333)
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            # Send action to robot (mimics lerobot_record line 356)
            robot.send_action(base_action)

            # Get observation (skip camera since we're using GLFW for visualization)
            # In actual lerobot_record, camera images are captured
            obs = {
                "ee.pos_x": robot.data.site_xpos[robot.ee_site_id][0],
                "ee.pos_y": robot.data.site_xpos[robot.ee_site_id][1],
                "ee.pos_z": robot.data.site_xpos[robot.ee_site_id][2],
                "gripper.pos": robot.data.qpos[robot.dof_ids["gripper"]],
            }

            # Render scene (manual GLFW rendering)
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            mj.mjv_updateScene(
                robot.model, robot.data, opt, None, cam,
                mj.mjtCatBit.mjCAT_ALL, scene
            )
            mj.mjr_render(viewport, scene, ctx)
            glfw.swap_buffers(window)
            glfw.poll_events()

            # Print status every second
            if time.time() - last_print_time > 1.0:
                ee_pos = obs["ee.pos_x"], obs["ee.pos_y"], obs["ee.pos_z"]

                # Print joint angles for finding good configurations
                joint_angles = [
                    robot.data.qpos[robot.dof_ids["shoulder_pan"]],
                    robot.data.qpos[robot.dof_ids["shoulder_lift"]],
                    robot.data.qpos[robot.dof_ids["elbow_flex"]],
                    robot.data.qpos[robot.dof_ids["wrist_flex"]],
                    robot.data.qpos[robot.dof_ids["wrist_roll"]],
                    robot.data.qpos[robot.dof_ids["gripper"]],
                ]

                logger.info(
                    f"Step {step_count:5d} | "
                    f"EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}] | "
                    f"Gripper: {obs['gripper.pos']:5.3f}"
                )
                logger.info(
                    f"  Joints [pan, lift, elbow, wrist_flex, wrist_roll, gripper]: "
                    f"[{joint_angles[0]:6.3f}, {joint_angles[1]:6.3f}, {joint_angles[2]:6.3f}, "
                    f"{joint_angles[3]:6.3f}, {joint_angles[4]:6.3f}, {joint_angles[5]:6.3f}]"
                )
                last_print_time = time.time()

            step_count += 1

            # Maintain 30 FPS
            elapsed = time.time() - step_start
            sleep_time = (1.0 / robot_config.record_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        glfw.terminate()
        robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        logger.info("Test completed")


if __name__ == "__main__":
    main()
