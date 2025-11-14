#!/usr/bin/env env python3
"""
Test script for EarthRover Mini with keyboard teleop and camera streaming.

Controls:
- W: Forward
- S: Backward
- A: Turn left (with slight forward motion)
- D: Turn right (with slight forward motion)
- Q: Rotate left in place
- E: Rotate right in place
- Space: Stop
- +/=: Increase speed
- -: Decrease speed
- ESC: Exit

Press 'q' in the camera window to stop streaming.
"""

import logging
import time
import sys
import cv2

from lerobot.robots.earthrover_mini_plus import EarthRover_Mini
from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlusConfig
from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

UPDATE_RATE = 0.03  # 30ms between commands


def print_controls():
    """Print control instructions"""
    print("\n" + "="*60)
    print("EarthRover Mini Keyboard Teleop Test")
    print("="*60)
    print("\nMovement Controls:")
    print("  W          - Move forward")
    print("  S          - Move backward")
    print("  A          - Turn left (with slight forward)")
    print("  D          - Turn right (with slight forward)")
    print("  Q          - Rotate left in place")
    print("  E          - Rotate right in place")
    print("  SPACE      - Stop all movement")
    print("\nSpeed Controls:")
    print("  + or =     - Increase speed")
    print("  -          - Decrease speed")
    print("\nExit:")
    print("  ESC        - Disconnect and exit")
    print("  q (in CV window) - Stop camera stream")
    print("="*60 + "\n")

#main test
def main():
    print_controls()
    
    # Initialize robot configuration
    robot_config = EarthRoverMiniPlusConfig()
    robot = EarthRover_Mini(robot_config)
    
    # Initialize keyboard teleop
    teleop_config = KeyboardRoverTeleopConfig(
    )
    teleop = KeyboardRoverTeleop(teleop_config)
    
    try:
        # Connect to robot
        logger.info("Connecting to EarthRover Mini...")
        robot.connect(calibrate=False)
        logger.info("Robot connected successfully!")
        
        # Connect keyboard teleop
        logger.info("Connecting keyboard teleop...")
        teleop.connect()
        logger.info("Keyboard teleop connected!")
        
        # Start camera streaming in separate thread
        logger.info("Starting camera stream...")
        robot.start_camera_stream()
        logger.info("Camera stream started!")
        
        logger.info("\nRobot ready! Use WASD keys to control.")
        logger.info("Current speeds - Linear: {}, Angular: {}".format(
            teleop.current_linear_speed, 
            teleop.current_angular_speed
        ))
        
        # Main control loop
        last_action = {"linear_velocity": 0.0, "angular_velocity": 0.0}
        
        while robot.is_connected and teleop.is_connected:
        
            try:
                # Get action from keyboard
                action = teleop.get_action()
                
                # Only send action if it changed (reduces unnecessary commands)
                if action != last_action:
                    # Send action to robot
                    robot.send_action(action)
                    
                    # Log action if not zero
                    if action["linear_velocity"] != 0 or action["angular_velocity"] != 0:
                        logger.info(f"Action: linear={action['linear_velocity']:.1f}, "
                                  f"angular={action['angular_velocity']:.1f}")
                    
                    last_action = action
                
                # Sleep to maintain update rate
                time.sleep(UPDATE_RATE)
                
            except KeyboardInterrupt:
                logger.info("\nKeyboard interrupt received. Stopping...")
                break
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                break

    except Exception as e:
        logger.error(f"Error during operation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        
        # Stop the robot
        try:
            stop_action = {"linear_velocity": 0.0, "angular_velocity": 0.0}
            robot.send_action(stop_action)
            logger.info("Robot stopped.")
        except:
            pass
        
        # Close camera stream
        try:
            robot.close_camera_stream()
            logger.info("Camera stream closed.")
        except:
            pass
        
        # Disconnect teleop
        try:
            if teleop.is_connected:
                teleop.disconnect()
            logger.info("Teleop disconnected.")
        except:
            pass
        
        # Disconnect robot
        try:
            if robot.is_connected:
                robot.disconnect()
            logger.info("Robot disconnected.")
        except:
            pass
        
        logger.info("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)