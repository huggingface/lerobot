import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from termcolor import colored

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem5A7B2890981", help="Serial port")
    parser.add_argument("--id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--camera", type=int, default=0, help="Webcam ID")
    args = parser.parse_args()

    # 1. Initialize Robot
    print(colored("Initializing robot...", "cyan"))
    robot_config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        print(colored("Failed to connect to robot.", "red", attrs=["bold"]))
        return

    # 2. Initialize Kinematics Solver
    print(colored("Initializing kinematics solver...", "cyan"))
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(colored(f"Error opening camera {args.camera}.", "red", attrs=["bold"]))
        robot.disconnect()
        return

    # 4. State Variables
    enabled = False
    latched_robot_pose = None
    latched_human_wrist = None
    latched_robot_joints = None
    
    # Scale factors for translation (MediaPipe normalized coords to meters)
    SCALE_X = 0.5  # Left/Right 
    SCALE_Y = 0.5  # Up/Down
    SCALE_Z = 0.5  # Depth

    # Gripper state
    obs = robot.get_observation()
    # Assume gripper is the last motor
    gripper_name = "gripper"
    gripper_pos = obs[f"{gripper_name}.pos"]
    GRIPPER_STEP = 20.0
    
    print(colored("==========================================================", "green"))
    print(colored("  HUMAN ARM TELEOP (IK) - MEDIA_PIPE POSE", "green", attrs=["bold"]))
    print(colored("==========================================================", "green"))
    print("Controls:")
    print("  [Space] : Toggle tracking on/off (latches current position)")
    print("  [O]     : Open gripper")
    print("  [C]     : Close gripper")
    print("  [Q]     : Quit")
    print("Ensure you are visible in the camera before enabling.")

    fps = 30
    dt = 1.0 / fps

    # Init MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            t0 = time.perf_counter()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect pose landmarks
            detection_result = detector.detect(mp_image)

            current_human_wrist = None
            if len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                # Right Wrist is landmark index 16
                right_wrist = landmarks[16]
                current_human_wrist = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
                
                # Draw a circle on the right wrist for visualization
                h, w, c = image.shape
                cx, cy = int(right_wrist.x * w), int(right_wrist.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            # Handle Keyboard Input from OpenCV window
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                enabled = not enabled
                if enabled:
                    if current_human_wrist is None:
                        print(colored("Cannot enable: No human wrist detected!", "red"))
                        enabled = False
                    else:
                        print(colored("Tracking ENABLED. Latching poses.", "green", attrs=["bold"]))
                        latched_human_wrist = current_human_wrist.copy()
                        
                        # Get current robot pose
                        obs = robot.get_observation()
                        q_curr = np.array([obs[f"{m}.pos"] for m in robot.bus.motors.keys()], dtype=float)
                        latched_robot_pose = kinematics_solver.forward_kinematics(q_curr)
                        latched_robot_joints = q_curr
                else:
                    print(colored("Tracking DISABLED.", "yellow", attrs=["bold"]))
            
            # Gripper control (runs independent of tracking enabled)
            if key == ord('o'):
                gripper_pos -= GRIPPER_STEP
            elif key == ord('c'):
                gripper_pos += GRIPPER_STEP
            
            # Keep gripper pos in typical 0-100 bounds, adjust if your arm is different
            gripper_pos = np.clip(gripper_pos, 0, 100)

            # --- Robot Control Logic ---
            if enabled and current_human_wrist is not None and latched_robot_pose is not None:
                # Calculate delta in MediaPipe space
                delta = current_human_wrist - latched_human_wrist
                
                # MediaPipe Coords:
                # X: left to right (normalized 0 to 1) -> Robot Y (positive is left usually, wait: let's test this)
                # Y: top to bottom (normalized 0 to 1) -> Robot Z (positive is up)
                # Z: depth (normalized, smaller is closer) -> Robot X (positive is forward)
                
                # Map MediaPipe delta to Robot delta
                # You might need to adjust the signs depending on camera setup (mirrored vs not)
                robot_dx = -delta[2] * SCALE_Z  # Depth -> X (forward)
                robot_dy = -delta[0] * SCALE_X  # Right/Left -> Y
                robot_dz = -delta[1] * SCALE_Y  # Down/Up -> Z
                
                mapped_delta = np.array([robot_dx, robot_dy, robot_dz])
                
                # Compute desired pose
                t_des = latched_robot_pose.copy()
                t_des[:3, 3] = t_des[:3, 3] + mapped_delta
                
                # Solve IK
                # Using latched_robot_joints as initial guess instead of current to prevent drift
                assert latched_robot_joints is not None, "latched_robot_joints should not be None when enabled"
                q_target = kinematics_solver.inverse_kinematics(latched_robot_joints, t_des)
                
                # Send Action
                action = {}
                for i, m in enumerate(robot.bus.motors.keys()):
                    if m != gripper_name:
                        action[f"{m}.pos"] = q_target[i]
                action[f"{gripper_name}.pos"] = gripper_pos
                
                robot.send_action(action)
                
                # Update latched joints for next IK guess to be smooth
                latched_robot_joints = q_target
                
                # Draw target on screen
                cv2.putText(image, f"dx:{robot_dx:.2f} dy:{robot_dy:.2f} dz:{robot_dz:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Just send gripper action if not tracking arm
                obs = robot.get_observation()
                action = {f"{m}.pos": obs[f"{m}.pos"] for m in robot.bus.motors.keys()}
                action[f"{gripper_name}.pos"] = gripper_pos
                robot.send_action(action)

            # Draw Status
            status_color = (0, 255, 0) if enabled else (0, 0, 255)
            status_text = "TRACKING: ON" if enabled else "TRACKING: OFF"
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('MediaPipe Pose - Robot Teleop', image)
            
            # Precise timing
            elapsed = time.perf_counter() - t0
            precise_sleep(max(dt - elapsed, 0.0))

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()


if __name__ == "__main__":
    main()
