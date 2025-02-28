import base64
import json
import threading
import time
from pathlib import Path

import cv2
import zmq

from lerobot.common.robot_devices.robots.mobile_manipulator import LeKiwi


def setup_zmq_sockets(config):
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.bind(f"tcp://*:{config.port}")

    video_socket = context.socket(zmq.PUSH)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.bind(f"tcp://*:{config.video_port}")

    return context, cmd_socket, video_socket


def run_camera_capture(cameras, images_lock, latest_images_dict, stop_event):
    while not stop_event.is_set():
        local_dict = {}
        for name, cam in cameras.items():
            frame = cam.async_read()
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                local_dict[name] = base64.b64encode(buffer).decode("utf-8")
            else:
                local_dict[name] = ""
        with images_lock:
            latest_images_dict.update(local_dict)
        time.sleep(0.01)


def calibrate_follower_arm(motors_bus, calib_dir_str):
    """
    Calibrates the follower arm. Attempts to load an existing calibration file;
    if not found, runs manual calibration and saves the result.
    """
    calib_dir = Path(calib_dir_str)
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_file = calib_dir / "main_follower.json"
    try:
        from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
    except ImportError:
        print("[WARNING] Calibration function not available. Skipping calibration.")
        return

    if calib_file.exists():
        with open(calib_file) as f:
            calibration = json.load(f)
        print(f"[INFO] Loaded calibration from {calib_file}")
    else:
        print("[INFO] Calibration file not found. Running manual calibration...")
        calibration = run_arm_manual_calibration(motors_bus, "lekiwi", "follower_arm", "follower")
        print(f"[INFO] Calibration complete. Saving to {calib_file}")
        with open(calib_file, "w") as f:
            json.dump(calibration, f)
    try:
        motors_bus.set_calibration(calibration)
        print("[INFO] Applied calibration for follower arm.")
    except Exception as e:
        print(f"[WARNING] Could not apply calibration: {e}")


def run_lekiwi(robot_config):
    """
    Runs the LeKiwi robot:
      - Sets up cameras and connects them.
      - Initializes the follower arm motors.
      - Calibrates the follower arm if necessary.
      - Creates ZeroMQ sockets for receiving commands and streaming observations.
      - Processes incoming commands (arm and wheel commands) and sends back sensor and camera data.
    """
    # Import helper functions and classes
    from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode

    # Initialize cameras from the robot configuration.
    cameras = make_cameras_from_configs(robot_config.cameras)
    for cam in cameras.values():
        cam.connect()

    # Initialize the motors bus using the follower arm configuration.
    motor_config = robot_config.follower_arms.get("main")
    if motor_config is None:
        print("[ERROR] Follower arm 'main' configuration not found.")
        return
    motors_bus = FeetechMotorsBus(motor_config)
    motors_bus.connect()

    # Calibrate the follower arm.
    calibrate_follower_arm(motors_bus, robot_config.calibration_dir)

    # Create the LeKiwi robot instance.
    robot = LeKiwi(motors_bus)

    # Define the expected arm motor IDs.
    arm_motor_ids = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Disable torque for each arm motor.
    for motor in arm_motor_ids:
        motors_bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor)

    # Set up ZeroMQ sockets.
    context, cmd_socket, video_socket = setup_zmq_sockets(robot_config)

    # Start the camera capture thread.
    latest_images_dict = {}
    images_lock = threading.Lock()
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=run_camera_capture, args=(cameras, images_lock, latest_images_dict, stop_event), daemon=True
    )
    cam_thread.start()

    last_cmd_time = time.time()
    print("LeKiwi robot server started. Waiting for commands...")

    try:
        while True:
            loop_start_time = time.time()

            # Process incoming commands (non-blocking).
            while True:
                try:
                    msg = cmd_socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    data = json.loads(msg)
                    # Process arm position commands.
                    if "arm_positions" in data:
                        arm_positions = data["arm_positions"]
                        if not isinstance(arm_positions, list):
                            print(f"[ERROR] Invalid arm_positions: {arm_positions}")
                        elif len(arm_positions) < len(arm_motor_ids):
                            print(
                                f"[WARNING] Received {len(arm_positions)} arm positions, expected {len(arm_motor_ids)}"
                            )
                        else:
                            for motor, pos in zip(arm_motor_ids, arm_positions, strict=False):
                                motors_bus.write("Goal_Position", pos, motor)
                    # Process wheel (base) commands.
                    if "raw_velocity" in data:
                        raw_command = data["raw_velocity"]
                        # Expect keys: "left_wheel", "back_wheel", "right_wheel".
                        command_speeds = [
                            int(raw_command.get("left_wheel", 0)),
                            int(raw_command.get("back_wheel", 0)),
                            int(raw_command.get("right_wheel", 0)),
                        ]
                        robot.set_velocity(command_speeds)
                        last_cmd_time = time.time()
                except Exception as e:
                    print(f"[ERROR] Parsing message failed: {e}")

            # Watchdog: stop the robot if no command is received for over 0.5 seconds.
            now = time.time()
            if now - last_cmd_time > 0.5:
                robot.stop()
                last_cmd_time = now

            # Read current wheel speeds from the robot.
            current_velocity = robot.read_velocity()

            # Read the follower arm state from the motors bus.
            follower_arm_state = []
            for motor in arm_motor_ids:
                try:
                    pos = motors_bus.read("Present_Position", motor)
                    # Convert the position to a float (or use as is if already numeric).
                    follower_arm_state.append(float(pos) if not isinstance(pos, (int, float)) else pos)
                except Exception as e:
                    print(f"[ERROR] Reading motor {motor} failed: {e}")

            # Get the latest camera images.
            with images_lock:
                images_dict_copy = dict(latest_images_dict)

            # Build the observation dictionary.
            observation = {
                "images": images_dict_copy,
                "present_speed": current_velocity,
                "follower_arm_state": follower_arm_state,
            }
            # Send the observation over the video socket.
            video_socket.send_string(json.dumps(observation))

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time
            time.sleep(
                max(0.033 - elapsed, 0)
            )  # If robot jitters increase the sleep and monitor cpu load with `top` in cmd
    except KeyboardInterrupt:
        print("Shutting down LeKiwi server.")
    finally:
        stop_event.set()
        cam_thread.join()
        robot.stop()
        motors_bus.disconnect()
        cmd_socket.close()
        video_socket.close()
        context.term()
