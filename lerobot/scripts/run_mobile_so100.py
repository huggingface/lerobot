import base64
import json
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import cv2
import torch
import zmq

from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode
from lerobot.common.robot_devices.robots.mobileso100 import MobileSO100

# Setup Motors Bus, Calibration, and Robot
motor_config = FeetechMotorsBusConfig(
    port="/dev/ttyACM0",
    motors={
        # Arm joints
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
        # Wheel motors (do not calibrate these)
        "wheel_1": (7, "sts3215"),
        "wheel_2": (8, "sts3215"),
        "wheel_3": (9, "sts3215"),
    },
)
motors_bus = FeetechMotorsBus(motor_config)
motors_bus.connect()

# Calibration directory for the so100 configuration
calibration_dir: str = ".cache/calibration/so100"
robot = MobileSO100(motors_bus)

# Set Up ZeroMQ Sockets
context = zmq.Context()

cmd_socket = context.socket(zmq.PULL)
cmd_socket.setsockopt(zmq.CONFLATE, 1)
cmd_socket.bind("tcp://*:5555")

video_socket = context.socket(zmq.PUB)
video_socket.setsockopt(zmq.SNDHWM, 1)
video_socket.bind("tcp://*:5556")

print("Robot server started, waiting for commands and streaming video...")

# Setup Camera
cap = cv2.VideoCapture("/dev/video0")  # Use: v4l2-ctl --list-devices
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define list of arm motor ids
arm_motor_ids = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def calibrate_follower_arm(motors_bus, calib_dir_str):
    """
    Calibrates the follower arm as a whole (producing one calibration file)
    and then applies the calibration data to each joint of the arm.
    (Wheel motors are not calibrated.)
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
        print(f"[INFO] Loaded calibration for follower arm from {calib_file}")
    else:
        print(
            "[INFO] Calibration file for follower arm not found. Running manual calibration for follower arm..."
        )
        calibration = run_arm_manual_calibration(motors_bus, "so100", "follower_arm", "follower")
        print(f"[INFO] Calibration complete for follower arm. Saving calibration data to {calib_file}")
        with open(calib_file, "w") as f:
            json.dump(calibration, f)
    try:
        motors_bus.set_calibration(calibration)
        print("[INFO] Applied calibration for follower arm.")
    except Exception as e:
        print(f"[WARNING] Could not apply calibration for follower arm: {e}")


calibrate_follower_arm(motors_bus, calibration_dir)

for motor in arm_motor_ids:
    motors_bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor)

# Camera Capture and Encoding Thread
# Create a queue to hold the latest encoded frame.
frame_queue = Queue(maxsize=1)


def camera_capture(cap, frame_queue):
    """Continuously capture and encode frames, storing the latest encoded JPEG (base64) in a queue."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            time.sleep(0.005)
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("Failed to encode image.")
            continue

        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(jpg_as_text)
        time.sleep(0.005)


# Start the camera capture thread.
capture_thread = Thread(target=camera_capture, args=(cap, frame_queue), daemon=True)
capture_thread.start()

# Main Loop
start_time = time.perf_counter()
iteration_count = 0
total_elapsed_time = 0.0
print_interval = 100
last_cmd_time = time.time()

last_jpg_as_text = None

try:
    while True:
        loop_start_time = time.perf_counter()
        now = time.time()

        # Process any incoming commands from the ZeroMQ cmd_socket.
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
                        print(f"[ERROR] Invalid arm_positions format: {arm_positions}")
                    elif len(arm_positions) < len(arm_motor_ids):
                        print(
                            f"[WARNING] Received arm_positions list of length {len(arm_positions)} but expected {len(arm_motor_ids)}"
                        )
                    else:
                        for motor, pos in zip(arm_motor_ids, arm_positions, strict=False):
                            motors_bus.write("Goal_Position", pos, motor)
                # Process wheel commands.
                if "raw_velocity" in data:
                    raw_command = data["raw_velocity"]
                    command_speeds = [int(raw_command.get(f"wheel_{i}", 0)) for i in [1, 2, 3]]
                    robot.set_velocity(command_speeds)
                    last_cmd_time = now
            except Exception as e:
                print(f"Error parsing message: {e}")

        # Stop the robot if no command is received for a period.
        if now - last_cmd_time > 0.5:
            robot.stop()
            last_cmd_time = now

        current_velocity = robot.read_velocity()

        # Read the current arm positions.
        follower_arm_state = []
        for motor in arm_motor_ids:
            try:
                pos = motors_bus.read("Present_Position", motor)
                follower_arm_state.append(torch.from_numpy(pos))
            except Exception as e:
                print(f"Error reading motor {motor}: {e}")

        follower_arm_state = torch.cat(follower_arm_state) if follower_arm_state else torch.tensor([])
        arm_state_list = follower_arm_state.tolist() if follower_arm_state.numel() > 0 else []

        try:
            # Try to get a new frame; wait up to 5ms
            jpg_as_text = frame_queue.get(timeout=0.005)
            last_jpg_as_text = jpg_as_text
        except Empty:
            # If no new frame is available, use the previous one if available
            if last_jpg_as_text is not None:
                jpg_as_text = last_jpg_as_text
            else:
                # Or—if no frame has ever been captured—block until one arrives.
                jpg_as_text = frame_queue.get()
                last_jpg_as_text = jpg_as_text

        # Prepare and send the observation message.
        observation = {
            "image": jpg_as_text,
            "present_speed": {
                "1": int(current_velocity.get("1", 0)),
                "2": int(current_velocity.get("2", 0)),
                "3": int(current_velocity.get("3", 0)),
            },
            "follower_arm_state": arm_state_list,
        }
        observation_json = json.dumps(observation)
        video_socket.send_string(observation_json)

        loop_end_time = time.perf_counter()
        iteration_time = loop_end_time - loop_start_time
        total_elapsed_time += iteration_time
        iteration_count += 1
        if iteration_count % print_interval == 0:
            average_iteration_time = total_elapsed_time / iteration_count
            frequency_hz = 1.0 / average_iteration_time
            print(f"Average loop frequency: {frequency_hz:.2f} Hz over {iteration_count} iterations")

        time.sleep(max(0, 0.001 - (time.perf_counter() - loop_start_time)))

except KeyboardInterrupt:
    print("Shutting down robot server.")

finally:
    robot.stop()
    motors_bus.disconnect()
    cmd_socket.close()
    video_socket.close()
    context.term()
    cap.release()
