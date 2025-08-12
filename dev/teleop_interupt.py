import signal
import sys
import threading
import time
from pynput import keyboard, mouse
from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="follower10",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="leader10",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

# Global variables for position recording and fine-tuning
recording_mode = False  # True when using recorded positions, False when following leader
recorded_positions = None
selected_joint = None  # 1-6 for joint selection
position_lock = threading.Lock()

# Mapping from joint numbers to motor names
joint_to_motor = {
    1: "shoulder_pan",
    2: "shoulder_lift", 
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper"
}

def cleanup_and_exit(signum=None, frame=None):
    """Disable torque and disconnect motors on exit."""
    print("\nDisabling torque on both devices")
    robot.bus.disable_torque()
    teleop_device.bus.disable_torque()
    robot.disconnect()
    teleop_device.disconnect()
    print("✓ Motors disabled")
    sys.exit(0)

def on_key_press(key):
    """Handle keyboard input for recording and joint selection."""
    global recording_mode, recorded_positions, selected_joint
    
    try:
        if key.char == 's':
            # Record current leader positions
            with position_lock:
                recorded_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
                recording_mode = True
                selected_joint = None
            print(f"✓ Positions recorded: {recorded_positions}")
            print("Follower will now hold this position. Press 1-6 to select joint for adjustment.")
            
        elif key.char in '123456':
            # Select joint for adjustment
            if recording_mode and recorded_positions is not None:
                with position_lock:
                    selected_joint = int(key.char)
                motor_name = joint_to_motor.get(selected_joint)
                print(f"✓ Joint {selected_joint} ({motor_name}) selected for adjustment. Use mouse scroll to adjust.")
            else:
                print("Please press 's' first to record positions.")
                
    except AttributeError:
        # Special keys (like arrow keys) don't have .char
        pass

def on_scroll(x, y, dx, dy):
    """Handle mouse scroll for fine-tuning selected joint."""
    global recorded_positions, selected_joint
    
    print(f"Scroll detected: dx={dx}, dy={dy}, recording_mode={recording_mode}, selected_joint={selected_joint}")
    
    if recording_mode and recorded_positions is not None and selected_joint is not None:
        with position_lock:
            # Adjust the selected joint position by scroll amount
            # Scale the scroll input to appropriate motor position units
            adjustment = dy * 1  # Increased for more noticeable changes
            motor_name = joint_to_motor.get(selected_joint)
            
            if motor_name and motor_name in recorded_positions:
                old_value = recorded_positions[motor_name]
                recorded_positions[motor_name] += adjustment
                print(f"Joint {selected_joint} ({motor_name}) adjusted: {old_value} -> {recorded_positions[motor_name]} (change: {adjustment})")
            else:
                print(f"Motor name '{motor_name}' not found in recorded_positions: {list(recorded_positions.keys())}")
    else:
        if not recording_mode:
            print("Not in recording mode - press 's' first")
        elif recorded_positions is None:
            print("No positions recorded")
        elif selected_joint is None:
            print("No joint selected - press 1-6 to select a joint")

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, cleanup_and_exit)

# Set up keyboard and mouse listeners
keyboard_listener = keyboard.Listener(on_press=on_key_press)
mouse_listener = mouse.Listener(on_scroll=on_scroll)

keyboard_listener.start()
mouse_listener.start()

robot.connect(calibrate=True)
teleop_device.connect(calibrate=True)

print("Teleoperation started.")
print("Commands:")
print("  's' - Record current leader positions and switch to hold mode")
print("  '1-6' - Select joint for adjustment (after recording):")
print("         1=shoulder_pan, 2=shoulder_lift, 3=elbow_flex")
print("         4=wrist_flex, 5=wrist_roll, 6=gripper")
print("  Mouse scroll - Adjust selected joint position")
print("  Ctrl+C - Exit")
print("\nCurrently following leader. Press 's' to record and hold position.")

while True:
    with position_lock:
        if recording_mode and recorded_positions is not None:
            # Use recorded/adjusted positions
            robot.bus.sync_write("Goal_Position", recorded_positions, normalize=False)
        else:
            # Follow leader in real-time
            raw_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
            robot.bus.sync_write("Goal_Position", raw_positions, normalize=False)
    
    # Small delay to prevent overwhelming the motors
    time.sleep(0.01)