import signal
import sys
import json
import time
import select
import termios
import tty
from datetime import datetime
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

# Recording data storage
recorded_positions = []
recording_count = 0

def cleanup_and_exit(signum=None, frame=None):
    """Disable torque, save recordings, and disconnect motors on exit."""
    print("\nSaving recordings and cleaning up...")
    
    # Save recorded positions to JSON file
    if recorded_positions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/tianchongj/workspace/lerobot/dev/teleop_recordings_{timestamp}.json"
        
        recording_data = {
            "timestamp": timestamp,
            "total_recordings": len(recorded_positions),
            "motor_names": {
                "leader": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                "follower": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            },
            "recordings": recorded_positions
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(recording_data, f, indent=2)
            print(f"✓ Saved {len(recorded_positions)} recordings to {filename}")
        except Exception as e:
            print(f"✗ Failed to save recordings: {e}")
    else:
        print("No recordings to save")
    
    # Disable torque and disconnect
    print("Disabling torque on both devices...")
    try:
        robot.bus.disable_torque()
        teleop_device.bus.disable_torque()
        robot.disconnect()
        teleop_device.disconnect()
        print("✓ Motors disabled and disconnected")
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    sys.exit(0)

def record_current_positions():
    """Record current positions of both leader and follower."""
    global recording_count
    
    try:
        # Read positions from both devices
        leader_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
        follower_positions = robot.bus.sync_read("Present_Position", normalize=False)
        
        # Create recording entry
        recording_entry = {
            "recording_id": recording_count + 1,
            "timestamp": datetime.now().isoformat(),
            "leader_positions": leader_positions,
            "follower_positions": follower_positions
        }
        
        recorded_positions.append(recording_entry)
        recording_count += 1
        
        print(f"✓ Recording #{recording_count} saved")
        print(f"  Leader:   {leader_positions}")
        print(f"  Follower: {follower_positions}")
        
    except Exception as e:
        print(f"✗ Failed to record positions: {e}")

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

print("Connecting to devices...")
robot.connect(calibrate=True)
teleop_device.connect(calibrate=True)

print("=" * 60)
print("TELEOPERATION WITH RECORDING")
print("=" * 60)
print("Controls:")
print("  'r' - Record current joint positions")
print("  'q' - Quit and save recordings")
print("  Ctrl+C - Emergency exit")
print(f"Recordings will be saved to: teleop_recordings_YYYYMMDD_HHMMSS.json")
print("=" * 60)
print("Teleoperation started...")

# Setup terminal for non-blocking input
stdin_fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(stdin_fd)

try:
    tty.setraw(sys.stdin.fileno())
    
    while True:
        # Check for keyboard input (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1).lower()
            
            if char == 'r':
                record_current_positions()
            elif char == 'q' or char == '\x03':  # 'q' or Ctrl+C
                break
        
        # Continue teleoperation
        try:
            # Read raw positions from leader without normalization
            raw_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
            
            # Send raw positions to follower
            robot.bus.sync_write("Goal_Position", raw_positions, normalize=False)
            
        except Exception as e:
            print(f"Teleoperation error: {e}")
            break
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.001)  # 1ms delay for ~1000Hz operation

finally:
    # Restore terminal settings
    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
    cleanup_and_exit()