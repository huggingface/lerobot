import signal
import sys
sys.path.insert(0, "/home/tianchongj/workspace/lerobot/src")
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="follower0",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="leader0",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

def cleanup_and_exit(signum=None, frame=None):
    """Disable torque and disconnect motors on exit."""
    print("\nDisabling torque on both devices")
    robot.bus.disable_torque()
    teleop_device.bus.disable_torque()
    robot.disconnect()
    teleop_device.disconnect()
    print("✓ Motors disabled")
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, cleanup_and_exit)

robot.connect(calibrate=True)
teleop_device.connect(calibrate=True)

print("Teleoperation started. Press Ctrl+C to exit.")

while True:
    # Read raw positions without normalization
    raw_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
    
    # Send raw positions to follower
    robot.bus.sync_write("Goal_Position", raw_positions, normalize=False)