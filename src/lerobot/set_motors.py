from lerobot.common.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.common.robots.so101_follower_torque.so101_follower_t import SO101FollowerT

config = SO101FollowerTConfig(
    port="/dev/tty.usbmodem58760428721",
    id="my_awesome_follower_arm",
)
follower = SO101FollowerT(config)
follower.setup_motors()
