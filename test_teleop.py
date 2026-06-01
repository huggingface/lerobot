from lerobot.robots.so_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.config_so101_follower import SO101FollowerConfig

robot = SO101Follower(
    SO101FollowerConfig(
        port="/dev/ttyACM1",
        id="kibub0",
    )
)

robot.connect()

try:
    while True:
        cmd = input("joint delta (example: 0 5): ")

        joint, delta = map(float, cmd.split())

        obs = robot.get_observation()
        pos = obs["joint_position"]

        pos[int(joint)] += delta

        robot.send_action(pos)

except KeyboardInterrupt:
    pass

finally:
    robot.disconnect()
