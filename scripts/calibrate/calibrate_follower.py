from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="COM6",
    id="DI_VLA_FOLLOWER",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()
