from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

config = SO101LeaderConfig(
    port="COM5",
    id="DI_VLA_LEADER",
)
leader = SO101Leader(config)
leader.setup_motors()
