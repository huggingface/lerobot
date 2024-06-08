from gymnasium.envs.registration import register

register(
    id="gym_real_world/RealEnv-v0",
    entry_point="gym_real_world.gym_environment:RealEnv",
    max_episode_steps=300,
    nondeterministic=True,
)
