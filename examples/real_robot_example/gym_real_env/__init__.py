from gymnasium.envs.registration import register

register(
    id="gym_real_env/RealEnv-v0",
    entry_point="gym_real_env.env:RealEnv",
    max_episode_steps=300,
    nondeterministic=True,
)
