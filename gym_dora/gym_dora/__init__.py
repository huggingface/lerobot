from gymnasium.envs.registration import register

register(
    id="gym_dora/DoraAloha-v0",
    entry_point="gym_dora.env:DoraEnv",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={"model": "aloha"},
)

register(
    id="gym_dora/DoraKoch-v0",
    entry_point="gym_dora.env:DoraEnv",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={"model": "koch"},
)
