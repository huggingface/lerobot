from gymnasium.envs.registration import register

register(
    id="gym_aloha/AlohaInsertion-v0",
    entry_point="lerobot.common.envs.aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "state", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaTransferCube-v0",
    entry_point="lerobot.common.envs.aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "state", "task": "transfer_cube"},
)
