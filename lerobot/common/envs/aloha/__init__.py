from gymnasium.envs.registration import register

register(
    id="gym_aloha/AlohaInsertion-v0",
    entry_point="lerobot.common.envs.aloha.env:AlohaEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaTransferCube-v0",
    entry_point="lerobot.common.envs.aloha.env:AlohaEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "task": "transfer_cube"},
)
