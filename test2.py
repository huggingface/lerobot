from lerobot.envs.factory import make_env, make_env_config
config = make_env_config("libero", task="libero_spatial")
envs_dict = make_env(config)
env = envs_dict["libero_spatial"][0]

seed = 42

# First rollout
obs1, info1 = env.reset(seed=seed)
