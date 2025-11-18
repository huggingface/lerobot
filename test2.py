from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.observation_processor import LiberoProcessorStep
config = make_env_config("libero", task="libero_spatial")
envs_dict = make_env(config)
env = envs_dict["libero_spatial"][0]

seed = 42

# First rollout
obs1, info1 = env.reset(seed=seed)

observation = preprocess_observation(obs1)
observation = add_envs_task(env, observation)

libero_preprocessor = PolicyProcessorPipeline(
    steps=[
        LiberoProcessorStep(),
    ]
)
observation = libero_preprocessor(observation)
