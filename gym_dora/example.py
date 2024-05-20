import gymnasium as gym

import gym_dora  # noqa: F401

env = gym.make("gym_dora/DoraAloha-v0", disable_env_checker=True)
obs = env.reset()

policy = ...  # make_policy

done = False
while not done:
    actions = policy.select_action(obs)
    observation, reward, terminated, truncated, info = env.step(actions)

    done = terminated | truncated | done

env.close()
