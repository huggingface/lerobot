import numpy as np


def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low reward values for each envs
    """
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        # This assertion is from the JAX implementation, but in PyTorch we might not always have reward_neg
        # for dense rewards, so we can remove it or make it conditional.
        # For now, keeping it as a comment to reflect the original JAX logic.
        # assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."
        pass

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    Update every transition in the trajectory and add mc_returns
    return the updated trajectory
    """
    rewards = [t["reward"] for t in trajectory]
    terminals = [t["done"] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        # Ensure mc_returns is a float, not a numpy array
        transition["mc_returns"] = float(mc_returns[i])

    return trajectory


def add_next_embeddings_to_trajectory(trajectory):
    """
    Update every transition in the trajectory and add next_embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            # For the last transition, next_embeddings is the same as current embeddings
            trajectory[i]["next_action_embeddings"] = trajectory[i]["action_embeddings"]
        else:
            trajectory[i]["next_action_embeddings"] = trajectory[i + 1]["action_embeddings"]

    return trajectory
