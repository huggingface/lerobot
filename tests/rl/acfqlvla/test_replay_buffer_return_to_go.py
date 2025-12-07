import torch

from lerobot.rl.acfqlvla.buffer import add_mc_returns_to_trajectory


def test_add_mc_returns_to_trajectory_basic():
    """Test basic functionality of add_mc_returns_to_trajectory."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 4.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Calculate expected returns manually
    # G_3 = 4.0 (terminal)
    # G_2 = 3.0 + 0.9 * 4.0 = 6.6
    # G_1 = 2.0 + 0.9 * 6.6 = 7.94
    # G_0 = 1.0 + 0.9 * 7.94 = 8.146
    expected_returns = [8.146, 7.94, 6.6, 4.0]

    for i, transition in enumerate(result):
        assert "complementary_info" in transition
        assert "mc_returns" in transition["complementary_info"]
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_single_step():
    """Test with single step trajectory."""
    transitions = [
        {"reward": 5.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    torch.testing.assert_close(
        result[0]["complementary_info"]["mc_returns"], torch.tensor(5.0), rtol=1e-5, atol=1e-3
    )


def test_add_mc_returns_to_trajectory_no_terminal():
    """Test with trajectory that doesn't end in terminal state."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Without terminal state, returns accumulate from the end
    # G_2 = 3.0
    # G_1 = 2.0 + 0.9 * 3.0 = 4.7
    # G_0 = 1.0 + 0.9 * 4.7 = 5.23
    expected_returns = [5.23, 4.7, 3.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_zero_gamma():
    """Test with gamma = 0 (no discounting)."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 4.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.0

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # With gamma=0, each return equals immediate reward
    expected_returns = [1.0, 2.0, 3.0, 4.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_preserves_original():
    """Test that original trajectory data is preserved."""
    transitions = [
        {
            "reward": 1.0,
            "done": False,
            "truncated": False,
            "observation": 0.5,
            "action": 1,
            "complementary_info": {},
        },
        {
            "reward": 2.0,
            "done": True,
            "truncated": False,
            "observation": 0.7,
            "action": 0,
            "complementary_info": {},
        },
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Original keys should be preserved
    for i, transition in enumerate(result):
        assert "reward" in transition
        assert "done" in transition
        assert "observation" in transition
        assert "action" in transition
        assert transition["reward"] == transitions[i]["reward"]
        assert transition["done"] == transitions[i]["done"]


def test_add_mc_returns_to_trajectory_empty_trajectory():
    """Test with empty trajectory."""
    transitions = []
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    assert len(result) == 0


def test_add_mc_returns_to_trajectory_negative_rewards():
    """Test with negative rewards."""
    transitions = [
        {"reward": -1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": -2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # G_2 = 3.0
    # G_1 = -2.0 + 0.9 * 3.0 = 0.7
    # G_0 = -1.0 + 0.9 * 0.7 = -0.37
    expected_returns = [-0.37, 0.7, 3.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_truncated_end():
    """Test with trajectory ending in truncation."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": True, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # With truncation, return calculation stops (terminal = done OR truncated)
    # G_2 = 3.0 (truncated, so no future return)
    # G_1 = 2.0 + 0.9 * 3.0 = 4.7
    # G_0 = 1.0 + 0.9 * 4.7 = 5.23
    expected_returns = [5.23, 4.7, 3.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_truncated_middle():
    """Test with truncation in the middle of trajectory."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": True, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 4.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Episode 1 (truncated at step 1): [1.0, 2.0]
    # Episode 2 (done at step 3): [3.0, 4.0]
    # G_1 = 2.0 (truncated)
    # G_0 = 1.0 + 0.9 * 2.0 = 2.8
    # G_3 = 4.0 (done)
    # G_2 = 3.0 + 0.9 * 4.0 = 6.6
    expected_returns = [2.8, 2.0, 6.6, 4.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_done_middle():
    """Test with done flag in the middle of trajectory."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": True, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 4.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Episode 1 (done at step 1): [1.0, 2.0]
    # Episode 2 (done at step 3): [3.0, 4.0]
    # G_1 = 2.0 (done)
    # G_0 = 1.0 + 0.9 * 2.0 = 2.8
    # G_3 = 4.0 (done)
    # G_2 = 3.0 + 0.9 * 4.0 = 6.6
    expected_returns = [2.8, 2.0, 6.6, 4.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_multiple_truncations():
    """Test with multiple truncations."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": True, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": True, "complementary_info": {}},
        {"reward": 3.0, "done": False, "truncated": True, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Each truncation ends the return calculation
    # G_0 = 1.0, G_1 = 2.0, G_2 = 3.0
    expected_returns = [1.0, 2.0, 3.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_done_and_truncated():
    """Test with both done and truncated flags set (edge case)."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": True, "truncated": True, "complementary_info": {}},
    ]
    gamma = 0.9

    result = add_mc_returns_to_trajectory(
        transitions, gamma, reward_scale=1.0, reward_bias=0.0, reward_neg=0.0, is_sparse_reward=False
    )

    # Both done and truncated are terminal conditions
    # G_1 = 2.0
    # G_0 = 1.0 + 0.9 * 2.0 = 2.8
    expected_returns = [2.8, 2.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_reward_all_negative():
    """Test sparse reward case with all negative rewards."""
    transitions = [
        {"reward": -1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": -1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": -1.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = -1.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # For sparse reward with all negative rewards:
    # return_to_go = reward_neg / (1 - gamma) = -1.0 / (1 - 0.99) = -100.0
    expected_return = -100.0
    expected_returns = [expected_return, expected_return, expected_return]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_reward_with_success():
    """Test sparse reward case with success (positive reward at end)."""
    transitions = [
        {"reward": -1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": -1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 10.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = -1.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Standard MC return calculation (not all negative)
    # G_2 = 10.0
    # G_1 = -1.0 + 0.99 * 10.0 = 8.9
    # G_0 = -1.0 + 0.99 * 8.9 = 7.811
    expected_returns = [7.811, 8.9, 10.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_reward_scale_and_bias():
    """Test with reward scaling and bias."""
    transitions = [
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 2.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 3.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.9
    reward_scale = 2.0
    reward_bias = 1.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=0.0,
        is_sparse_reward=False,
    )

    # Rewards are used as-is in the calculation (scaling/bias applied elsewhere if needed)
    # G_2 = 3.0
    # G_1 = 2.0 + 0.9 * 3.0 = 4.7
    # G_0 = 1.0 + 0.9 * 4.7 = 5.23
    expected_returns = [5.23, 4.7, 3.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_success():
    """Test sparse binary reward with success (1.0 on last transition)."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 1.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Not all zeros, so standard MC calculation:
    # G_3 = 1.0
    # G_2 = 0.0 + 0.99 * 1.0 = 0.99
    # G_1 = 0.0 + 0.99 * 0.99 = 0.9801
    # G_0 = 0.0 + 0.99 * 0.9801 = 0.970299
    expected_returns = [0.970299, 0.9801, 0.99, 1.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_failure():
    """Test sparse binary reward with failure (0.0 on last transition)."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # All rewards are 0.0, which equals reward_neg (0.0), so special sparse handling:
    # return_to_go = 0.0 / (1 - 0.99) = 0.0
    expected_returns = [0.0, 0.0, 0.0, 0.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_truncated_success():
    """Test sparse binary reward with truncation and success."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 1.0, "done": False, "truncated": True, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Not all zeros (has 1.0), so standard MC calculation:
    # G_2 = 1.0 (truncated)
    # G_1 = 0.0 + 0.99 * 1.0 = 0.99
    # G_0 = 0.0 + 0.99 * 0.99 = 0.9801
    expected_returns = [0.9801, 0.99, 1.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_long_success():
    """Test sparse binary reward with longer trajectory and success."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 1.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Standard MC calculation (not all zeros):
    # Working backwards from the end:
    # G_6 = 1.0
    # G_5 = 0.0 + 0.99 * 1.0 = 0.99
    # G_4 = 0.0 + 0.99 * 0.99 = 0.9801
    # G_3 = 0.0 + 0.99 * 0.9801 = 0.970299
    # G_2 = 0.0 + 0.99 * 0.970299 = 0.96059601
    # G_1 = 0.0 + 0.99 * 0.96059601 = 0.9509900499
    # G_0 = 0.0 + 0.99 * 0.9509900499 = 0.941480149401
    expected_returns = [0.941480149401, 0.9509900499, 0.96059601, 0.970299, 0.9801, 0.99, 1.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_multiple_episodes_mixed():
    """Test sparse binary reward with multiple episodes - some failures, some successes."""
    # This simulates a trajectory batch containing multiple episodes
    transitions = [
        # Episode 1: failure (all 0.0)
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": True, "truncated": False, "complementary_info": {}},
        # Episode 2: success (1.0 at end)
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 1.0, "done": True, "truncated": False, "complementary_info": {}},
        # Episode 3: failure (all 0.0)
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Episode 1 (indices 0-2): all zeros, special sparse handling
    # return_to_go = 0.0 / (1 - 0.99) = 0.0
    # Episode 2 (indices 3-5): has success, standard MC
    # G_5 = 1.0
    # G_4 = 0.0 + 0.99 * 1.0 = 0.99
    # G_3 = 0.0 + 0.99 * 0.99 = 0.9801
    # Episode 3 (indices 6-7): all zeros, special sparse handling
    # return_to_go = 0.0 / (1 - 0.99) = 0.0
    expected_returns = [0.0, 0.0, 0.0, 0.9801, 0.99, 1.0, 0.0, 0.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_success_in_middle():
    """Test sparse binary reward with success appearing in middle of longer sequence."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 1.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": True, "truncated": False, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # Not all zeros (has 1.0 in middle), so standard MC calculation:
    # G_3 = 0.0
    # G_2 = 0.0 + 0.99 * 0.0 = 0.0
    # G_1 = 1.0 + 0.99 * 0.0 = 1.0
    # G_0 = 0.0 + 0.99 * 1.0 = 0.99
    expected_returns = [0.99, 1.0, 0.0, 0.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )


def test_add_mc_returns_to_trajectory_sparse_binary_truncated_failure():
    """Test sparse binary reward with truncation and failure (all zeros)."""
    transitions = [
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": False, "complementary_info": {}},
        {"reward": 0.0, "done": False, "truncated": True, "complementary_info": {}},
    ]
    gamma = 0.99
    reward_scale = 1.0
    reward_bias = 0.0
    reward_neg = 0.0

    result = add_mc_returns_to_trajectory(
        transitions,
        gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=True,
    )

    # All zeros with truncation, special sparse handling:
    # return_to_go = 0.0 / (1 - 0.99) = 0.0
    expected_returns = [0.0, 0.0, 0.0]

    for i, transition in enumerate(result):
        torch.testing.assert_close(
            transition["complementary_info"]["mc_returns"],
            torch.tensor(expected_returns[i]),
            rtol=1e-5,
            atol=1e-3,
        )
