"""
Some of this tests are based on
https://github.com/DLR-RM/stable-baselines3/blob/e206fc55cf7768b8adde3ed3b87c27faa4edf6fe/tests/test_n_step_replay.py
"""

from collections.abc import Callable

import pytest
import torch

from lerobot.rl.acfqlvla.buffer import ReplayBufferNSteps as ReplayBuffer


def state_dims() -> list[str]:
    return ["observation.image", "observation.state"]


def clone_state(state: dict) -> dict:
    return {k: v.clone() for k, v in state.items()}


def create_empty_replay_buffer(
    optimize_memory: bool = False,
    use_drq: bool = False,
    image_augmentation_function: Callable | None = None,
) -> ReplayBuffer:
    buffer_capacity = 10
    device = "cpu"
    return ReplayBuffer(
        buffer_capacity,
        device,
        state_dims(),
        optimize_memory=optimize_memory,
        use_drq=use_drq,
        image_augmentation_function=image_augmentation_function,
        # force_full_n_steps=False,
        # use_terminal_for_next_state=False,
    )


def fill_buffer(
    buffer: ReplayBuffer, length: int, done_at: int | None = None, truncated_at: int | None = None
):
    """
    Fill the buffer with:
    - reward = 1.0
    - observation = index
    - optional `done` at index `done_at`
    - optional truncation at index `truncated_at`
    """
    for i in range(length):
        # obs = torch.full((1, 4), float(i), dtype=torch.float32)
        # next_obs = torch.full((1, 4), float(i + 1), dtype=torch.float32)
        # action = torch.zeros((1, 2), dtype=torch.float32)
        # reward = torch.tensor([1.0], dtype=torch.float32)
        # done = torch.tensor([1.0 if i == done_at else 0.0], dtype=torch.float32)
        # truncated = i == truncated_at
        # infos = [{"TimeLimit.truncated": truncated}]
        state = {
            "observation.image": torch.full((3, 84, 84), float(i), dtype=torch.float32),
            "observation.state": torch.full((1, 4), float(i), dtype=torch.float32),
        }
        next_state = {
            "observation.image": torch.full((3, 84, 84), float(i + 1), dtype=torch.float32),
            "observation.state": torch.full((1, 4), float(i + 1), dtype=torch.float32),
        }
        action = torch.zeros((1, 2), dtype=torch.float32)
        reward = torch.tensor([1.0], dtype=torch.float32)
        done = torch.tensor([1.0 if i == done_at else 0.0], dtype=torch.bool)
        truncated = torch.tensor([1.0 if i == truncated_at else 0.0], dtype=torch.bool)
        complementary_info = {}
        # action
        buffer.add(
            state=clone_state(state),
            next_state=clone_state(next_state),
            action=action,
            reward=reward,
            done=done,
            truncated=truncated,
            complementary_info=complementary_info,
        )


def compute_expected_nstep_reward(gamma, n_steps, stop_idx=None):
    """
    Compute the expected n-step reward for the test env (reward=1 for each step),
    optionally stopping early due to termination/truncation.
    Uses torch for computation.
    """
    rewards = torch.ones(n_steps, dtype=torch.float32)
    returns = torch.zeros(n_steps, dtype=torch.float32)
    last_sum = torch.tensor(0.0, dtype=torch.float32)
    for step in reversed(range(n_steps)):
        next_non_terminal = step != stop_idx
        last_sum = rewards[step] + gamma * next_non_terminal * last_sum
        returns[step] = last_sum
    return returns[0]


@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("done_at", [1, 2])
@pytest.mark.parametrize("n_steps", [2, 3, 5])
@pytest.mark.parametrize("base_idx", [0, 1, 2, 3])
def test_sample_nstep_full(optimize_memory, done_at, n_steps, base_idx):
    gamma = 0.98
    replay_buffer = create_empty_replay_buffer(optimize_memory=optimize_memory)
    fill_buffer(replay_buffer, length=10, done_at=done_at)

    batch = replay_buffer.sample_nstep_full(batch_size=1, gamma=gamma, n_steps=n_steps)

    assert batch["state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["state"]["observation.state"].shape == (1, 4)

    assert batch["next_state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["next_state"]["observation.state"].shape == (1, 4)

    assert batch["action"].shape == (1, n_steps, 2)
    assert batch["reward"].shape == (1, n_steps)
    assert batch["masks"].shape == (1, n_steps)
    assert batch["terminals"].shape == (1, n_steps)
    assert batch["valid"].shape == (1, n_steps)


@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("done_at", [1, 2])
@pytest.mark.parametrize("n_steps", [2, 3, 5])
@pytest.mark.parametrize("base_idx", [0, 1, 2, 3])
def test_nstep_early_termination(optimize_memory, done_at, n_steps, base_idx):
    gamma = 0.98
    replay_buffer = create_empty_replay_buffer(optimize_memory=optimize_memory)
    fill_buffer(replay_buffer, length=10, done_at=done_at)

    batch = replay_buffer.sample_nstep_full_for_indices(
        torch.tensor([base_idx]), batch_size=1, gamma=gamma, n_steps=n_steps
    )
    expected = compute_expected_nstep_reward(gamma=gamma, n_steps=n_steps, stop_idx=done_at - base_idx)

    terminals = batch["terminals"]
    terminal_idx = terminals.nonzero(as_tuple=True)[1].min().item() if terminals.any() else -1
    actual = batch["reward"][-1][terminal_idx].item()

    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)

    assert batch["terminals"][-1][terminal_idx].item() == float(
        base_idx <= done_at and done_at < base_idx + n_steps
    )
    # assert batch["truncated_nsteps"].item() == 0.0

    expected_valid = torch.ones((1, n_steps), dtype=torch.bool)
    if done_at >= base_idx:
        start_idx = min(done_at + 1 - base_idx, base_idx + n_steps)
        expected_valid[0, start_idx:] = False
    assert torch.equal(batch["valid"], expected_valid)

    # import pdb; pdb.set_trace()
    assert batch["state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["state"]["observation.image"].max() == base_idx
    assert batch["state"]["observation.state"].shape == (1, 4)
    assert batch["state"]["observation.state"].max() == base_idx

    assert batch["next_state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["next_state"]["observation.image"].max() == base_idx + n_steps
    assert batch["next_state"]["observation.state"].shape == (1, 4)
    assert batch["next_state"]["observation.state"].max() == base_idx + n_steps


@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("truncated_at", [1, 2])
@pytest.mark.parametrize("n_steps", [2, 3, 5])
@pytest.mark.parametrize("base_idx", [0, 1, 2, 3])
def test_nstep_early_truncation(optimize_memory, truncated_at, n_steps, base_idx):
    replay_buffer = create_empty_replay_buffer(optimize_memory=optimize_memory)
    fill_buffer(replay_buffer, length=10, truncated_at=truncated_at)

    batch = replay_buffer.sample_nstep_full_for_indices(
        torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps
    )

    terminals = batch["terminals"]
    terminal_idx = terminals.nonzero(as_tuple=True)[1].min().item() if terminals.any() else -1

    actual = batch["reward"][-1][terminal_idx].item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=truncated_at - base_idx)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    # assert batch["done_nsteps"].item() == 0.0
    assert batch["terminals"][-1][terminal_idx].item() == float(
        base_idx <= truncated_at and truncated_at < base_idx + n_steps
    )

    expected_valid = torch.ones((1, n_steps), dtype=torch.bool)
    if truncated_at >= base_idx:
        start_idx = min(truncated_at + 1 - base_idx, base_idx + n_steps)
        expected_valid[0, start_idx:] = False
    assert torch.equal(batch["valid"], expected_valid)

    assert batch["state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["state"]["observation.image"].max() == base_idx
    assert batch["state"]["observation.state"].shape == (1, 4)
    assert batch["state"]["observation.state"].max() == base_idx

    assert batch["next_state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["next_state"]["observation.image"].max() == base_idx + n_steps
    assert batch["next_state"]["observation.state"].shape == (1, 4)
    assert batch["next_state"]["observation.state"].max() == base_idx + n_steps


@pytest.mark.parametrize("use_drq", [False, True])
@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("done_at", [None, 1, 2])
@pytest.mark.parametrize("n_steps", [1, 3, 5])
@pytest.mark.parametrize("base_idx", [0, 1, 3])
def test_use_drq_early_termination(
    use_drq: bool, optimize_memory: bool, done_at: int | None, n_steps: int, base_idx: int
):
    """
    Test that the replay buffer can handle DRQ-style data augmentation.
    """
    # Create a buffer with DRQ enabled
    buffer = create_empty_replay_buffer(use_drq=use_drq, optimize_memory=optimize_memory)

    # Fill the buffer with some data
    fill_buffer(buffer, length=10, done_at=done_at)

    # Sample a batch and check if the data augmentation is applied
    batch = buffer.sample_nstep_full_for_indices(
        torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps
    )

    # Check if the augmented data is present
    # assert "observation.image_augmented" in batch
    assert batch["state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["state"]["observation.image"].max() == base_idx
    assert batch["state"]["observation.state"].shape == (1, 4)
    assert batch["state"]["observation.state"].max() == base_idx

    assert batch["next_state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["next_state"]["observation.image"].max() == base_idx + n_steps
    assert batch["next_state"]["observation.state"].shape == (1, 4)
    assert batch["next_state"]["observation.state"].max() == base_idx + n_steps


@pytest.mark.parametrize("use_drq", [False, True])
@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("truncated_at", [None, 1, 2])
@pytest.mark.parametrize("n_steps", [1, 3, 5])
@pytest.mark.parametrize("base_idx", [0, 1, 3])
def test_use_drq_early_truncation(
    use_drq: bool, optimize_memory: bool, truncated_at: int | None, n_steps: int, base_idx: int
):
    """
    Test that the replay buffer can handle DRQ-style data augmentation.
    """
    # Create a buffer with DRQ enabled
    buffer = create_empty_replay_buffer(use_drq=use_drq, optimize_memory=optimize_memory)

    # Fill the buffer with some data
    fill_buffer(buffer, length=10, truncated_at=truncated_at)

    # Sample a batch and check if the data augmentation is applied
    batch = buffer.sample_nstep_full_for_indices(
        torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps
    )

    # Check if the augmented data is present
    # assert "observation.image_augmented" in batch
    assert batch["state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["state"]["observation.image"].max() == base_idx
    assert batch["state"]["observation.state"].shape == (1, 4)
    assert batch["state"]["observation.state"].max() == base_idx

    assert batch["next_state"]["observation.image"].shape == (1, 3, 84, 84)
    assert batch["next_state"]["observation.image"].max() == base_idx + n_steps
    assert batch["next_state"]["observation.state"].shape == (1, 4)
    assert batch["next_state"]["observation.state"].max() == base_idx + n_steps


def test_sparse_reward_computation():
    buffer = create_empty_replay_buffer(use_drq=False, optimize_memory=False)
    fill_buffer(buffer, length=10, done_at=3)

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.0000, 0.0000, 0.9703]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 1, 1, 0]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 1, 1]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 0, 0, 0]], dtype=torch.float32))

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 1, 1, 1]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 1, 1]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32))

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.9900, 1.9701, 1.970]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 0, 0, 0]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 1, 1, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 0, 0, 0]], dtype=torch.float32))

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.9900, 0.9900, 0.9900]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 0, 0, 0]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 1, 1, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 0, 1, 1]], dtype=torch.float32))

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.9900, 0.9900, 1.9603]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 0, 0, 0]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 1, 1, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 0, 1, 1]], dtype=torch.float32))

    buffer.rewards.copy_(torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32))
    buffer.dones.copy_(torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.bool))
    buffer.truncateds.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))

    batch = buffer.sample_nstep_full_for_indices(torch.tensor([2]), batch_size=1, gamma=0.99, n_steps=4)
    assert torch.allclose(
        batch["reward"], torch.tensor([[0.0000, 0.0000, 0.9801, 1.9504]]), rtol=1e-4, atol=1e-4
    )
    assert batch["masks"].equal(torch.tensor([[1, 1, 0, 0]], dtype=torch.float32))
    assert batch["valid"].equal(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert batch["terminals"].equal(torch.tensor([[0, 1, 1, 1]], dtype=torch.float32))
    assert batch["truncateds"].equal(torch.tensor([[0, 1, 1, 1]], dtype=torch.float32))
