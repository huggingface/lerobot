import pytest
import torch

from lerobot.scripts.server.buffer import BatchTransition, ReplayBuffer
from tests.fixtures.constants import DUMMY_REPO_ID


def state_dims() -> list[str]:
    return ["observation.image", "observation.state"]


@pytest.fixture
def replay_buffer() -> ReplayBuffer:
    return create_empty_replay_buffer()


def create_empty_replay_buffer(optimize_memory=False) -> ReplayBuffer:
    buffer_capacity = 10
    device = "cpu"
    return ReplayBuffer(buffer_capacity, device, state_dims(), optimize_memory=optimize_memory)


def create_dummy_transition() -> dict:
    return {
        "observation.image": torch.rand(3, 84, 84),
        "action": torch.randn(4),
        "reward": torch.tensor(1.0),
        "observation.state": torch.randn(
            10,
        ),
        "done": torch.tensor(False),
        "truncated": torch.tensor(False),
        "complementary_info": {},
    }


def create_dataset_from_replay_buffer(tmp_path):
    dummy_state_1 = create_dummy_state()
    dummy_action_1 = create_dummy_action()

    dummy_state_2 = create_dummy_state()
    dummy_action_2 = create_dummy_action()

    dummy_state_3 = create_dummy_state()
    dummy_action_3 = create_dummy_action()

    dummy_state_4 = create_dummy_state()
    dummy_action_4 = create_dummy_action()

    replay_buffer = create_empty_replay_buffer()
    replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_1, False, False)
    replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_3, True, True)
    replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, dummy_state_4, True, True)

    root = tmp_path / "test"
    return replay_buffer.to_lerobot_dataset(DUMMY_REPO_ID, root=root)


def create_dummy_state() -> dict:
    return {
        "observation.image": torch.rand(3, 84, 84),
        "observation.state": torch.randn(
            10,
        ),
    }


def create_dummy_action() -> torch.Tensor:
    return torch.randn(4)


def dict_properties() -> list:
    return ["state", "next_state"]


@pytest.fixture
def dummy_state() -> dict:
    return create_dummy_state()


@pytest.fixture
def next_dummy_state() -> dict:
    return create_dummy_state()


@pytest.fixture
def dummy_action() -> torch.Tensor:
    return torch.randn(4)


def test_empty_buffer_sample_raises_error(replay_buffer):
    assert len(replay_buffer) == 0, "Replay buffer should be empty."
    assert replay_buffer.capacity == 10, "Replay buffer capacity should be 10."
    with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
        replay_buffer.sample(1)


def test_zero_capacity_buffer_raises_error():
    with pytest.raises(ValueError, match="Capacity must be greater than 0."):
        ReplayBuffer(0, "cpu", ["observation", "next_observation"])


def test_add_transition(replay_buffer, dummy_state, dummy_action):
    replay_buffer.add(dummy_state, dummy_action, 1.0, dummy_state, False, False)
    assert len(replay_buffer) == 1, "Replay buffer should have one transition after adding."
    assert torch.equal(replay_buffer.actions[0], dummy_action), (
        "Action should be equal to the first transition."
    )
    assert replay_buffer.rewards[0] == 1.0, "Reward should be equal to the first transition."
    assert replay_buffer.dones[0] == False, "Done should be False for the first transition."
    assert replay_buffer.truncateds[0] == False, "Truncated should be False for the first transition."

    for dim in state_dims():
        assert torch.equal(replay_buffer.states[dim][0], dummy_state[dim]), (
            "Observation should be equal to the first transition."
        )
        assert torch.equal(replay_buffer.next_states[dim][0], dummy_state[dim]), (
            "Next observation should be equal to the first transition."
        )


def test_add_over_capacity():
    replay_buffer = ReplayBuffer(2, "cpu", ["observation", "next_observation"])
    dummy_state_1 = create_dummy_state()
    dummy_action_1 = create_dummy_action()

    dummy_state_2 = create_dummy_state()
    dummy_action_2 = create_dummy_action()

    dummy_state_3 = create_dummy_state()
    dummy_action_3 = create_dummy_action()

    replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_1, False, False)
    replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_3, True, True)

    assert len(replay_buffer) == 2, "Replay buffer should have 2 transitions after adding 3."

    for dim in state_dims():
        assert torch.equal(replay_buffer.states[dim][0], dummy_state_3[dim]), (
            "Observation should be equal to the first transition."
        )
        assert torch.equal(replay_buffer.next_states[dim][0], dummy_state_3[dim]), (
            "Next observation should be equal to the first transition."
        )

    assert torch.equal(replay_buffer.actions[0], dummy_action_3), (
        "Action should be equal to the last transition."
    )
    assert replay_buffer.rewards[0] == 1.0, "Reward should be equal to the last transition."
    assert replay_buffer.dones[0] == True, "Done should be True for the first transition."
    assert replay_buffer.truncateds[0] == True, "Truncated should be True for the first transition."


def test_sample_from_empty_buffer(replay_buffer):
    with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
        replay_buffer.sample(1)


def test_sample_with_1_transition(replay_buffer, dummy_state, next_dummy_state, dummy_action):
    replay_buffer.add(dummy_state, dummy_action, 1.0, next_dummy_state, False, False)
    got_batch_transition = replay_buffer.sample(1)

    expected_batch_transition = BatchTransition(
        state=dummy_state.clone(),
        action=dummy_action.clone(),
        reward=1.0,
        next_state=next_dummy_state.clone(),
        done=False,
        truncated=False,
    )

    for buffer_property in dict_properties():
        for k, v in expected_batch_transition[buffer_property].items():
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 1, f"{k} should have 1 transition."
            assert got_state.device.type == "cpu", f"{k} should be on cpu."
            assert torch.equal(got_state[0], v), f"{k} should be equal to the expected batch transition."

    for k, v in expected_batch_transition.items():
        if k in dict_properties():
            continue

        got_value = got_batch_transition[k]

        v_tensor = v
        if not isinstance(v, torch.Tensor):
            v_tensor = torch.tensor(v)

        assert got_value.shape[0] == 1, f"{k} should have 1 transition."
        assert got_value.device.type == "cpu", f"{k} should be on cpu."
        assert torch.equal(got_value[0], v_tensor), f"{k} should be equal to the expected batch transition."


def test_sample_with_batch_bigger_than_buffer_size(
    replay_buffer, dummy_state, next_dummy_state, dummy_action
):
    replay_buffer.add(dummy_state, dummy_action, 1.0, next_dummy_state, False, False)
    got_batch_transition = replay_buffer.sample(10)

    expected_batch_transition = BatchTransition(
        state=dummy_state,
        action=dummy_action,
        reward=1.0,
        next_state=next_dummy_state,
        done=False,
        truncated=False,
    )

    for buffer_property in dict_properties():
        for k, v in expected_batch_transition[buffer_property].items():
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 1, f"{k} should have 1 transition."

    for k, v in expected_batch_transition.items():
        if k in dict_properties():
            continue

        got_value = got_batch_transition[k]
        assert got_value.shape[0] == 1, f"{k} should have 1 transition."


def test_sample_batch(replay_buffer):
    dummy_state_1 = create_dummy_state()
    dummy_action_1 = create_dummy_action()

    dummy_state_2 = create_dummy_state()
    dummy_action_2 = create_dummy_action()

    dummy_state_3 = create_dummy_state()
    dummy_action_3 = create_dummy_action()

    dummy_state_4 = create_dummy_state()
    dummy_action_4 = create_dummy_action()

    replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_1, False, False)
    replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_3, True, True)
    replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, dummy_state_4, True, True)

    got_batch_transition = replay_buffer.sample(3)

    for buffer_property in dict_properties():
        for k in got_batch_transition[buffer_property]:
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 3, f"{k} should have 3 transition."

    for k in got_batch_transition:
        if k in dict_properties():
            continue

        got_value = got_batch_transition[k]
        assert got_value.shape[0] == 3, f"{k} should have 3 transition."


def test_to_lerobot_dataset_with_empty_buffer(replay_buffer):
    with pytest.raises(ValueError, match="The replay buffer is empty. Cannot convert to a dataset."):
        replay_buffer.to_lerobot_dataset("dummy_repo")


def test_to_lerobot_dataset(tmp_path):
    ds = create_dataset_from_replay_buffer(tmp_path)

    assert ds.fps == 1, "FPS should be 1"

    for dim in state_dims():
        assert dim in ds.features
        assert ds.features[dim]["shape"] == create_dummy_state()[dim].shape

    assert ds.num_episodes == 2
    assert ds.num_frames == 4


def test_from_lerobot_dataset(tmp_path):
    dummy_state_1 = create_dummy_state()
    dummy_action_1 = create_dummy_action()

    dummy_state_2 = create_dummy_state()
    dummy_action_2 = create_dummy_action()

    dummy_state_3 = create_dummy_state()
    dummy_action_3 = create_dummy_action()

    dummy_state_4 = create_dummy_state()
    dummy_action_4 = create_dummy_action()

    replay_buffer = create_empty_replay_buffer()
    replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_1, False, False)
    replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_3, True, True)
    replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, dummy_state_4, True, True)

    root = tmp_path / "test"
    ds = replay_buffer.to_lerobot_dataset(DUMMY_REPO_ID, root=root)

    reconverted_buffer = ReplayBuffer.from_lerobot_dataset(ds, state_keys=list(state_dims()), device="cpu")

    assert len(reconverted_buffer) == 4, "Reconverted Replay buffer should have the same size as original"

    assert torch.equal(reconverted_buffer.actions, replay_buffer.actions), (
        "Actions from converted buffer should be equal to the original replay buffer."
    )
    assert torch.equal(reconverted_buffer.rewards, replay_buffer.rewards), (
        "Rewards from converted buffer should be equal to the original replay buffer."
    )
    assert torch.equal(reconverted_buffer.dones, replay_buffer.dones), (
        "Dones from converted buffer should be equal to the original replay buffer."
    )
    assert torch.equal(reconverted_buffer.truncateds, replay_buffer.truncateds), (
        "Truncateds from converted buffer should be equal to the original replay buffer."
    )

    assert torch.equal(reconverted_buffer.states, replay_buffer.states), (
        "Observations from converted buffer should be equal to the original replay buffer."
    )
    assert torch.equal(reconverted_buffer.next_states, replay_buffer.next_states), (
        "Next observations from converted buffer should be equal to the original replay buffer."
    )


def test_buffer_sample_alignment():
    # Initialize buffer
    buffer = ReplayBuffer(capacity=100, device="cpu", state_keys=["state_value"], storage_device="cpu")

    # Fill buffer with patterned data
    for i in range(100):
        signature = float(i) / 100.0
        state = {"state_value": torch.tensor([[signature]]).float()}
        action = torch.tensor([[2.0 * signature]]).float()
        reward = 3.0 * signature

        is_end = (i + 1) % 10 == 0
        if is_end:
            next_state = {"state_value": torch.tensor([[signature]]).float()}
            done = True
        else:
            next_signature = float(i + 1) / 100.0
            next_state = {"state_value": torch.tensor([[next_signature]]).float()}
            done = False

        buffer.add(state, action, reward, next_state, done, False)

    # Sample and verify
    batch = buffer.sample(50)

    for i in range(50):
        state_sig = batch["state"]["state_value"][i].item()
        action_val = batch["action"][i].item()
        reward_val = batch["reward"][i].item()
        next_state_sig = batch["next_state"]["state_value"][i].item()
        is_done = batch["done"][i].item() > 0.5

        # Verify relationships
        assert abs(action_val - 2.0 * state_sig) < 1e-4, (
            f"Action {action_val} should be 2x state signature {state_sig}"
        )

        assert abs(reward_val - 3.0 * state_sig) < 1e-4, (
            f"Reward {reward_val} should be 3x state signature {state_sig}"
        )

        if is_done:
            assert abs(next_state_sig - state_sig) < 1e-4, (
                f"For done states, next_state {next_state_sig} should equal state {state_sig}"
            )
        else:
            # Either it's the next sequential state (+0.01) or same state (for episode boundaries)
            valid_next = (
                abs(next_state_sig - state_sig - 0.01) < 1e-4 or abs(next_state_sig - state_sig) < 1e-4
            )
            assert valid_next, (
                f"Next state {next_state_sig} should be either state+0.01 or same as state {state_sig}"
            )


def test_memory_optimization():
    dummy_state_1 = create_dummy_state()
    dummy_action_1 = create_dummy_action()

    dummy_state_2 = create_dummy_state()
    dummy_action_2 = create_dummy_action()

    dummy_state_3 = create_dummy_state()
    dummy_action_3 = create_dummy_action()

    dummy_state_4 = create_dummy_state()
    dummy_action_4 = create_dummy_action()

    replay_buffer = create_empty_replay_buffer()
    replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_3, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_4, False, False)
    replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, None, True, True)

    optimized_replay_buffer = create_empty_replay_buffer(True)
    optimized_replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_2, False, False)
    optimized_replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_3, False, False)
    optimized_replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_4, False, False)
    optimized_replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, None, True, True)

    sampled_transitions = replay_buffer.sample(4)
    sampled_transitions_opt = optimized_replay_buffer.sample(4)

    #  Check sampled vresions
    # Срусл ьуьщкн гыфпу


def test_check_image_augmentations_without_drq():
    pass
