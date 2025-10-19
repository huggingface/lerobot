#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections.abc import Callable

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import BatchTransition, ReplayBuffer, random_crop_vectorized
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_STATE, OBS_STR, REWARD
from tests.fixtures.constants import DUMMY_REPO_ID


def state_dims() -> list[str]:
    return [OBS_IMAGE, OBS_STATE]


@pytest.fixture
def replay_buffer() -> ReplayBuffer:
    return create_empty_replay_buffer()


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
    )


def create_random_image() -> torch.Tensor:
    return torch.rand(3, 84, 84)


def create_dummy_transition() -> dict:
    return {
        OBS_IMAGE: create_random_image(),
        ACTION: torch.randn(4),
        "reward": torch.tensor(1.0),
        OBS_STATE: torch.randn(
            10,
        ),
        "done": torch.tensor(False),
        "truncated": torch.tensor(False),
        "complementary_info": {},
    }


def create_dataset_from_replay_buffer(tmp_path) -> tuple[LeRobotDataset, ReplayBuffer]:
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
    return (replay_buffer.to_lerobot_dataset(DUMMY_REPO_ID, root=root), replay_buffer)


def create_dummy_state() -> dict:
    return {
        OBS_IMAGE: create_random_image(),
        OBS_STATE: torch.randn(
            10,
        ),
    }


def get_tensor_memory_consumption(tensor):
    return tensor.nelement() * tensor.element_size()


def get_tensors_memory_consumption(obj, visited_addresses):
    total_size = 0

    address = id(obj)
    if address in visited_addresses:
        return 0

    visited_addresses.add(address)

    if isinstance(obj, torch.Tensor):
        return get_tensor_memory_consumption(obj)
    elif isinstance(obj, (list | tuple)):
        for item in obj:
            total_size += get_tensors_memory_consumption(item, visited_addresses)
    elif isinstance(obj, dict):
        for value in obj.values():
            total_size += get_tensors_memory_consumption(value, visited_addresses)
    elif hasattr(obj, "__dict__"):
        # It's an object, we need to get the size of the attributes
        for _, attr in vars(obj).items():
            total_size += get_tensors_memory_consumption(attr, visited_addresses)

    return total_size


def get_object_memory(obj):
    # Track visited addresses to avoid infinite loops
    # and cases when two properties point to the same object
    visited_addresses = set()

    # Get the size of the object in bytes
    total_size = sys.getsizeof(obj)

    # Get the size of the tensor attributes
    total_size += get_tensors_memory_consumption(obj, visited_addresses)

    return total_size


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
        ReplayBuffer(0, "cpu", [OBS_STR, "next_observation"])


def test_add_transition(replay_buffer, dummy_state, dummy_action):
    replay_buffer.add(dummy_state, dummy_action, 1.0, dummy_state, False, False)
    assert len(replay_buffer) == 1, "Replay buffer should have one transition after adding."
    assert torch.equal(replay_buffer.actions[0], dummy_action), (
        "Action should be equal to the first transition."
    )
    assert replay_buffer.rewards[0] == 1.0, "Reward should be equal to the first transition."
    assert not replay_buffer.dones[0], "Done should be False for the first transition."
    assert not replay_buffer.truncateds[0], "Truncated should be False for the first transition."

    for dim in state_dims():
        assert torch.equal(replay_buffer.states[dim][0], dummy_state[dim]), (
            "Observation should be equal to the first transition."
        )
        assert torch.equal(replay_buffer.next_states[dim][0], dummy_state[dim]), (
            "Next observation should be equal to the first transition."
        )


def test_add_over_capacity():
    replay_buffer = ReplayBuffer(2, "cpu", [OBS_STR, "next_observation"])
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
    assert replay_buffer.dones[0], "Done should be True for the first transition."
    assert replay_buffer.truncateds[0], "Truncated should be True for the first transition."


def test_sample_from_empty_buffer(replay_buffer):
    with pytest.raises(RuntimeError, match="Cannot sample from an empty buffer"):
        replay_buffer.sample(1)


def test_sample_with_1_transition(replay_buffer, dummy_state, next_dummy_state, dummy_action):
    replay_buffer.add(dummy_state, dummy_action, 1.0, next_dummy_state, False, False)
    got_batch_transition = replay_buffer.sample(1)

    expected_batch_transition = BatchTransition(
        state=clone_state(dummy_state),
        action=dummy_action.clone(),
        reward=1.0,
        next_state=clone_state(next_dummy_state),
        done=False,
        truncated=False,
    )

    for buffer_property in dict_properties():
        for k, v in expected_batch_transition[buffer_property].items():
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 1, f"{k} should have 1 transition."
            assert got_state.device.type == "cpu", f"{k} should be on cpu."

            assert torch.equal(got_state[0], v), f"{k} should be equal to the expected batch transition."

    for key, _value in expected_batch_transition.items():
        if key in dict_properties():
            continue

        got_value = got_batch_transition[key]

        v_tensor = expected_batch_transition[key]
        if not isinstance(v_tensor, torch.Tensor):
            v_tensor = torch.tensor(v_tensor)

        assert got_value.shape[0] == 1, f"{key} should have 1 transition."
        assert got_value.device.type == "cpu", f"{key} should be on cpu."
        assert torch.equal(got_value[0], v_tensor), f"{key} should be equal to the expected batch transition."


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
        for k in expected_batch_transition[buffer_property]:
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 1, f"{k} should have 1 transition."

    for key in expected_batch_transition:
        if key in dict_properties():
            continue

        got_value = got_batch_transition[key]
        assert got_value.shape[0] == 1, f"{key} should have 1 transition."


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
    replay_buffer.add(dummy_state_2, dummy_action_2, 2.0, dummy_state_2, False, False)
    replay_buffer.add(dummy_state_3, dummy_action_3, 3.0, dummy_state_3, True, True)
    replay_buffer.add(dummy_state_4, dummy_action_4, 4.0, dummy_state_4, True, True)

    dummy_states = [dummy_state_1, dummy_state_2, dummy_state_3, dummy_state_4]
    dummy_actions = [dummy_action_1, dummy_action_2, dummy_action_3, dummy_action_4]

    got_batch_transition = replay_buffer.sample(3)

    for buffer_property in dict_properties():
        for k in got_batch_transition[buffer_property]:
            got_state = got_batch_transition[buffer_property][k]

            assert got_state.shape[0] == 3, f"{k} should have 3 transition."

            for got_state_item in got_state:
                assert any(torch.equal(got_state_item, dummy_state[k]) for dummy_state in dummy_states), (
                    f"{k} should be equal to one of the dummy states."
                )

    for got_action_item in got_batch_transition[ACTION]:
        assert any(torch.equal(got_action_item, dummy_action) for dummy_action in dummy_actions), (
            "Actions should be equal to the dummy actions."
        )

    for k in got_batch_transition:
        if k in dict_properties() or k == "complementary_info":
            continue

        got_value = got_batch_transition[k]
        assert got_value.shape[0] == 3, f"{k} should have 3 transition."


def test_to_lerobot_dataset_with_empty_buffer(replay_buffer):
    with pytest.raises(ValueError, match="The replay buffer is empty. Cannot convert to a dataset."):
        replay_buffer.to_lerobot_dataset("dummy_repo")


def test_to_lerobot_dataset(tmp_path):
    ds, buffer = create_dataset_from_replay_buffer(tmp_path)

    assert len(ds) == len(buffer), "Dataset should have the same size as the Replay Buffer"
    assert ds.fps == 1, "FPS should be 1"
    assert ds.repo_id == "dummy/repo", "The dataset should have `dummy/repo` repo id"

    for dim in state_dims():
        assert dim in ds.features
        assert ds.features[dim]["shape"] == buffer.states[dim][0].shape

    assert ds.num_episodes == 2
    assert ds.num_frames == 4

    for j, value in enumerate(ds):
        print(torch.equal(value[OBS_IMAGE], buffer.next_states[OBS_IMAGE][j]))

    for i in range(len(ds)):
        for feature, value in ds[i].items():
            if feature == ACTION:
                assert torch.equal(value, buffer.actions[i])
            elif feature == REWARD:
                assert torch.equal(value, buffer.rewards[i])
            elif feature == DONE:
                assert torch.equal(value, buffer.dones[i])
            elif feature == OBS_IMAGE:
                # Tensor -> numpy is not precise, so we have some diff there
                # TODO: Check and fix it
                torch.testing.assert_close(value, buffer.states[OBS_IMAGE][i], rtol=0.3, atol=0.003)
            elif feature == OBS_STATE:
                assert torch.equal(value, buffer.states[OBS_STATE][i])


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

    reconverted_buffer = ReplayBuffer.from_lerobot_dataset(
        ds, state_keys=list(state_dims()), device="cpu", capacity=replay_buffer.capacity, use_drq=False
    )

    # Check only the part of the buffer that's actually filled with data
    assert torch.equal(
        reconverted_buffer.actions[: len(replay_buffer)],
        replay_buffer.actions[: len(replay_buffer)],
    ), "Actions from converted buffer should be equal to the original replay buffer."
    assert torch.equal(
        reconverted_buffer.rewards[: len(replay_buffer)], replay_buffer.rewards[: len(replay_buffer)]
    ), "Rewards from converted buffer should be equal to the original replay buffer."
    assert torch.equal(
        reconverted_buffer.dones[: len(replay_buffer)], replay_buffer.dones[: len(replay_buffer)]
    ), "Dones from converted buffer should be equal to the original replay buffer."

    # Lerobot DS haven't supported truncateds yet
    expected_truncateds = torch.zeros(len(replay_buffer)).bool()
    assert torch.equal(reconverted_buffer.truncateds[: len(replay_buffer)], expected_truncateds), (
        "Truncateds from converted buffer should be equal False"
    )

    assert torch.equal(
        replay_buffer.states[OBS_STATE][: len(replay_buffer)],
        reconverted_buffer.states[OBS_STATE][: len(replay_buffer)],
    ), "State should be the same after converting to dataset and return back"

    for i in range(4):
        torch.testing.assert_close(
            replay_buffer.states[OBS_IMAGE][i],
            reconverted_buffer.states[OBS_IMAGE][i],
            rtol=0.4,
            atol=0.004,
        )

    # The 2, 3 frames have done flag, so their values will be equal to the current state
    for i in range(2):
        # In the current implementation we take the next state from the `states` and ignore `next_states`
        next_index = (i + 1) % 4

        torch.testing.assert_close(
            replay_buffer.states[OBS_IMAGE][next_index],
            reconverted_buffer.next_states[OBS_IMAGE][i],
            rtol=0.4,
            atol=0.004,
        )

    for i in range(2, 4):
        assert torch.equal(
            replay_buffer.states[OBS_STATE][i],
            reconverted_buffer.next_states[OBS_STATE][i],
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
        action_val = batch[ACTION][i].item()
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
    replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, dummy_state_4, True, True)

    optimized_replay_buffer = create_empty_replay_buffer(True)
    optimized_replay_buffer.add(dummy_state_1, dummy_action_1, 1.0, dummy_state_2, False, False)
    optimized_replay_buffer.add(dummy_state_2, dummy_action_2, 1.0, dummy_state_3, False, False)
    optimized_replay_buffer.add(dummy_state_3, dummy_action_3, 1.0, dummy_state_4, False, False)
    optimized_replay_buffer.add(dummy_state_4, dummy_action_4, 1.0, None, True, True)

    assert get_object_memory(optimized_replay_buffer) < get_object_memory(replay_buffer), (
        "Optimized replay buffer should be smaller than the original replay buffer"
    )


def test_check_image_augmentations_with_drq_and_dummy_image_augmentation_function(dummy_state, dummy_action):
    def dummy_image_augmentation_function(x):
        return torch.ones_like(x) * 10

    replay_buffer = create_empty_replay_buffer(
        use_drq=True, image_augmentation_function=dummy_image_augmentation_function
    )

    replay_buffer.add(dummy_state, dummy_action, 1.0, dummy_state, False, False)

    sampled_transitions = replay_buffer.sample(1)
    assert torch.all(sampled_transitions["state"][OBS_IMAGE] == 10), "Image augmentations should be applied"
    assert torch.all(sampled_transitions["next_state"][OBS_IMAGE] == 10), (
        "Image augmentations should be applied"
    )


def test_check_image_augmentations_with_drq_and_default_image_augmentation_function(
    dummy_state, dummy_action
):
    replay_buffer = create_empty_replay_buffer(use_drq=True)

    replay_buffer.add(dummy_state, dummy_action, 1.0, dummy_state, False, False)

    # Let's check that it doesn't fail and shapes are correct
    sampled_transitions = replay_buffer.sample(1)
    assert sampled_transitions["state"][OBS_IMAGE].shape == (1, 3, 84, 84)
    assert sampled_transitions["next_state"][OBS_IMAGE].shape == (1, 3, 84, 84)


def test_random_crop_vectorized_basic():
    # Create a batch of 2 images with known patterns
    batch_size, channels, height, width = 2, 3, 10, 8
    images = torch.zeros((batch_size, channels, height, width))

    # Fill with unique values for testing
    for b in range(batch_size):
        images[b] = b + 1

    crop_size = (6, 4)  # Smaller than original
    cropped = random_crop_vectorized(images, crop_size)

    # Check output shape
    assert cropped.shape == (batch_size, channels, *crop_size)

    # Check that values are preserved (should be either 1s or 2s for respective batches)
    assert torch.all(cropped[0] == 1)
    assert torch.all(cropped[1] == 2)


def test_random_crop_vectorized_invalid_size():
    images = torch.zeros((2, 3, 10, 8))

    # Test crop size larger than image
    with pytest.raises(ValueError, match="Requested crop size .* is bigger than the image size"):
        random_crop_vectorized(images, (12, 8))

    with pytest.raises(ValueError, match="Requested crop size .* is bigger than the image size"):
        random_crop_vectorized(images, (10, 10))


def _populate_buffer_for_async_test(capacity: int = 10) -> ReplayBuffer:
    """Create a small buffer with deterministic 3×128×128 images and 11-D state."""
    buffer = ReplayBuffer(
        capacity=capacity,
        device="cpu",
        state_keys=[OBS_IMAGE, OBS_STATE],
        storage_device="cpu",
    )

    for i in range(capacity):
        img = torch.ones(3, 128, 128) * i
        state_vec = torch.arange(11).float() + i
        state = {
            OBS_IMAGE: img,
            OBS_STATE: state_vec,
        }
        buffer.add(
            state=state,
            action=torch.tensor([0.0]),
            reward=0.0,
            next_state=state,
            done=False,
            truncated=False,
        )
    return buffer


def test_async_iterator_shapes_basic():
    buffer = _populate_buffer_for_async_test()
    batch_size = 2
    iterator = buffer.get_iterator(batch_size=batch_size, async_prefetch=True, queue_size=1)
    batch = next(iterator)

    images = batch["state"][OBS_IMAGE]
    states = batch["state"][OBS_STATE]

    assert images.shape == (batch_size, 3, 128, 128)
    assert states.shape == (batch_size, 11)

    next_images = batch["next_state"][OBS_IMAGE]
    next_states = batch["next_state"][OBS_STATE]

    assert next_images.shape == (batch_size, 3, 128, 128)
    assert next_states.shape == (batch_size, 11)


def test_async_iterator_multiple_iterations():
    buffer = _populate_buffer_for_async_test()
    batch_size = 2
    iterator = buffer.get_iterator(batch_size=batch_size, async_prefetch=True, queue_size=2)

    for _ in range(5):
        batch = next(iterator)
        images = batch["state"][OBS_IMAGE]
        states = batch["state"][OBS_STATE]
        assert images.shape == (batch_size, 3, 128, 128)
        assert states.shape == (batch_size, 11)

        next_images = batch["next_state"][OBS_IMAGE]
        next_states = batch["next_state"][OBS_STATE]
        assert next_images.shape == (batch_size, 3, 128, 128)
        assert next_states.shape == (batch_size, 11)

    # Ensure iterator can be disposed without blocking
    del iterator
