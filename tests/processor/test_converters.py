import numpy as np
import pytest
import torch

from lerobot.processor import TransitionKey
from lerobot.processor.converters import (
    action_to_transition,
    batch_to_transition,
    observation_to_transition,
    to_tensor,
    transition_to_batch,
    transition_to_dataset_frame,
    transition_to_robot_action,
)


def test_to_transition_teleop_action_prefix_and_tensor_conversion():
    # Scalars, arrays, and uint8 arrays are all converted to tensors
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    act = {
        "ee.x": 0.5,  # scalar to torch tensor
        "delta": np.array([1.0, 2.0]),  # ndarray to torch tensor
        "raw_img": img,  # uint8 HWC to torch tensor
    }

    tr = action_to_transition(act)

    # Should be an EnvTransition-like dict with ACTION populated
    assert isinstance(tr, dict)
    assert TransitionKey.ACTION in tr
    assert "ee.x" in tr[TransitionKey.ACTION]
    assert "delta" in tr[TransitionKey.ACTION]
    assert "raw_img" in tr[TransitionKey.ACTION]

    # Types: all values -> torch tensor
    assert isinstance(tr[TransitionKey.ACTION]["ee.x"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["ee.x"].item() == pytest.approx(0.5)

    assert isinstance(tr[TransitionKey.ACTION]["delta"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["delta"].shape == (2,)
    assert torch.allclose(tr[TransitionKey.ACTION]["delta"], torch.tensor([1.0, 2.0]))

    assert isinstance(tr[TransitionKey.ACTION]["raw_img"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["raw_img"].dtype == torch.float32  # converted from uint8
    assert tr[TransitionKey.ACTION]["raw_img"].shape == (8, 12, 3)

    # Observation is created as empty dict by make_transition
    assert TransitionKey.OBSERVATION in tr
    assert isinstance(tr[TransitionKey.OBSERVATION], dict)
    assert tr[TransitionKey.OBSERVATION] == {}


def test_to_transition_robot_observation_state_vs_images_split():
    # Create an observation with mixed content
    img = np.full((10, 20, 3), 255, dtype=np.uint8)  # image (uint8 HWC)
    obs = {
        "j1.pos": 10.0,  # scalar to state to torch tensor
        "j2.pos": np.float32(20.0),  # scalar np to state to torch tensor
        "image_front": img,  # to images passthrough
        "flag": np.int32(7),  # scalar to state to torch tensor
        "arr": np.array([1.5, 2.5]),  # vector to state to torch tensor
    }

    tr = observation_to_transition(obs)
    assert isinstance(tr, dict)
    assert TransitionKey.OBSERVATION in tr

    out = tr[TransitionKey.OBSERVATION]
    # Check state keys are present and converted to tensors
    for k in ("j1.pos", "j2.pos", "flag", "arr"):
        key = f"{k}"
        assert key in out
        v = out[key]
        if k != "arr":
            assert isinstance(v, torch.Tensor) and v.ndim == 0
        else:
            assert isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape == (2,)

    # Check image present as is
    assert "observation.images.image_front" in out
    assert isinstance(out["observation.images.image_front"], np.ndarray)
    assert out["observation.images.image_front"].dtype == np.uint8
    assert out["observation.images.image_front"].shape == (10, 20, 3)

    # ACTION should be empty dict by make_transition
    assert TransitionKey.ACTION in tr
    assert isinstance(tr[TransitionKey.ACTION], dict)
    assert tr[TransitionKey.ACTION] == {}


def test_to_output_robot_action_strips_prefix_and_filters_pos_keys_only():
    # Build a transition with mixed action keys
    tr = {
        TransitionKey.ACTION: {
            "j1.pos": 11.0,  # keep "j1.pos"
            "gripper.pos": torch.tensor(33.0),  # keep: tensor accepted
            "ee.x": 0.5,  # ignore (doesn't end with .pos)
            "misc": "ignore_me",  # ignore (no 'action.' prefix)
        }
    }

    out = transition_to_robot_action(tr)
    # Only ".pos" keys with "action." prefix are retained and stripped to base names
    assert set(out.keys()) == {"j1.pos", "gripper.pos"}
    # Values converted to float
    assert isinstance(out["j1.pos"], float)
    assert isinstance(out["gripper.pos"], float)
    assert out["j1.pos"] == pytest.approx(11.0)
    assert out["gripper.pos"] == pytest.approx(33.0)


def test_transition_to_dataset_frame_merge_and_pack_vectors_and_metadata():
    # Fabricate dataset features (as stored in dataset.meta["features"])
    features = {
        # Action vector: 3 elements in specific order
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["j1.pos", "j2.pos", "gripper.pos"],
        },
        # Observation state vector: 2 elements
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["j1.pos", "j2.pos"],
        },
        # Image spec (video/image dtype acceptable)
        "observation.images.front": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["h", "w", "c"],
        },
    }

    # Build two transitions to be merged: teleop (action) and robot obs (state/images)
    img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    teleop_transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: {
            "action.j1.pos": torch.tensor(1.1),
            "action.j2.pos": torch.tensor(2.2),
            # gripper.pos missing → defaults to 0.0
            "action.ee.x": 0.5,  # ignored, not in features["action"]["names"]
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "frame_is_pad": True,
            "task": "Pick cube",
        },
    }

    robot_transition = {
        TransitionKey.OBSERVATION: {
            "observation.state.j1.pos": torch.tensor(10.0),
            "observation.state.j2.pos": torch.tensor(20.0),
            "observation.images.front": img,
        },
        TransitionKey.REWARD: torch.tensor(5.0),
        TransitionKey.DONE: True,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {"note": "ok"},
    }

    # Directly call the refactored function
    batch = transition_to_dataset_frame([teleop_transition, robot_transition], features)

    # Images passthrough
    assert "observation.images.front" in batch
    assert batch["observation.images.front"].shape == img.shape
    assert batch["observation.images.front"].dtype == np.uint8
    assert np.shares_memory(batch["observation.images.front"], img) or np.array_equal(
        batch["observation.images.front"], img
    )

    # Observation.state vector
    assert "observation.state" in batch
    obs_vec = batch["observation.state"]
    assert isinstance(obs_vec, np.ndarray) and obs_vec.dtype == np.float32
    assert obs_vec.shape == (2,)
    assert obs_vec[0] == pytest.approx(10.0)
    assert obs_vec[1] == pytest.approx(20.0)

    # Action vector
    assert "action" in batch
    act_vec = batch["action"]
    assert isinstance(act_vec, np.ndarray) and act_vec.dtype == np.float32
    assert act_vec.shape == (3,)
    assert act_vec[0] == pytest.approx(1.1)
    assert act_vec[1] == pytest.approx(2.2)
    assert act_vec[2] == pytest.approx(0.0)  # default for missing gripper.pos

    # Next.* metadata
    assert batch["next.reward"] == pytest.approx(5.0)
    assert batch["next.done"] is True
    assert batch["next.truncated"] is False

    # Complementary data
    assert batch["frame_is_pad"] is True
    assert batch["task"] == "Pick cube"


# Tests for the unified to_tensor function
def test_to_tensor_numpy_arrays():
    """Test to_tensor with various numpy arrays."""
    # Regular numpy array
    arr = np.array([1.0, 2.0, 3.0])
    result = to_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # Different numpy dtypes should convert to float32 by default
    int_arr = np.array([1, 2, 3], dtype=np.int64)
    result = to_tensor(int_arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # uint8 arrays (previously "preserved") should now convert
    uint8_arr = np.array([100, 150, 200], dtype=np.uint8)
    result = to_tensor(uint8_arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([100.0, 150.0, 200.0]))


def test_to_tensor_numpy_scalars():
    """Test to_tensor with numpy scalars (0-dimensional arrays)."""
    # numpy float32 scalar
    scalar = np.float32(3.14)
    result = to_tensor(scalar)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0  # Should be 0-dimensional tensor
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(3.14)

    # numpy int32 scalar
    int_scalar = np.int32(42)
    result = to_tensor(int_scalar)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(42.0)


def test_to_tensor_python_scalars():
    """Test to_tensor with Python scalars."""
    # Python int
    result = to_tensor(42)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(42.0)

    # Python float
    result = to_tensor(3.14)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(3.14)


def test_to_tensor_sequences():
    """Test to_tensor with lists and tuples."""
    # List
    result = to_tensor([1, 2, 3])
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # Tuple
    result = to_tensor((4.5, 5.5, 6.5))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([4.5, 5.5, 6.5]))


def test_to_tensor_existing_tensors():
    """Test to_tensor with existing PyTorch tensors."""
    # Tensor with same dtype should pass through with potential device change
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = to_tensor(tensor)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, tensor)

    # Tensor with different dtype should convert
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    result = to_tensor(int_tensor)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_dictionaries():
    """Test to_tensor with nested dictionaries."""
    # Simple dictionary
    data = {"mean": [0.1, 0.2], "std": np.array([1.0, 2.0]), "count": 42}
    result = to_tensor(data)
    assert isinstance(result, dict)
    assert isinstance(result["mean"], torch.Tensor)
    assert isinstance(result["std"], torch.Tensor)
    assert isinstance(result["count"], torch.Tensor)
    assert torch.allclose(result["mean"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(result["std"], torch.tensor([1.0, 2.0]))
    assert result["count"].item() == pytest.approx(42.0)

    # Nested dictionary
    nested = {
        "action": {"mean": [0.1, 0.2], "std": [1.0, 2.0]},
        "observation": {"mean": np.array([0.5, 0.6]), "count": 10},
    }
    result = to_tensor(nested)
    assert isinstance(result, dict)
    assert isinstance(result["action"], dict)
    assert isinstance(result["observation"], dict)
    assert isinstance(result["action"]["mean"], torch.Tensor)
    assert isinstance(result["observation"]["mean"], torch.Tensor)
    assert torch.allclose(result["action"]["mean"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(result["observation"]["mean"], torch.tensor([0.5, 0.6]))


def test_to_tensor_none_filtering():
    """Test that None values are filtered out from dictionaries."""
    data = {"valid": [1, 2, 3], "none_value": None, "nested": {"valid": [4, 5], "also_none": None}}
    result = to_tensor(data)
    assert "none_value" not in result
    assert "also_none" not in result["nested"]
    assert "valid" in result
    assert "valid" in result["nested"]
    assert torch.allclose(result["valid"], torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_dtype_parameter():
    """Test to_tensor with different dtype parameters."""
    arr = np.array([1, 2, 3])

    # Default dtype (float32)
    result = to_tensor(arr)
    assert result.dtype == torch.float32

    # Explicit float32
    result = to_tensor(arr, dtype=torch.float32)
    assert result.dtype == torch.float32

    # Float64
    result = to_tensor(arr, dtype=torch.float64)
    assert result.dtype == torch.float64

    # Preserve original dtype
    float64_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = to_tensor(float64_arr, dtype=None)
    assert result.dtype == torch.float64


def test_to_tensor_device_parameter():
    """Test to_tensor with device parameter."""
    arr = np.array([1.0, 2.0, 3.0])

    # CPU device (default)
    result = to_tensor(arr, device="cpu")
    assert result.device.type == "cpu"

    # CUDA device (if available)
    if torch.cuda.is_available():
        result = to_tensor(arr, device="cuda")
        assert result.device.type == "cuda"


def test_to_tensor_empty_dict():
    """Test to_tensor with empty dictionary."""
    result = to_tensor({})
    assert isinstance(result, dict)
    assert len(result) == 0


def test_to_tensor_unsupported_type():
    """Test to_tensor with unsupported types raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported type for tensor conversion"):
        to_tensor("unsupported_string")

    with pytest.raises(TypeError, match="Unsupported type for tensor conversion"):
        to_tensor(object())


def create_transition(
    observation=None, action=None, reward=0.0, done=False, truncated=False, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def test_batch_to_transition_with_index_fields():
    """Test that batch_to_transition handles index and task_index fields correctly."""

    # Create batch with index and task_index fields
    batch = {
        "observation.state": torch.randn(1, 7),
        "action": torch.randn(1, 4),
        "next.reward": 1.5,
        "next.done": False,
        "task": ["pick_cube"],
        "index": torch.tensor([42], dtype=torch.int64),
        "task_index": torch.tensor([3], dtype=torch.int64),
    }

    transition = batch_to_transition(batch)

    # Check basic transition structure
    assert TransitionKey.OBSERVATION in transition
    assert TransitionKey.ACTION in transition
    assert TransitionKey.COMPLEMENTARY_DATA in transition

    # Check that index and task_index are in complementary_data
    comp_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert "index" in comp_data
    assert "task_index" in comp_data
    assert "task" in comp_data

    # Verify values
    assert torch.equal(comp_data["index"], batch["index"])
    assert torch.equal(comp_data["task_index"], batch["task_index"])
    assert comp_data["task"] == batch["task"]


def testtransition_to_batch_with_index_fields():
    """Test that transition_to_batch handles index and task_index fields correctly."""

    # Create transition with index and task_index in complementary_data
    transition = create_transition(
        observation={"observation.state": torch.randn(1, 7)},
        action=torch.randn(1, 4),
        reward=1.5,
        done=False,
        complementary_data={
            "task": ["navigate"],
            "index": torch.tensor([100], dtype=torch.int64),
            "task_index": torch.tensor([5], dtype=torch.int64),
        },
    )

    batch = transition_to_batch(transition)

    # Check that index and task_index are in the batch
    assert "index" in batch
    assert "task_index" in batch
    assert "task" in batch

    # Verify values
    assert torch.equal(batch["index"], transition[TransitionKey.COMPLEMENTARY_DATA]["index"])
    assert torch.equal(batch["task_index"], transition[TransitionKey.COMPLEMENTARY_DATA]["task_index"])
    assert batch["task"] == transition[TransitionKey.COMPLEMENTARY_DATA]["task"]


def test_batch_to_transition_without_index_fields():
    """Test that conversion works without index and task_index fields."""

    # Batch without index/task_index
    batch = {
        "observation.state": torch.randn(1, 7),
        "action": torch.randn(1, 4),
        "task": ["pick_cube"],
    }

    transition = batch_to_transition(batch)
    comp_data = transition[TransitionKey.COMPLEMENTARY_DATA]

    # Should have task but not index/task_index
    assert "task" in comp_data
    assert "index" not in comp_data
    assert "task_index" not in comp_data


def test_transition_to_batch_without_index_fields():
    """Test that conversion works without index and task_index fields."""

    # Transition without index/task_index
    transition = create_transition(
        observation={"observation.state": torch.randn(1, 7)},
        action=torch.randn(1, 4),
        complementary_data={"task": ["navigate"]},
    )

    batch = transition_to_batch(transition)

    # Should have task but not index/task_index
    assert "task" in batch
    assert "index" not in batch
    assert "task_index" not in batch
