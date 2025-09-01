import numpy as np
import pytest
import torch

from lerobot.processor.converters import (
    robot_observation_to_transition,
    to_dataset_frame,
    to_output_robot_action,
    to_transition_teleop_action,
)
from lerobot.processor.pipeline import TransitionKey


def test_to_transition_teleop_action_prefix_and_tensor_conversion():
    # Scalars, arrays, and "image-like" uint8 arrays are supported
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    act = {
        "ee.x": 0.5,  # scalar to torch tensor
        "delta": np.array([1.0, 2.0]),  # ndarray to torch tensor
        "raw_img": img,  # uint8 HWC to passthrough ndarray
    }

    tr = to_transition_teleop_action(act)

    # Should be an EnvTransition-like dict with ACTION populated
    assert isinstance(tr, dict)
    assert TransitionKey.ACTION in tr
    assert "action.ee.x" in tr[TransitionKey.ACTION]
    assert "action.delta" in tr[TransitionKey.ACTION]
    assert "action.raw_img" in tr[TransitionKey.ACTION]

    # Types: scalars/arrays -> torch tensor; images to np.ndarray
    assert isinstance(tr[TransitionKey.ACTION]["action.ee.x"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["action.ee.x"].item() == pytest.approx(0.5)

    assert isinstance(tr[TransitionKey.ACTION]["action.delta"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["action.delta"].shape == (2,)
    assert torch.allclose(tr[TransitionKey.ACTION]["action.delta"], torch.tensor([1.0, 2.0]))

    assert isinstance(tr[TransitionKey.ACTION]["action.raw_img"], np.ndarray)
    assert tr[TransitionKey.ACTION]["action.raw_img"].dtype == np.uint8
    assert tr[TransitionKey.ACTION]["action.raw_img"].shape == (8, 12, 3)

    # Observation is created as empty dict by make_transition
    assert TransitionKey.OBSERVATION in tr
    assert isinstance(tr[TransitionKey.OBSERVATION], dict)
    assert tr[TransitionKey.OBSERVATION] == {}


def test_robot_observation_to_transition_state_vs_images_split():
    # Create an observation with mixed content
    img = np.full((10, 20, 3), 255, dtype=np.uint8)  # image (uint8 HWC)
    obs = {
        "j1.pos": 10.0,  # scalar to state to torch tensor
        "j2.pos": np.float32(20.0),  # scalar np to state to torch tensor
        "image_front": img,  # to images passthrough
        "flag": np.int32(7),  # scalar to state to torch tensor
        "arr": np.array([1.5, 2.5]),  # vector to state to torch tensor
    }

    tr = robot_observation_to_transition(obs)
    assert isinstance(tr, dict)
    assert TransitionKey.OBSERVATION in tr

    out = tr[TransitionKey.OBSERVATION]
    # Check state keys are present and converted to tensors
    for k in ("j1.pos", "j2.pos", "flag", "arr"):
        key = f"observation.state.{k}"
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
            "action.j1.pos": 11.0,  # keep "j1.pos"
            "action.gripper.pos": torch.tensor(33.0),  # keep: tensor accepted
            "action.ee.x": 0.5,  # ignore (doesn't end with .pos)
            "misc": "ignore_me",  # ignore (no 'action.' prefix)
        }
    }

    out = to_output_robot_action(tr)
    # Only ".pos" keys with "action." prefix are retained and stripped to base names
    assert set(out.keys()) == {"j1.pos", "gripper.pos"}
    # Values converted to float
    assert isinstance(out["j1.pos"], float)
    assert isinstance(out["gripper.pos"], float)
    assert out["j1.pos"] == pytest.approx(11.0)
    assert out["gripper.pos"] == pytest.approx(33.0)


def test_to_dataset_frame_merge_and_pack_vectors_and_metadata():
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
    batch = to_dataset_frame([teleop_transition, robot_transition], features)

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
