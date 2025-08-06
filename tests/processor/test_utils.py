import numpy as np
import pytest
import torch

from lerobot.processor.pipeline import TransitionKey
from lerobot.processor.utils import (
    to_dataset_frame,
    to_output_robot_action,
    to_transition_robot_observation,
    to_transition_teleop_action,
)


def test_to_transition_teleop_action_prefix_and_tensor_conversion():
    # Scalars, arrays, and "image-like" uint8 arrays are supported
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    act = {
        "ee.x": 0.5,  # scalar -> torch tensor
        "delta": np.array([1.0, 2.0]),  # ndarray -> torch tensor
        "raw_img": img,  # uint8 HWC -> passthrough ndarray
    }

    tr = to_transition_teleop_action(act)

    # Should be an EnvTransition-like dict with ACTION populated
    assert isinstance(tr, dict)
    assert TransitionKey.ACTION in tr
    assert "action.ee.x" in tr[TransitionKey.ACTION]
    assert "action.delta" in tr[TransitionKey.ACTION]
    assert "action.raw_img" in tr[TransitionKey.ACTION]

    # Types: scalars/arrays -> torch tensor; images -> np.ndarray
    assert isinstance(tr[TransitionKey.ACTION]["action.ee.x"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["action.ee.x"].dtype == torch.float32
    assert tr[TransitionKey.ACTION]["action.ee.x"].item() == pytest.approx(0.5)

    assert isinstance(tr[TransitionKey.ACTION]["action.delta"], torch.Tensor)
    assert tr[TransitionKey.ACTION]["action.delta"].dtype == torch.float32
    assert tr[TransitionKey.ACTION]["action.delta"].shape == (2,)
    assert torch.allclose(tr[TransitionKey.ACTION]["action.delta"], torch.tensor([1.0, 2.0]))

    assert isinstance(tr[TransitionKey.ACTION]["action.raw_img"], np.ndarray)
    assert tr[TransitionKey.ACTION]["action.raw_img"].dtype == np.uint8
    assert tr[TransitionKey.ACTION]["action.raw_img"].shape == (8, 12, 3)

    # Observation is created as empty dict by make_transition
    assert TransitionKey.OBSERVATION in tr
    assert isinstance(tr[TransitionKey.OBSERVATION], dict)
    assert tr[TransitionKey.OBSERVATION] == {}


def test_to_transition_robot_observation_state_vs_images_split():
    # Create an observation with mixed content
    img = np.full((10, 20, 3), 255, dtype=np.uint8)  # image (uint8 HWC)
    obs = {
        "j1.pos": 10.0,  # scalar -> state -> torch tensor
        "j2.pos": np.float32(20.0),  # scalar np -> state -> torch tensor
        "image_front": img,  # -> images passthrough
        "flag": np.int32(7),  # scalar -> state -> torch tensor
        "arr": np.array([1.5, 2.5]),  # vector -> state -> torch tensor
    }

    tr = to_transition_robot_observation(obs)
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
            "action.j1.pos": 11.0,  # keep -> "j1.pos"
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
        # Image spec (video/image dtype acceptable): names unimportant here
        "observation.images.front": {"dtype": "image", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
    }

    to_out = to_dataset_frame(features)

    # Build two transitions to be merged: teleop (action) and robot obs (state/images)
    img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    teleop_transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: {
            "action.j1.pos": torch.tensor(1.1),  # will be picked in vector order
            "action.j2.pos": torch.tensor(2.2),
            # "action.gripper.pos" intentionally missing -> default 0.0
            "action.ee.x": 0.5,  # should be ignored by final robot_action output, but stored in action vector only if present in names (it's not)
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "frame_is_pad": True,  # should be copied to batch
            "task": "Pick cube",  # special 'task' key should be propagated
        },
    }

    robot_transition = {
        TransitionKey.OBSERVATION: {
            "observation.state.j1.pos": torch.tensor(10.0),
            "observation.state.j2.pos": torch.tensor(20.0),
            "observation.images.front": img,  # passthrough
        },
        TransitionKey.REWARD: torch.tensor(5.0),
        TransitionKey.DONE: True,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {"note": "ok"},
    }

    # Merge two transitions -> to_out should combine action, observation, images, and metadata
    batch = to_out([teleop_transition, robot_transition])

    # 1) Images are passed through ONLY if declared in features and present in the obs transition
    assert "observation.images.front" in batch
    assert batch["observation.images.front"].shape == img.shape
    assert batch["observation.images.front"].dtype == np.uint8
    assert np.shares_memory(batch["observation.images.front"], img) or np.array_equal(
        batch["observation.images.front"], img
    )

    # 2) observation.state packed as vector in the declared order
    assert "observation.state" in batch
    obs_vec = batch["observation.state"]
    assert isinstance(obs_vec, np.ndarray) and obs_vec.dtype == np.float32
    assert obs_vec.shape == (2,)
    assert obs_vec[0] == pytest.approx(10.0)  # j1.pos
    assert obs_vec[1] == pytest.approx(20.0)  # j2.pos

    # 3) action packed in the declared order with missing default 0.0
    assert "action" in batch
    act_vec = batch["action"]
    assert isinstance(act_vec, np.ndarray) and act_vec.dtype == np.float32
    assert act_vec.shape == (3,)
    assert act_vec[0] == pytest.approx(1.1)  # j1.pos
    assert act_vec[1] == pytest.approx(2.2)  # j2.pos
    assert act_vec[2] == pytest.approx(0.0)  # gripper.pos missing -> default 0.0

    # 4) next.* metadata
    assert batch["next.reward"] == pytest.approx(5.0)
    assert batch["next.done"] is True
    assert batch["next.truncated"] is False
    assert batch["info"] == {"note": "ok"}

    # 5) complementary data: *_is_pad and 'task'
    assert batch["frame_is_pad"] is True
    assert batch["task"] == "Pick cube"


def test_to_dataset_frame_single_transition_works_and_last_writer_wins():
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
    }
    to_out = to_dataset_frame(features)

    tr1 = {
        TransitionKey.OBSERVATION: {"observation.state.x": torch.tensor(1.0)},
        TransitionKey.ACTION: {"action.a": torch.tensor(0.1)},
        TransitionKey.REWARD: 1.0,
    }
    tr2 = {
        TransitionKey.ACTION: {"action.b": torch.tensor(0.9)},
        TransitionKey.REWARD: 2.0,  # last writer wins
    }

    # Single element list
    b1 = to_out([tr1])
    assert np.allclose(b1["observation.state"], np.array([1.0], dtype=np.float32))
    assert np.allclose(b1["action"], np.array([0.1, 0.0], dtype=np.float32))
    assert b1.get("next.reward", None) == 1.0

    # Merge both - reward should come from tr2 (last)
    b2 = to_out([tr1, tr2])
    assert np.allclose(b2["observation.state"], np.array([1.0], dtype=np.float32))
    assert np.allclose(b2["action"], np.array([0.1, 0.9], dtype=np.float32))
    assert b2["next.reward"] == 2.0

    # Passing a single transition object (not list) also works
    b3 = to_out(tr2)
    assert np.allclose(b3["action"], np.array([0.0, 0.9], dtype=np.float32))
    assert "observation.state" in b3  # filled with default 0.0
    assert np.allclose(b3["observation.state"], np.array([0.0], dtype=np.float32))
