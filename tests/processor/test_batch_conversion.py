import torch

from lerobot.processor.pipeline import (
    RobotProcessor,
    TransitionKey,
    _default_batch_to_transition,
    _default_transition_to_batch,
)


def _dummy_batch():
    """Create a dummy batch using the new format with observation.* and next.* keys."""
    return {
        "observation.image.left": torch.randn(1, 3, 128, 128),
        "observation.image.right": torch.randn(1, 3, 128, 128),
        "observation.state": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "action": torch.tensor([[0.5]]),
        "next.reward": 1.0,
        "next.done": False,
        "next.truncated": False,
        "info": {"key": "value"},
    }


def test_observation_grouping_roundtrip():
    """Test that observation.* keys are properly grouped and ungrouped."""
    proc = RobotProcessor([])
    batch_in = _dummy_batch()
    batch_out = proc(batch_in)

    # Check that all observation.* keys are preserved
    original_obs_keys = {k: v for k, v in batch_in.items() if k.startswith("observation.")}
    reconstructed_obs_keys = {k: v for k, v in batch_out.items() if k.startswith("observation.")}

    assert set(original_obs_keys.keys()) == set(reconstructed_obs_keys.keys())

    # Check tensor values
    assert torch.allclose(batch_out["observation.image.left"], batch_in["observation.image.left"])
    assert torch.allclose(batch_out["observation.image.right"], batch_in["observation.image.right"])
    assert torch.allclose(batch_out["observation.state"], batch_in["observation.state"])

    # Check other fields
    assert torch.allclose(batch_out["action"], batch_in["action"])
    assert batch_out["next.reward"] == batch_in["next.reward"]
    assert batch_out["next.done"] == batch_in["next.done"]
    assert batch_out["next.truncated"] == batch_in["next.truncated"]
    assert batch_out["info"] == batch_in["info"]


def test_batch_to_transition_observation_grouping():
    """Test that _default_batch_to_transition correctly groups observation.* keys."""
    batch = {
        "observation.image.top": torch.randn(1, 3, 128, 128),
        "observation.image.left": torch.randn(1, 3, 128, 128),
        "observation.state": [1, 2, 3, 4],
        "action": "action_data",
        "next.reward": 1.5,
        "next.done": True,
        "next.truncated": False,
        "info": {"episode": 42},
    }

    transition = _default_batch_to_transition(batch)

    # Check observation is a dict with all observation.* keys
    assert isinstance(transition[TransitionKey.OBSERVATION], dict)
    assert "observation.image.top" in transition[TransitionKey.OBSERVATION]
    assert "observation.image.left" in transition[TransitionKey.OBSERVATION]
    assert "observation.state" in transition[TransitionKey.OBSERVATION]

    # Check values are preserved
    assert torch.allclose(
        transition[TransitionKey.OBSERVATION]["observation.image.top"], batch["observation.image.top"]
    )
    assert torch.allclose(
        transition[TransitionKey.OBSERVATION]["observation.image.left"], batch["observation.image.left"]
    )
    assert transition[TransitionKey.OBSERVATION]["observation.state"] == [1, 2, 3, 4]

    # Check other fields
    assert transition[TransitionKey.ACTION] == "action_data"
    assert transition[TransitionKey.REWARD] == 1.5
    assert transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {"episode": 42}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_transition_to_batch_observation_flattening():
    """Test that _default_transition_to_batch correctly flattens observation dict."""
    observation_dict = {
        "observation.image.top": torch.randn(1, 3, 128, 128),
        "observation.image.left": torch.randn(1, 3, 128, 128),
        "observation.state": [1, 2, 3, 4],
    }

    transition = {
        TransitionKey.OBSERVATION: observation_dict,
        TransitionKey.ACTION: "action_data",
        TransitionKey.REWARD: 1.5,
        TransitionKey.DONE: True,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {"episode": 42},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    batch = _default_transition_to_batch(transition)

    # Check that observation.* keys are flattened back to batch
    assert "observation.image.top" in batch
    assert "observation.image.left" in batch
    assert "observation.state" in batch

    # Check values are preserved
    assert torch.allclose(batch["observation.image.top"], observation_dict["observation.image.top"])
    assert torch.allclose(batch["observation.image.left"], observation_dict["observation.image.left"])
    assert batch["observation.state"] == [1, 2, 3, 4]

    # Check other fields are mapped to next.* format
    assert batch["action"] == "action_data"
    assert batch["next.reward"] == 1.5
    assert batch["next.done"]
    assert not batch["next.truncated"]
    assert batch["info"] == {"episode": 42}


def test_no_observation_keys():
    """Test behavior when there are no observation.* keys."""
    batch = {
        "action": "action_data",
        "next.reward": 2.0,
        "next.done": False,
        "next.truncated": True,
        "info": {"test": "no_obs"},
    }

    transition = _default_batch_to_transition(batch)

    # Observation should be None when no observation.* keys
    assert transition[TransitionKey.OBSERVATION] is None

    # Check other fields
    assert transition[TransitionKey.ACTION] == "action_data"
    assert transition[TransitionKey.REWARD] == 2.0
    assert not transition[TransitionKey.DONE]
    assert transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {"test": "no_obs"}

    # Round trip should work
    reconstructed_batch = _default_transition_to_batch(transition)
    assert reconstructed_batch["action"] == "action_data"
    assert reconstructed_batch["next.reward"] == 2.0
    assert not reconstructed_batch["next.done"]
    assert reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {"test": "no_obs"}


def test_minimal_batch():
    """Test with minimal batch containing only observation.* and action."""
    batch = {"observation.state": "minimal_state", "action": "minimal_action"}

    transition = _default_batch_to_transition(batch)

    # Check observation
    assert transition[TransitionKey.OBSERVATION] == {"observation.state": "minimal_state"}
    assert transition[TransitionKey.ACTION] == "minimal_action"

    # Check defaults
    assert transition[TransitionKey.REWARD] == 0.0
    assert not transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}

    # Round trip
    reconstructed_batch = _default_transition_to_batch(transition)
    assert reconstructed_batch["observation.state"] == "minimal_state"
    assert reconstructed_batch["action"] == "minimal_action"
    assert reconstructed_batch["next.reward"] == 0.0
    assert not reconstructed_batch["next.done"]
    assert not reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {}


def test_empty_batch():
    """Test behavior with empty batch."""
    batch = {}

    transition = _default_batch_to_transition(batch)

    # All fields should have defaults
    assert transition[TransitionKey.OBSERVATION] is None
    assert transition[TransitionKey.ACTION] is None
    assert transition[TransitionKey.REWARD] == 0.0
    assert not transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}

    # Round trip
    reconstructed_batch = _default_transition_to_batch(transition)
    assert reconstructed_batch["action"] is None
    assert reconstructed_batch["next.reward"] == 0.0
    assert not reconstructed_batch["next.done"]
    assert not reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {}


def test_complex_nested_observation():
    """Test with complex nested observation data."""
    batch = {
        "observation.image.top": {"image": torch.randn(1, 3, 128, 128), "timestamp": 1234567890},
        "observation.image.left": {"image": torch.randn(1, 3, 128, 128), "timestamp": 1234567891},
        "observation.state": torch.randn(7),
        "action": torch.randn(8),
        "next.reward": 3.14,
        "next.done": False,
        "next.truncated": True,
        "info": {"episode_length": 200, "success": True},
    }

    transition = _default_batch_to_transition(batch)
    reconstructed_batch = _default_transition_to_batch(transition)

    # Check that all observation keys are preserved
    original_obs_keys = {k for k in batch if k.startswith("observation.")}
    reconstructed_obs_keys = {k for k in reconstructed_batch if k.startswith("observation.")}

    assert original_obs_keys == reconstructed_obs_keys

    # Check tensor values
    assert torch.allclose(batch["observation.state"], reconstructed_batch["observation.state"])

    # Check nested dict with tensors
    assert torch.allclose(
        batch["observation.image.top"]["image"], reconstructed_batch["observation.image.top"]["image"]
    )
    assert torch.allclose(
        batch["observation.image.left"]["image"], reconstructed_batch["observation.image.left"]["image"]
    )

    # Check action tensor
    assert torch.allclose(batch["action"], reconstructed_batch["action"])

    # Check other fields
    assert batch["next.reward"] == reconstructed_batch["next.reward"]
    assert batch["next.done"] == reconstructed_batch["next.done"]
    assert batch["next.truncated"] == reconstructed_batch["next.truncated"]
    assert batch["info"] == reconstructed_batch["info"]


def test_custom_converter():
    """Test that custom converters can still be used."""

    def to_tr(batch):
        # Custom converter that modifies the reward
        tr = _default_batch_to_transition(batch)
        # Double the reward
        reward = tr.get(TransitionKey.REWARD, 0.0)
        new_tr = tr.copy()
        new_tr[TransitionKey.REWARD] = reward * 2 if reward is not None else 0.0
        return new_tr

    def to_batch(tr):
        batch = _default_transition_to_batch(tr)
        return batch

    processor = RobotProcessor(steps=[], to_transition=to_tr, to_output=to_batch)

    batch = {
        "observation.state": torch.randn(1, 4),
        "action": torch.randn(1, 2),
        "next.reward": 1.0,
        "next.done": False,
    }

    result = processor(batch)

    # Check the reward was doubled by our custom converter
    assert result["next.reward"] == 2.0
    assert torch.allclose(result["observation.state"], batch["observation.state"])
    assert torch.allclose(result["action"], batch["action"])
