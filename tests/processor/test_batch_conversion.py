import torch

from lerobot.processor.pipeline import (
    RobotProcessor,
    TransitionIndex,
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
    assert isinstance(transition[TransitionIndex.OBSERVATION], dict)
    assert "observation.image.top" in transition[TransitionIndex.OBSERVATION]
    assert "observation.image.left" in transition[TransitionIndex.OBSERVATION]
    assert "observation.state" in transition[TransitionIndex.OBSERVATION]

    # Check values are preserved
    assert torch.allclose(
        transition[TransitionIndex.OBSERVATION]["observation.image.top"], batch["observation.image.top"]
    )
    assert torch.allclose(
        transition[TransitionIndex.OBSERVATION]["observation.image.left"], batch["observation.image.left"]
    )
    assert transition[TransitionIndex.OBSERVATION]["observation.state"] == [1, 2, 3, 4]

    # Check other fields
    assert transition[TransitionIndex.ACTION] == "action_data"
    assert transition[TransitionIndex.REWARD] == 1.5
    assert transition[TransitionIndex.DONE]
    assert not transition[TransitionIndex.TRUNCATED]
    assert transition[TransitionIndex.INFO] == {"episode": 42}
    assert transition[TransitionIndex.COMPLEMENTARY_DATA] == {}


def test_transition_to_batch_observation_flattening():
    """Test that _default_transition_to_batch correctly flattens observation dict."""
    observation_dict = {
        "observation.image.top": torch.randn(1, 3, 128, 128),
        "observation.image.left": torch.randn(1, 3, 128, 128),
        "observation.state": [1, 2, 3, 4],
    }

    transition = (
        observation_dict,  # observation
        "action_data",  # action
        1.5,  # reward
        True,  # done
        False,  # truncated
        {"episode": 42},  # info
        {},  # complementary_data
    )

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
    assert transition[TransitionIndex.OBSERVATION] is None

    # Check other fields
    assert transition[TransitionIndex.ACTION] == "action_data"
    assert transition[TransitionIndex.REWARD] == 2.0
    assert not transition[TransitionIndex.DONE]
    assert transition[TransitionIndex.TRUNCATED]
    assert transition[TransitionIndex.INFO] == {"test": "no_obs"}

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
    assert transition[TransitionIndex.OBSERVATION] == {"observation.state": "minimal_state"}
    assert transition[TransitionIndex.ACTION] == "minimal_action"

    # Check defaults
    assert transition[TransitionIndex.REWARD] == 0.0
    assert not transition[TransitionIndex.DONE]
    assert not transition[TransitionIndex.TRUNCATED]
    assert transition[TransitionIndex.INFO] == {}
    assert transition[TransitionIndex.COMPLEMENTARY_DATA] == {}

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
    assert transition[TransitionIndex.OBSERVATION] is None
    assert transition[TransitionIndex.ACTION] is None
    assert transition[TransitionIndex.REWARD] == 0.0
    assert not transition[TransitionIndex.DONE]
    assert not transition[TransitionIndex.TRUNCATED]
    assert transition[TransitionIndex.INFO] == {}
    assert transition[TransitionIndex.COMPLEMENTARY_DATA] == {}

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
        reward = tr[TransitionIndex.REWARD] * 2 if tr[TransitionIndex.REWARD] is not None else 0.0
        return (
            tr[TransitionIndex.OBSERVATION],
            tr[TransitionIndex.ACTION],
            reward,
            tr[TransitionIndex.DONE],
            tr[TransitionIndex.TRUNCATED],
            tr[TransitionIndex.INFO],
            tr[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def to_batch(tr):
        # Custom converter that adds a custom field
        batch = _default_transition_to_batch(tr)
        batch["custom_field"] = "custom_value"
        return batch

    proc = RobotProcessor([], to_transition=to_tr, to_batch=to_batch)
    batch = _dummy_batch()
    out = proc(batch)

    # Check that custom modifications were applied
    assert out["next.reward"] == batch["next.reward"] * 2
    assert out["custom_field"] == "custom_value"

    # Check that observation.* keys are still preserved
    original_obs_keys = {k: v for k, v in batch.items() if k.startswith("observation.")}
    output_obs_keys = {k: v for k, v in out.items() if k.startswith("observation.")}

    assert set(original_obs_keys.keys()) == set(output_obs_keys.keys())
