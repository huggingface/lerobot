import torch

from lerobot.processor import DataProcessorPipeline, TransitionKey
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_PREFIX, OBS_STATE


def _dummy_batch():
    """Create a dummy batch using the new format with observation.* and next.* keys."""
    return {
        f"{OBS_IMAGE}.left": torch.randn(1, 3, 128, 128),
        f"{OBS_IMAGE}.right": torch.randn(1, 3, 128, 128),
        OBS_STATE: torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        ACTION: torch.tensor([[0.5]]),
        "next.reward": 1.0,
        "next.done": False,
        "next.truncated": False,
        "info": {"key": "value"},
    }


def test_observation_grouping_roundtrip():
    """Test that observation.* keys are properly grouped and ungrouped."""
    proc = DataProcessorPipeline([])
    batch_in = _dummy_batch()
    batch_out = proc(batch_in)

    # Check that all observation.* keys are preserved
    original_obs_keys = {k: v for k, v in batch_in.items() if k.startswith(OBS_PREFIX)}
    reconstructed_obs_keys = {k: v for k, v in batch_out.items() if k.startswith(OBS_PREFIX)}

    assert set(original_obs_keys.keys()) == set(reconstructed_obs_keys.keys())

    # Check tensor values
    assert torch.allclose(batch_out[f"{OBS_IMAGE}.left"], batch_in[f"{OBS_IMAGE}.left"])
    assert torch.allclose(batch_out[f"{OBS_IMAGE}.right"], batch_in[f"{OBS_IMAGE}.right"])
    assert torch.allclose(batch_out[OBS_STATE], batch_in[OBS_STATE])

    # Check other fields
    assert torch.allclose(batch_out[ACTION], batch_in[ACTION])
    assert batch_out["next.reward"] == batch_in["next.reward"]
    assert batch_out["next.done"] == batch_in["next.done"]
    assert batch_out["next.truncated"] == batch_in["next.truncated"]
    assert batch_out["info"] == batch_in["info"]


def test_batch_to_transition_observation_grouping():
    """Test that batch_to_transition correctly groups observation.* keys."""
    batch = {
        f"{OBS_IMAGE}.top": torch.randn(1, 3, 128, 128),
        f"{OBS_IMAGE}.left": torch.randn(1, 3, 128, 128),
        OBS_STATE: [1, 2, 3, 4],
        ACTION: torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "next.reward": 1.5,
        "next.done": True,
        "next.truncated": False,
        "info": {"episode": 42},
    }

    transition = batch_to_transition(batch)

    # Check observation is a dict with all observation.* keys
    assert isinstance(transition[TransitionKey.OBSERVATION], dict)
    assert f"{OBS_IMAGE}.top" in transition[TransitionKey.OBSERVATION]
    assert f"{OBS_IMAGE}.left" in transition[TransitionKey.OBSERVATION]
    assert OBS_STATE in transition[TransitionKey.OBSERVATION]

    # Check values are preserved
    assert torch.allclose(
        transition[TransitionKey.OBSERVATION][f"{OBS_IMAGE}.top"], batch[f"{OBS_IMAGE}.top"]
    )
    assert torch.allclose(
        transition[TransitionKey.OBSERVATION][f"{OBS_IMAGE}.left"], batch[f"{OBS_IMAGE}.left"]
    )
    assert transition[TransitionKey.OBSERVATION][OBS_STATE] == [1, 2, 3, 4]

    # Check other fields
    assert torch.allclose(transition[TransitionKey.ACTION], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert transition[TransitionKey.REWARD] == 1.5
    assert transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {"episode": 42}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_transition_to_batch_observation_flattening():
    """Test that transition_to_batch correctly flattens observation dict."""
    observation_dict = {
        f"{OBS_IMAGE}.top": torch.randn(1, 3, 128, 128),
        f"{OBS_IMAGE}.left": torch.randn(1, 3, 128, 128),
        OBS_STATE: [1, 2, 3, 4],
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

    batch = transition_to_batch(transition)

    # Check that observation.* keys are flattened back to batch
    assert f"{OBS_IMAGE}.top" in batch
    assert f"{OBS_IMAGE}.left" in batch
    assert OBS_STATE in batch

    # Check values are preserved
    assert torch.allclose(batch[f"{OBS_IMAGE}.top"], observation_dict[f"{OBS_IMAGE}.top"])
    assert torch.allclose(batch[f"{OBS_IMAGE}.left"], observation_dict[f"{OBS_IMAGE}.left"])
    assert batch[OBS_STATE] == [1, 2, 3, 4]

    # Check other fields are mapped to next.* format
    assert batch[ACTION] == "action_data"
    assert batch["next.reward"] == 1.5
    assert batch["next.done"]
    assert not batch["next.truncated"]
    assert batch["info"] == {"episode": 42}


def test_no_observation_keys():
    """Test behavior when there are no observation.* keys."""
    batch = {
        ACTION: torch.tensor([1.0, 2.0]),
        "next.reward": 2.0,
        "next.done": False,
        "next.truncated": True,
        "info": {"test": "no_obs"},
    }

    transition = batch_to_transition(batch)

    # Observation should be None when no observation.* keys
    assert transition[TransitionKey.OBSERVATION] is None

    # Check other fields
    assert torch.allclose(transition[TransitionKey.ACTION], torch.tensor([1.0, 2.0]))
    assert transition[TransitionKey.REWARD] == 2.0
    assert not transition[TransitionKey.DONE]
    assert transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {"test": "no_obs"}

    # Round trip should work
    reconstructed_batch = transition_to_batch(transition)
    assert torch.allclose(reconstructed_batch[ACTION], torch.tensor([1.0, 2.0]))
    assert reconstructed_batch["next.reward"] == 2.0
    assert not reconstructed_batch["next.done"]
    assert reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {"test": "no_obs"}


def test_minimal_batch():
    """Test with minimal batch containing only observation.* and action."""
    batch = {OBS_STATE: "minimal_state", ACTION: torch.tensor([0.5])}

    transition = batch_to_transition(batch)

    # Check observation
    assert transition[TransitionKey.OBSERVATION] == {OBS_STATE: "minimal_state"}
    assert torch.allclose(transition[TransitionKey.ACTION], torch.tensor([0.5]))

    # Check defaults
    assert transition[TransitionKey.REWARD] == 0.0
    assert not transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}

    # Round trip
    reconstructed_batch = transition_to_batch(transition)
    assert reconstructed_batch[OBS_STATE] == "minimal_state"
    assert torch.allclose(reconstructed_batch[ACTION], torch.tensor([0.5]))
    assert reconstructed_batch["next.reward"] == 0.0
    assert not reconstructed_batch["next.done"]
    assert not reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {}


def test_empty_batch():
    """Test behavior with empty batch."""
    batch = {}

    transition = batch_to_transition(batch)

    # All fields should have defaults
    assert transition[TransitionKey.OBSERVATION] is None
    assert transition[TransitionKey.ACTION] is None
    assert transition[TransitionKey.REWARD] == 0.0
    assert not transition[TransitionKey.DONE]
    assert not transition[TransitionKey.TRUNCATED]
    assert transition[TransitionKey.INFO] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}

    # Round trip
    reconstructed_batch = transition_to_batch(transition)
    assert reconstructed_batch[ACTION] is None
    assert reconstructed_batch["next.reward"] == 0.0
    assert not reconstructed_batch["next.done"]
    assert not reconstructed_batch["next.truncated"]
    assert reconstructed_batch["info"] == {}


def test_complex_nested_observation():
    """Test with complex nested observation data."""
    batch = {
        f"{OBS_IMAGE}.top": {"image": torch.randn(1, 3, 128, 128), "timestamp": 1234567890},
        f"{OBS_IMAGE}.left": {"image": torch.randn(1, 3, 128, 128), "timestamp": 1234567891},
        OBS_STATE: torch.randn(7),
        ACTION: torch.randn(8),
        "next.reward": 3.14,
        "next.done": False,
        "next.truncated": True,
        "info": {"episode_length": 200, "success": True},
    }

    transition = batch_to_transition(batch)
    reconstructed_batch = transition_to_batch(transition)

    # Check that all observation keys are preserved
    original_obs_keys = {k for k in batch if k.startswith(OBS_PREFIX)}
    reconstructed_obs_keys = {k for k in reconstructed_batch if k.startswith(OBS_PREFIX)}

    assert original_obs_keys == reconstructed_obs_keys

    # Check tensor values
    assert torch.allclose(batch[OBS_STATE], reconstructed_batch[OBS_STATE])

    # Check nested dict with tensors
    assert torch.allclose(
        batch[f"{OBS_IMAGE}.top"]["image"], reconstructed_batch[f"{OBS_IMAGE}.top"]["image"]
    )
    assert torch.allclose(
        batch[f"{OBS_IMAGE}.left"]["image"], reconstructed_batch[f"{OBS_IMAGE}.left"]["image"]
    )

    # Check action tensor
    assert torch.allclose(batch[ACTION], reconstructed_batch[ACTION])

    # Check other fields
    assert batch["next.reward"] == reconstructed_batch["next.reward"]
    assert batch["next.done"] == reconstructed_batch["next.done"]
    assert batch["next.truncated"] == reconstructed_batch["next.truncated"]
    assert batch["info"] == reconstructed_batch["info"]


def test_custom_converter():
    """Test that custom converters can still be used."""

    def to_tr(batch):
        # Custom converter that modifies the reward
        tr = batch_to_transition(batch)
        # Double the reward
        reward = tr.get(TransitionKey.REWARD, 0.0)
        new_tr = tr.copy()
        new_tr[TransitionKey.REWARD] = reward * 2 if reward is not None else 0.0
        return new_tr

    def to_batch(tr):
        batch = transition_to_batch(tr)
        return batch

    processor = DataProcessorPipeline(steps=[], to_transition=to_tr, to_output=to_batch)

    batch = {
        OBS_STATE: torch.randn(1, 4),
        ACTION: torch.randn(1, 2),
        "next.reward": 1.0,
        "next.done": False,
    }

    result = processor(batch)

    # Check the reward was doubled by our custom converter
    assert result["next.reward"] == 2.0
    assert torch.allclose(result[OBS_STATE], batch[OBS_STATE])
    assert torch.allclose(result[ACTION], batch[ACTION])
