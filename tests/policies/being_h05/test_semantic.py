import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.being_h05.configuration_being_h05 import ROBOCASA_CAMERA_KEYS, BeingH05Config
from lerobot.policies.being_h05.processor_being_h05 import (
    ACTION_SLOTS,
    STATE_SLOTS,
    BeingH05SemanticPackStep,
    inverse_normalize,
    make_being_h05_pre_post_processors,
    normalize,
    pack_named,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def _named_state(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        f"observation.state.{name}": torch.zeros(batch, end - start)
        for name, (start, end) in STATE_SLOTS.items()
    }


def test_semantic_slots_and_missing_modality_masks():
    named = {"eef_position": torch.tensor([[1.0, 2.0, 3.0]]), "control_mode": torch.ones(1, 1)}
    state, state_valid = pack_named({"eef_position": named["eef_position"]}, STATE_SLOTS)
    action, action_valid = pack_named({"control_mode": named["control_mode"]}, ACTION_SLOTS)
    assert state.shape == action.shape == (1, 200)
    assert state_valid[0, :3].all() and state_valid.sum() == 3
    assert action[0, 74] == 1 and action_valid.sum() == 1
    assert not state_valid[0, 3:].any()


@pytest.mark.parametrize(
    ("mode", "stats", "expected"),
    [
        ("q99", {"q01": [2.0], "q99": [2.0]}, 1.0),
        ("mean_std", {"mean": [2.0], "std": [0.0]}, 3.0),
        ("min_max", {"min": [2.0], "max": [2.0]}, 0.0),
    ],
)
def test_author_constant_dimension_normalization(mode, stats, expected):
    value = torch.tensor([[3.0]])
    result = normalize(value, mode, stats)
    assert result.item() == expected
    if mode != "q99":
        assert torch.isfinite(inverse_normalize(result, mode, stats)).all()


def test_raw_task_reaches_audit_hook_unchanged_and_all_cameras():
    raw = "Pick the red mug — then stop?!"
    action = torch.zeros(1, 16, 12)
    observation = _named_state()
    for key in ROBOCASA_CAMERA_KEYS:
        observation[key] = torch.rand(1, 3, 256, 256)
    transition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {"task": [raw]},
    }
    step = BeingH05SemanticPackStep(
        image_keys=ROBOCASA_CAMERA_KEYS,
        prompt_template="TASK={task_description}; K={k}",
        chunk_size=16,
    )
    result = step(transition)
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["being_h05_raw_task"] == [raw]
    assert complementary["being_h05_prompt"] == [f"TASK={raw}; K=16"]
    assert result[TransitionKey.OBSERVATION]["being_h05.pixel_values"].shape == (1, 3, 3, 224, 224)
    assert result[TransitionKey.OBSERVATION]["being_h05.image_valid"].all()


def test_missing_middle_camera_is_masked_without_changing_camera_roles():
    transition = {
        TransitionKey.OBSERVATION: {
            **_named_state(),
            ROBOCASA_CAMERA_KEYS[0]: torch.rand(1, 3, 256, 256),
            ROBOCASA_CAMERA_KEYS[2]: torch.rand(1, 3, 256, 256),
        },
        TransitionKey.ACTION: None,
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["close the drawer"]},
    }
    step = BeingH05SemanticPackStep(
        image_keys=ROBOCASA_CAMERA_KEYS,
        prompt_template="{task_description} {k}",
        chunk_size=16,
    )
    observation = step(transition)[TransitionKey.OBSERVATION]
    assert observation["being_h05.image_valid"].tolist() == [[True, False, True]]


def test_config_and_factories_are_wired_without_importing_author_dependencies():
    config = make_policy_config(
        "being_h05",
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **{
                f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
                for name, (start, end) in STATE_SLOTS.items()
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )
    assert isinstance(config, BeingH05Config)
    assert get_policy_class("being_h05").name == "being_h05"
    preprocessor, postprocessor = make_pre_post_processors(config)
    assert isinstance(preprocessor.steps[1], BeingH05SemanticPackStep)
    assert postprocessor.steps[0].get_config() == {}


def test_processor_pipeline_save_reload(tmp_path):
    config = BeingH05Config(
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **{
                f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
                for name, (start, end) in STATE_SLOTS.items()
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )
    pre, post = make_being_h05_pre_post_processors(config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=str(tmp_path))
    assert any(isinstance(step, BeingH05SemanticPackStep) for step in loaded_pre.steps)
    assert loaded_post.name == "policy_postprocessor"
    semantic_action = torch.zeros(1, 200)
    semantic_action[:, 0:3] = torch.tensor([0.1, 0.2, 0.3])
    semantic_action[:, 18] = 1
    environment_action = loaded_post(semantic_action)
    assert environment_action.shape == (1, 12)
    torch.testing.assert_close(environment_action[:, 0:3], semantic_action[:, 0:3])
    assert environment_action[:, 6].item() == 1
    assert environment_action[:, 11].item() == 0
