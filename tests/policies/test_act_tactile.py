import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_tactile_feature_typing():
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
        "observation.tactile.sensor_1": {"dtype": "int16", "shape": (6, 6), "names": ["rows", "columns"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    policy_features = dataset_to_policy_features(features)
    assert policy_features["observation.tactile.sensor_1"].type is FeatureType.TACTILE
    assert policy_features["observation.tactile.sensor_1"].shape == (6, 6)
    assert policy_features["observation.state"].type is FeatureType.STATE


def _tactile_input_features(shapes):
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    for key, shape in shapes.items():
        feats[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    return feats


def test_tactile_config_property_and_validation():
    input_features = _tactile_input_features(
        {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)}
    )
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=True,
    )
    assert list(config.tactile_features) == ["observation.tactile.sensor_1", "observation.tactile.sensor_2"]
    assert config.normalization_mapping["TACTILE"].value == "MEAN_STD"

    with pytest.raises(ValueError):
        ACTConfig(
            input_features=input_features,
            output_features=output_features,
            use_tactile=True,
            tactile_encoder_type="bogus",
        )

    with pytest.raises(ValueError):
        # use_tactile enabled but no tactile features present
        ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            },
            output_features=output_features,
            use_tactile=True,
        )

    # Tactile as the sole input modality must be accepted (no images / state / env_state).
    tactile_only = ACTConfig(
        input_features={
            "observation.tactile.sensor_1": PolicyFeature(type=FeatureType.TACTILE, shape=(6, 6))
        },
        output_features=output_features,
        use_tactile=True,
    )
    tactile_only.validate_features()  # must not raise
    assert list(tactile_only.tactile_features) == ["observation.tactile.sensor_1"]


_B = 2
_CHUNK = 5
_ACTION_DIM = 6
_STATE_DIM = 10
_IMG = (3, 96, 96)
_SHAPES = {
    "single": {"observation.tactile.sensor_1": (6, 6)},
    "multi": {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)},
    "tactile_only": {"observation.tactile.sensor_1": (6, 6)},
}


def _encoder_seq_len_hook(policy):
    captured = {}

    def hook(module, args, kwargs):
        captured["seq"] = args[0].shape[0]

    handle = policy.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, handle


def _build_batch(shapes, with_image_state):
    batch = {
        ACTION: torch.randn(_B, _CHUNK, _ACTION_DIM),
        "action_is_pad": torch.zeros(_B, _CHUNK, dtype=torch.bool),
    }
    if with_image_state:
        batch[OBS_STATE] = torch.randn(_B, _STATE_DIM)
        batch[f"{OBS_IMAGES}.cam"] = torch.rand(_B, *_IMG)
    for key, shape in shapes.items():
        batch[key] = torch.randint(0, 4096, (_B, *shape), dtype=torch.int16)
    return batch


@pytest.mark.parametrize("encoder_type", ["cnn", "attention"])
@pytest.mark.parametrize("scenario", ["single", "multi", "tactile_only"])
def test_act_tactile_forward(encoder_type, scenario):
    shapes = _SHAPES[scenario]
    with_image_state = scenario != "tactile_only"
    use_vae = with_image_state  # no proprioceptive state in tactile_only -> skip VAE encoder

    input_features = {}
    if with_image_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,))
        input_features[f"{OBS_IMAGES}.cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=_IMG)
    for key, shape in shapes.items():
        input_features[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
        use_tactile=True,
        tactile_encoder_type=encoder_type,
        n_tactile_tokens=4,
        use_vae=use_vae,
    )
    policy = ACTPolicy(config)
    policy.train()

    batch = _build_batch(shapes, with_image_state)
    captured, handle = _encoder_seq_len_hook(policy)
    loss, loss_dict = policy.forward(batch)
    handle.remove()
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    seq_with_tactile = captured["seq"]

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (_B, _CHUNK, _ACTION_DIM)

    if with_image_state:
        base_features = {k: v for k, v in input_features.items() if v.type is not FeatureType.TACTILE}
        base_config = ACTConfig(
            input_features=base_features,
            output_features=output_features,
            chunk_size=_CHUNK,
            n_action_steps=_CHUNK,
            use_tactile=False,
            use_vae=use_vae,
        )
        base_policy = ACTPolicy(base_config)
        base_policy.train()
        base_batch = {k: v for k, v in batch.items() if not k.startswith("observation.tactile")}
        base_captured, base_handle = _encoder_seq_len_hook(base_policy)
        base_policy.forward(base_batch)
        base_handle.remove()
        expected_extra = len(shapes) * config.n_tactile_tokens
        assert seq_with_tactile - base_captured["seq"] == expected_extra
