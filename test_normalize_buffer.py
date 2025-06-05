import pytest
import torch

from lerobot.common.policies.normalize import (
    Normalize,
    NormalizeBuffer,
    Unnormalize,
    UnnormalizeBuffer,
)
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def _dummy_setup():
    # feature definitions
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }

    # map feature types to a normalization strategy
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.MIN_MAX,
    }

    # build statistics (include all stats for each feature)
    stats = {
        "observation.state": {
            "mean": torch.arange(5, dtype=torch.float32),
            "std": torch.arange(1, 6, dtype=torch.float32),
            "min": torch.zeros(5, dtype=torch.float32),
            "max": torch.ones(5, dtype=torch.float32) * 10.0,
        },
        # image statistics use (c,1,1) so they broadcast on spatial dims
        "observation.image": {
            "mean": torch.ones(3, 1, 1, dtype=torch.float32) * 127.5,
            "std": torch.ones(3, 1, 1, dtype=torch.float32) * 50.0,
            "min": torch.zeros(3, 1, 1, dtype=torch.float32),
            "max": torch.ones(3, 1, 1, dtype=torch.float32) * 255.0,
        },
    }

    return features, norm_map, stats


def _random_batch(stats):
    """Generate a batch consistent with the provided statistics."""
    torch.manual_seed(0)
    batch_size = 2

    state_mean = stats["observation.state"]["mean"]
    state_std = stats["observation.state"]["std"]
    state = torch.randn(batch_size, 5) * state_std + state_mean  # shape (b,5)

    image_min = stats["observation.image"]["min"]
    image_max = stats["observation.image"]["max"]
    image = torch.rand(batch_size, 3, 64, 64) * (image_max - image_min) + image_min  # shape (b,3,64,64)

    return {
        "observation.state": state,
        "observation.image": image,
    }


@pytest.mark.parametrize(
    "module_pair",
    [
        (Normalize, NormalizeBuffer),
        (Unnormalize, UnnormalizeBuffer),
    ],
)
def test_equivalence(module_pair):
    features, norm_map, stats = _dummy_setup()
    ParamCls, BufferCls = module_pair  # noqa: N806

    param_module = ParamCls(features=features, norm_map=norm_map, stats=stats)
    buffer_module = BufferCls(features=features, norm_map=norm_map, stats=stats)

    batch = _random_batch(stats)

    out_param = param_module(batch)
    out_buffer = buffer_module(batch)

    # every tensor in the output dictionaries should match closely
    for key in out_param:
        torch.testing.assert_close(out_param[key], out_buffer[key])


def test_round_trip():
    """Normalize then unnormalize should give the original input back for both impls."""
    features, norm_map, stats = _dummy_setup()

    norm_p = Normalize(features, norm_map, stats)
    unnorm_p = Unnormalize(features, norm_map, stats)

    norm_b = NormalizeBuffer(features, norm_map, stats)
    unnorm_b = UnnormalizeBuffer(features, norm_map, stats)

    batch = _random_batch(stats)
    recovered_p = unnorm_p(norm_p(batch))
    recovered_b = unnorm_b(norm_b(batch))

    for key in batch:
        torch.testing.assert_close(recovered_p[key], batch[key])
        torch.testing.assert_close(recovered_b[key], batch[key])


@pytest.mark.parametrize(
    "image_shape,use_numpy",
    [
        ((3, 64, 64), True),
        ((3, 128, 128), False),
    ],
)
def test_various_shapes_and_numpy(image_shape, use_numpy):
    """Ensure equivalence and round-trip correctness for different image shapes and numpy stats."""
    # feature definitions (state dim fixed at 5)
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
    }

    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.MIN_MAX,
    }

    # statistics (torch or numpy)
    state_mean = torch.arange(5, dtype=torch.float32)
    state_std = torch.arange(1, 6, dtype=torch.float32)
    img_min = torch.zeros(image_shape[0], 1, 1, dtype=torch.float32)
    img_max = torch.ones(image_shape[0], 1, 1, dtype=torch.float32) * 10.0  # simple range [0,10]

    if use_numpy:
        state_mean_stats = state_mean.numpy()
        state_std_stats = state_std.numpy()
        img_min_stats = img_min.numpy()
        img_max_stats = img_max.numpy()
    else:
        state_mean_stats = state_mean
        state_std_stats = state_std
        img_min_stats = img_min
        img_max_stats = img_max

    stats = {
        "observation.state": {"mean": state_mean_stats, "std": state_std_stats},
        "observation.image": {"min": img_min_stats, "max": img_max_stats},
    }

    # instantiate modules
    norm_p = Normalize(features, norm_map, stats)
    unnorm_p = Unnormalize(features, norm_map, stats)
    norm_b = NormalizeBuffer(features, norm_map, stats)
    unnorm_b = UnnormalizeBuffer(features, norm_map, stats)

    # build random batch following stats
    batch_size = 3
    torch.manual_seed(42)
    state = torch.randn(batch_size, 5) * state_std + state_mean
    image = torch.rand(batch_size, *image_shape) * (img_max - img_min) + img_min

    batch = {"observation.state": state, "observation.image": image}

    # equivalence between param and buffer implementations
    torch.testing.assert_close(norm_p(batch)["observation.state"], norm_b(batch)["observation.state"])
    torch.testing.assert_close(norm_p(batch)["observation.image"], norm_b(batch)["observation.image"])

    # round-trip
    recovered_p = unnorm_p(norm_p(batch))
    recovered_b = unnorm_b(norm_b(batch))

    for key in batch:
        torch.testing.assert_close(recovered_p[key], batch[key])
        torch.testing.assert_close(recovered_b[key], batch[key])


def test_state_dict_conversion():
    """Test that state dict can be converted from Normalize to NormalizeBuffer format."""
    from lerobot.common.policies.normalize import convert_normalize_to_buffer_state_dict

    features, norm_map, stats = _dummy_setup()

    # Create Normalize module and get its state dict
    normalize_module = Normalize(features=features, norm_map=norm_map, stats=stats)
    old_state_dict = normalize_module.state_dict()

    # Convert state dict
    new_state_dict = convert_normalize_to_buffer_state_dict(old_state_dict)

    # Create NormalizeBuffer module and load converted state dict
    buffer_module = NormalizeBuffer(features=features, norm_map=norm_map, stats=None)
    buffer_module.load_state_dict(new_state_dict)

    # Test that both modules produce the same output
    batch = _random_batch(stats)

    old_output = normalize_module(batch)
    new_output = buffer_module(batch)

    for key in old_output:
        torch.testing.assert_close(old_output[key], new_output[key])


def test_state_dict_conversion_unnormalize():
    """Test that state dict can be converted from Unnormalize to UnnormalizeBuffer format."""
    from lerobot.common.policies.normalize import convert_normalize_to_buffer_state_dict

    features, norm_map, stats = _dummy_setup()

    # Create Unnormalize module and get its state dict
    unnormalize_module = Unnormalize(features=features, norm_map=norm_map, stats=stats)
    old_state_dict = unnormalize_module.state_dict()

    # Convert state dict
    new_state_dict = convert_normalize_to_buffer_state_dict(old_state_dict)

    # Create UnnormalizeBuffer module and load converted state dict
    buffer_module = UnnormalizeBuffer(features=features, norm_map=norm_map, stats=None)
    buffer_module.load_state_dict(new_state_dict)

    # Test that both modules produce the same output on normalized data
    batch = _random_batch(stats)

    # First normalize the batch
    normalize_module = Normalize(features=features, norm_map=norm_map, stats=stats)
    normalized_batch = normalize_module(batch)

    old_output = unnormalize_module(normalized_batch)
    new_output = buffer_module(normalized_batch)

    for key in old_output:
        torch.testing.assert_close(old_output[key], new_output[key])


def test_state_dict_conversion_key_format():
    """Test that conversion produces the expected key format."""
    from lerobot.common.policies.normalize import convert_normalize_to_buffer_state_dict

    # Mock state dict with the old format
    old_state_dict = {
        "buffer_observation_image.mean": torch.randn(3, 1, 1),
        "buffer_observation_image.std": torch.randn(3, 1, 1),
        "buffer_observation_state.min": torch.randn(5),
        "buffer_observation_state.max": torch.randn(5),
        "some_other_param": torch.randn(10),  # Non-normalization parameter
    }

    new_state_dict = convert_normalize_to_buffer_state_dict(old_state_dict)

    # Check expected key transformations
    expected_keys = {
        "observation_image_mean",
        "observation_image_std",
        "observation_state_min",
        "observation_state_max",
        "some_other_param",  # Should be unchanged
    }

    assert set(new_state_dict.keys()) == expected_keys

    # Check values are preserved
    torch.testing.assert_close(
        new_state_dict["observation_image_mean"], old_state_dict["buffer_observation_image.mean"]
    )
    torch.testing.assert_close(
        new_state_dict["observation_image_std"], old_state_dict["buffer_observation_image.std"]
    )
    torch.testing.assert_close(new_state_dict["some_other_param"], old_state_dict["some_other_param"])
