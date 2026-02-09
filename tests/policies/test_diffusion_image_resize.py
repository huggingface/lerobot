"""Tests for the resize_shape feature in Diffusion and VQ-BeT RGB encoders."""

import warnings

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTRgbEncoder
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def _make_diffusion_config(image_shape=(3, 96, 96), crop_shape=(84, 84), resize_shape=None):
    config = DiffusionConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.crop_shape = crop_shape
    config.resize_shape = resize_shape
    config.device = "cpu"
    return config


def _make_vqbet_config(image_shape=(3, 96, 96), crop_shape=(84, 84), resize_shape=None):
    config = VQBeTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.crop_shape = crop_shape
    config.resize_shape = resize_shape
    config.device = "cpu"
    return config


# ── Diffusion encoder tests ──────────────────────────────────────────


def test_diffusion_resize_before_crop():
    """High-res image with resize_shape should produce correct output."""
    config = _make_diffusion_config(image_shape=(3, 480, 640), crop_shape=(84, 84), resize_shape=(96, 96))
    config.validate_features()
    encoder = DiffusionRgbEncoder(config)
    encoder.eval()

    x = torch.rand(2, 3, 480, 640)
    out = encoder(x)
    assert out.shape[0] == 2
    assert out.ndim == 2


def test_diffusion_no_resize_backward_compat():
    """Default (no resize) still works on small images."""
    config = _make_diffusion_config(image_shape=(3, 96, 96), crop_shape=(84, 84), resize_shape=None)
    config.validate_features()
    encoder = DiffusionRgbEncoder(config)
    encoder.eval()

    x = torch.rand(2, 3, 96, 96)
    out = encoder(x)
    assert out.shape[0] == 2
    assert out.ndim == 2


def test_diffusion_validate_warns_large_image():
    """Warning when crop is much smaller than image and no resize_shape set."""
    config = _make_diffusion_config(image_shape=(3, 480, 640), crop_shape=(84, 84), resize_shape=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config.validate_features()
        assert len(w) == 1
        assert "resize_shape" in str(w[0].message)


def test_diffusion_validate_no_warn_with_resize():
    """No warning when resize_shape is set."""
    config = _make_diffusion_config(image_shape=(3, 480, 640), crop_shape=(84, 84), resize_shape=(96, 96))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config.validate_features()
        assert len(w) == 0


def test_diffusion_crop_larger_than_resize_raises():
    """crop_shape > resize_shape must raise ValueError."""
    config = _make_diffusion_config(image_shape=(3, 480, 640), crop_shape=(128, 128), resize_shape=(96, 96))
    with pytest.raises(ValueError, match="must fit within"):
        config.validate_features()


# ── VQ-BeT encoder tests ─────────────────────────────────────────────


def test_vqbet_resize_before_crop():
    """High-res image with resize_shape should produce correct output."""
    config = _make_vqbet_config(image_shape=(3, 480, 640), crop_shape=(84, 84), resize_shape=(96, 96))
    config.validate_features()
    encoder = VQBeTRgbEncoder(config)
    encoder.eval()

    x = torch.rand(2, 3, 480, 640)
    out = encoder(x)
    assert out.shape[0] == 2
    assert out.ndim == 2


def test_vqbet_no_resize_backward_compat():
    """Default (no resize) still works on small images."""
    config = _make_vqbet_config(image_shape=(3, 96, 96), crop_shape=(84, 84), resize_shape=None)
    config.validate_features()
    encoder = VQBeTRgbEncoder(config)
    encoder.eval()

    x = torch.rand(2, 3, 96, 96)
    out = encoder(x)
    assert out.shape[0] == 2
    assert out.ndim == 2


def test_vqbet_validate_warns_large_image():
    """Warning when crop is much smaller than image and no resize_shape set."""
    config = _make_vqbet_config(image_shape=(3, 480, 640), crop_shape=(84, 84), resize_shape=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config.validate_features()
        assert len(w) == 1
        assert "resize_shape" in str(w[0].message)


def test_vqbet_crop_larger_than_resize_raises():
    """crop_shape > resize_shape must raise ValueError."""
    config = _make_vqbet_config(image_shape=(3, 480, 640), crop_shape=(128, 128), resize_shape=(96, 96))
    with pytest.raises(ValueError, match="must fit within"):
        config.validate_features()
