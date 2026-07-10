#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``PreTrainedConfig.resolve_compile_mode``.

CUDAGraphs (used by 'max-autotune' and 'reduce-overhead') is incompatible with
gradient accumulation because multiple forward passes before backward overwrite
tensors captured in the graph. ``resolve_compile_mode`` is the single source of
truth that picks a safe mode per policy.
"""

import pytest

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

ALL_CONFIGS = [PI0Config, PI05Config, PI0FastConfig, SmolVLAConfig, DiffusionConfig]


# -- Defaults ---------------------------------------------------------------


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_compile_mode_field_defaults_to_none(config_cls):
    assert config_cls().compile_mode is None


@pytest.mark.parametrize(
    "config_cls,expected_default,expected_safe",
    [
        (PI0Config, "max-autotune", "max-autotune-no-cudagraphs"),
        (PI05Config, "max-autotune", "max-autotune-no-cudagraphs"),
        (PI0FastConfig, "max-autotune", "max-autotune-no-cudagraphs"),
        (SmolVLAConfig, "max-autotune", "max-autotune-no-cudagraphs"),
        (DiffusionConfig, "reduce-overhead", "default"),
    ],
)
def test_class_compile_mode_constants(config_cls, expected_default, expected_safe):
    assert config_cls.DEFAULT_COMPILE_MODE == expected_default
    assert config_cls.SAFE_COMPILE_MODE == expected_safe


# -- resolve_compile_mode: implicit (compile_mode=None) ---------------------


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_resolve_none_without_accumulation_returns_default(config_cls):
    config = config_cls(compile_model=True)
    assert config.resolve_compile_mode(gradient_accumulation_steps=1) == config.DEFAULT_COMPILE_MODE


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_resolve_none_with_accumulation_returns_safe(config_cls):
    config = config_cls(compile_model=True)
    assert config.resolve_compile_mode(gradient_accumulation_steps=4) == config.SAFE_COMPILE_MODE


def test_resolve_diffusion_safe_mode_is_default_not_max_autotune():
    """Diffusion's safe fallback must be 'default' (lightweight), not the heavy
    autotune-based fallback used by pi0/pi05."""
    config = DiffusionConfig(compile_model=True)
    assert config.resolve_compile_mode(gradient_accumulation_steps=2) == "default"


# -- resolve_compile_mode: explicit user values -----------------------------


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_resolve_explicit_safe_mode_passes_through(config_cls):
    config = config_cls(compile_model=True, compile_mode="max-autotune-no-cudagraphs")
    assert config.resolve_compile_mode(gradient_accumulation_steps=4) == "max-autotune-no-cudagraphs"


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_resolve_explicit_default_mode_passes_through(config_cls):
    config = config_cls(compile_model=True, compile_mode="default")
    assert config.resolve_compile_mode(gradient_accumulation_steps=4) == "default"


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
@pytest.mark.parametrize("cudagraphs_mode", ["max-autotune", "reduce-overhead"])
def test_resolve_explicit_cudagraphs_mode_with_accumulation_raises(config_cls, cudagraphs_mode):
    config = config_cls(compile_model=True, compile_mode=cudagraphs_mode)
    with pytest.raises(ValueError, match="CUDAGraphs"):
        config.resolve_compile_mode(gradient_accumulation_steps=2)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
@pytest.mark.parametrize("cudagraphs_mode", ["max-autotune", "reduce-overhead"])
def test_resolve_explicit_cudagraphs_mode_without_accumulation_passes(config_cls, cudagraphs_mode):
    config = config_cls(compile_model=True, compile_mode=cudagraphs_mode)
    assert config.resolve_compile_mode(gradient_accumulation_steps=1) == cudagraphs_mode
