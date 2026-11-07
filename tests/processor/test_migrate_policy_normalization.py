#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import dataclasses

import draccus
import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor.migrate_policy_normalization import _coerce_tuple_fields


def test_coerce_tuple_fields_converts_optional_tuple():
    config = DiffusionConfig()
    config.crop_shape = [84, 84]  # what json.load() produces for a JSON array

    _coerce_tuple_fields(config)

    assert config.crop_shape == (84, 84)
    assert isinstance(config.crop_shape, tuple)


def test_coerce_tuple_fields_converts_variadic_tuple():
    config = DiffusionConfig()
    config.down_dims = [512, 1024, 2048]

    _coerce_tuple_fields(config)

    assert config.down_dims == (512, 1024, 2048)
    assert isinstance(config.down_dims, tuple)


def test_coerce_tuple_fields_leaves_none_untouched():
    config = DiffusionConfig()
    assert config.crop_shape is None

    _coerce_tuple_fields(config)

    assert config.crop_shape is None


def test_coerce_tuple_fields_leaves_list_fields_as_lists():
    @dataclasses.dataclass
    class DummyConfig:
        crop_shape: tuple[int, int] | None = None
        tags: list[str] | None = None

    config = DummyConfig(crop_shape=[84, 84], tags=["a", "b"])

    _coerce_tuple_fields(config)

    assert config.crop_shape == (84, 84)
    assert isinstance(config.crop_shape, tuple)
    assert config.tags == ["a", "b"]
    assert isinstance(config.tags, list)


def test_uncoerced_list_fails_draccus_encode():
    """Without coercion, encoding a list against a declared tuple[...] type crashes.
    This is the bug migrate_policy_normalization.py hit when config.json was loaded
    via plain json.load()."""
    config = DiffusionConfig()
    config.crop_shape = [84, 84]

    with pytest.raises(Exception, match="Couldn't encode"):
        draccus.encode(config, PreTrainedConfig)


def test_coerced_config_saves_with_draccus():
    config = DiffusionConfig()
    config.crop_shape = [84, 84]
    config.down_dims = [512, 1024, 2048]

    _coerce_tuple_fields(config)
    encoded = draccus.encode(config, PreTrainedConfig)

    assert encoded["crop_shape"] == [84, 84]
    assert encoded["down_dims"] == [512, 1024, 2048]
