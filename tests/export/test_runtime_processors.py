#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import numpy as np
import pytest

from lerobot.export.manifest import ProcessorSpec
from lerobot.export.normalize import Normalizer, save_stats_safetensors
from lerobot.export.processors.runtime import build_processor_pipeline


def test_runtime_processor_pipeline_executes_normalize_and_denormalize(tmp_path) -> None:
    stats = {
        "observation.state": {
            "mean": np.array([1.0, 2.0], dtype=np.float32),
            "std": np.array([0.5, 2.0], dtype=np.float32),
        },
        "action": {
            "mean": np.array([3.0, 4.0], dtype=np.float32),
            "std": np.array([2.0, 0.5], dtype=np.float32),
        },
    }
    save_stats_safetensors(stats, tmp_path / "stats.safetensors")
    preprocessors = [
        ProcessorSpec(
            type="normalize",
            mode="mean_std",
            artifact="stats.safetensors",
            features=["observation.state"],
        )
    ]
    postprocessors = [
        ProcessorSpec(type="denormalize", mode="mean_std", artifact="stats.safetensors", features=["action"])
    ]
    normalizer = Normalizer.from_specs(preprocessors, postprocessors, tmp_path)

    pre, _ = build_processor_pipeline(preprocessors, package_path=tmp_path, normalizer=normalizer)
    post, _ = build_processor_pipeline(postprocessors, package_path=tmp_path, normalizer=normalizer)

    processed = pre({"observation.state": np.array([[1.5, 6.0]], dtype=np.float32)})
    np.testing.assert_allclose(processed["observation.state"], np.array([[1.0, 2.0]], dtype=np.float32))

    output = post({"action": np.array([[1.0, 2.0]], dtype=np.float32)})
    np.testing.assert_allclose(output["action"], np.array([[5.0, 5.0]], dtype=np.float32))


def test_runtime_processor_pipeline_executes_relative_and_absolute_actions(tmp_path) -> None:
    preprocessors = [
        ProcessorSpec(
            type="relative_actions",
            extra_params={
                "enabled": True,
                "exclude_joints": ["gripper"],
                "action_names": ["joint", "gripper"],
            },
        )
    ]
    postprocessors = [ProcessorSpec(type="absolute_actions", extra_params={"enabled": True})]

    pre, relative = build_processor_pipeline(preprocessors, package_path=tmp_path)
    post, _ = build_processor_pipeline(postprocessors, package_path=tmp_path, relative_processor=relative)

    raw = {
        "observation.state": np.array([[10.0, 20.0]], dtype=np.float32),
        "action": np.array([[[11.0, 25.0]]], dtype=np.float32),
    }
    processed = pre(raw)
    np.testing.assert_allclose(processed["action"], np.array([[[1.0, 25.0]]], dtype=np.float32))

    recovered = post({"action": processed["action"]})
    np.testing.assert_allclose(recovered["action"], raw["action"])


def test_runtime_processor_pipeline_fails_on_unknown_processor(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown export processor type"):
        build_processor_pipeline([ProcessorSpec(type="not_real")], package_path=tmp_path)


def test_runtime_processor_pipeline_executes_tokenize(tmp_path) -> None:
    pytest.importorskip("transformers")
    from tests.export.conftest import load_cached_paligemma_tokenizer

    tokenizer = load_cached_paligemma_tokenizer()
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    specs = [
        ProcessorSpec(
            type="tokenize",
            artifact="tokenizer",
            extra_params={
                "max_length": 8,
                "padding_side": "right",
                "padding": "max_length",
                "truncation": True,
            },
        )
    ]

    pipeline, _ = build_processor_pipeline(specs, package_path=tmp_path)
    output = pipeline({"task": "pick cube"})

    assert output["observation.language.tokens"].shape == (1, 8)
    assert output["observation.language.attention_mask"].shape == (1, 8)
    assert output["observation.language.tokens"].dtype == np.int64
    assert output["observation.language.attention_mask"].dtype == np.bool_
