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

import inspect
import json
import logging
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from draccus.utils import ParsingError
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.policies.groot.configuration_groot import (
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    GROOT_N1_5,
    GROOT_N1_5_REMOVAL_GUIDANCE,
    GROOT_N1_7,
    GROOT_N1_7_BASE_MODEL,
    GrootConfig,
    infer_groot_model_version,
    infer_groot_n1_7_action_execution_horizon,
    infer_groot_n1_7_action_horizon,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm,
    make_groot_pre_post_processors,
)
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _groot_features(
    state_dim: int, action_dim: int
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )


def _groot_config(model_version: str = GROOT_N1_7) -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    kwargs = {"action_decode_transform": GROOT_ACTION_DECODE_TRANSFORM_LIBERO}
    return GrootConfig(
        model_version=model_version,
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        **kwargs,
    )


def _raw_n1_7_libero_config(model_path) -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    return GrootConfig(
        model_version=GROOT_N1_7,
        base_model_path=str(model_path),
        embodiment_tag="libero_sim",
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    )


def test_n1_7_backbone_accepts_transformers_5_layout_and_forwards_mm_token_type_ids(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.feature_extraction_utils import BatchFeature

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.visual = nn.Linear(1, 1)

    class FakeQwen3VLForConditionalGeneration(nn.Module):
        config = SimpleNamespace(image_token_id=42, video_token_id=43)

        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()
            self.forward_kwargs = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        @classmethod
        def _from_config(cls, *args, **kwargs):
            return cls()

        def eval(self):
            super().eval()
            return self

        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            assert "mm_token_type_ids" in kwargs
            batch_size, sequence_length = kwargs["input_ids"].shape
            features = torch.arange(batch_size * sequence_length * 4, dtype=torch.float32).view(
                batch_size, sequence_length, 4
            )
            return SimpleNamespace(hidden_states=[features])

    monkeypatch.setattr(
        groot_n1_7,
        "metadata",
        SimpleNamespace(version=lambda package: "5.3.0" if package == "transformers" else "0"),
        raising=False,
    )
    monkeypatch.setattr(groot_n1_7, "Qwen3VLForConditionalGeneration", FakeQwen3VLForConditionalGeneration)

    backbone = groot_n1_7.Qwen3Backbone(
        model_name="nvidia/Cosmos-Reason2-2B",
        select_layer=1,
        use_flash_attention=False,
    )

    assert len(backbone.language_model.layers) == 1
    output = backbone.forward(
        BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 42, 2]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
                "pixel_values": torch.zeros(1, 3, 2, 2),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        )
    )

    assert backbone.model.forward_kwargs["mm_token_type_ids"].tolist() == [[0, 1, 0]]
    assert output["backbone_features"].shape == (1, 3, 4)

    output = backbone.forward(
        BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 42, 43, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "pixel_values": torch.zeros(1, 3, 2, 2),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
                "pixel_values_videos": torch.zeros(1, 3, 2, 2),
                "video_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        )
    )

    assert backbone.model.forward_kwargs["mm_token_type_ids"].tolist() == [[0, 1, 2, 0]]
    assert backbone.model.forward_kwargs["mm_token_type_ids"].dtype == torch.int32
    assert output["backbone_features"].shape == (1, 4, 4)


def test_n1_7_backbone_preserves_missing_qwen_optional_dependency_error(monkeypatch):
    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    monkeypatch.setattr(
        groot_n1_7,
        "metadata",
        SimpleNamespace(version=lambda package: "5.3.0" if package == "transformers" else "0"),
        raising=False,
    )
    monkeypatch.setattr(groot_n1_7, "Qwen3VLForConditionalGeneration", None)

    with pytest.raises(ImportError, match="Qwen3VLForConditionalGeneration is required"):
        groot_n1_7.Qwen3Backbone(
            model_name="nvidia/Cosmos-Reason2-2B",
            select_layer=0,
            use_flash_attention=False,
        )


def _write_raw_n1_7_libero_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "architectures": ["Gr00tN1d7"],
                "model_name": "nvidia/Cosmos-Reason2-2B",
                "action_horizon": 40,
                "max_state_dim": 132,
                "max_action_dim": 132,
                "image_target_size": [256, 256],
            }
        )
    )
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {
                    "clip_outliers": True,
                    "formalize_language": True,
                    "image_crop_size": [230, 230],
                    "image_target_size": [256, 256],
                    "shortest_image_edge": 256,
                    "crop_fraction": 0.95,
                    "use_albumentations": True,
                    "max_action_horizon": 40,
                    "max_state_dim": 132,
                    "max_action_dim": 132,
                    "use_percentiles": True,
                    "use_relative_action": True,
                    "modality_configs": {
                        "libero_sim": {
                            "video": {
                                "delta_indices": [0],
                                "modality_keys": ["image", "wrist_image"],
                            },
                            "state": {
                                "delta_indices": [0],
                                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                            },
                            "action": {
                                "delta_indices": list(range(16)),
                                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                            },
                            "language": {
                                "delta_indices": [0],
                                "modality_keys": ["annotation.human.action.task_description"],
                            },
                        }
                    },
                },
            }
        )
    )
    (path / "embodiment_id.json").write_text(json.dumps({"libero_sim": 42}))
    (path / "statistics.json").write_text(
        json.dumps(
            {
                "libero_sim": {
                    "state": {
                        "x": _stats([0.0]),
                        "y": _stats([1.0]),
                        "z": _stats([2.0]),
                        "roll": _stats([3.0]),
                        "pitch": _stats([4.0]),
                        "yaw": _stats([5.0]),
                        "gripper": _stats([6.0, 7.0]),
                    },
                    "action": {
                        "x": _stats([10.0]),
                        "y": _stats([11.0]),
                        "z": _stats([12.0]),
                        "roll": _stats([13.0]),
                        "pitch": _stats([14.0]),
                        "yaw": _stats([15.0]),
                        "gripper": _stats([16.0]),
                    },
                    "relative_action": {},
                }
            }
        )
    )


def _stats(values):
    return {
        "min": values,
        "max": [value + 100.0 for value in values],
        "mean": [value + 50.0 for value in values],
        "std": [1.0 for _ in values],
        "q01": [value + 1.0 for value in values],
        "q99": [value + 99.0 for value in values],
    }


def _expected_albumentations_eval_image(image_np, cv2, *, target_size, shortest_edge, crop_fraction):
    height, width = image_np.shape[:2]
    if height != width:
        square_edge = max(height, width)
        pad_h = square_edge - height
        pad_w = square_edge - width
        image_np = cv2.copyMakeBorder(
            image_np,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    image_np = cv2.resize(image_np, (shortest_edge, shortest_edge), interpolation=cv2.INTER_AREA)
    crop_h = max(1, int(shortest_edge * crop_fraction))
    crop_w = max(1, int(shortest_edge * crop_fraction))
    top = (shortest_edge - crop_h) // 2
    left = (shortest_edge - crop_w) // 2
    image_np = image_np[top : top + crop_h, left : left + crop_w]
    return cv2.resize(image_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


class _DummyGrootModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        # Like the real GR00TN17, the dummy defines no compute_dtype attribute:
        # GrootPolicy only sets it when use_bf16 is enabled.
        self.config = SimpleNamespace()
        self.forward_inputs = None
        self.get_action_options = None

    def forward(self, inputs):
        self.forward_inputs = dict(inputs)
        return {"loss": self.weight + 1.0}

    def get_action(self, inputs, options=None):
        self.forward_inputs = dict(inputs)
        self.get_action_options = options
        batch_size = inputs["state"].shape[0]
        return {"action_pred": torch.zeros(batch_size, 40, 132, device=self.weight.device)}


def test_groot_defaults_use_n1_7():
    config = GrootConfig(device="cpu")

    assert config.model_version == GROOT_N1_7
    assert config.base_model_path == GROOT_N1_7_BASE_MODEL
    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.chunk_size == 40
    assert config.n_action_steps == 40
    assert len(config.action_delta_indices) == 40


def test_groot_n1_7_accepts_named_action_decode_transform():
    config = GrootConfig(
        model_version=GROOT_N1_7,
        action_decode_transform="libero",
        device="cpu",
    )

    assert config.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO


@pytest.mark.parametrize("legacy_transform", ["libero_gripper", "libero-gripper"])
def test_groot_n1_7_rejects_legacy_libero_gripper_action_decode_transform(legacy_transform):
    with pytest.raises(ValueError, match="Unsupported GR00T N1.7 action decode transform"):
        GrootConfig(
            model_version=GROOT_N1_7,
            action_decode_transform=legacy_transform,
            device="cpu",
        )


@pytest.mark.parametrize("legacy_version", ["n1.5", "n1_5", "n1d5", "n15", "1.5"])
def test_groot_rejects_n1_5_aliases_with_removal_guidance(legacy_version):
    with pytest.raises(ValueError, match="Unsupported GR00T model_version") as exc_info:
        GrootConfig(model_version=legacy_version, device="cpu")

    assert GROOT_N1_5_REMOVAL_GUIDANCE in str(exc_info.value)


def test_groot_rejected_non_n1_5_version_omits_removal_guidance():
    with pytest.raises(ValueError, match="Unsupported GR00T model_version") as exc_info:
        GrootConfig(model_version="n2.0", device="cpu")

    assert GROOT_N1_5_REMOVAL_GUIDANCE not in str(exc_info.value)


def test_groot_config_rejects_mismatched_n1_5_path_for_n1_7():
    with pytest.raises(ValueError, match="does not match base_model_path") as exc_info:
        GrootConfig(
            model_version=GROOT_N1_7,
            base_model_path="nvidia/GR00T-N1.5-3B",
            device="cpu",
        )

    assert GROOT_N1_5_REMOVAL_GUIDANCE in str(exc_info.value)


def test_groot_n1_7_can_be_selected_from_policy_config_factory_without_external_gr00t():
    sys.modules.pop("gr00t", None)

    config = make_policy_config("groot", model_version=GROOT_N1_7, device="cpu")

    assert isinstance(config, GrootConfig)
    assert config.model_version == GROOT_N1_7
    assert "gr00t" not in sys.modules


def test_groot_predict_action_chunk_accepts_rtc_kwargs():
    signature = inspect.signature(GrootPolicy.predict_action_chunk)

    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    signature.bind(object(), {}, inference_delay=2, prev_chunk_left_over=None)


def test_groot_predict_action_chunk_forwards_n1_7_rtc_prefix(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    dummy_model = _DummyGrootModel()
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: dummy_model))
    config = _groot_config(GROOT_N1_7)
    policy = GrootPolicy(config)
    policy.config.rtc_config = SimpleNamespace(execution_horizon=6)

    prev_chunk = torch.arange(8 * 7, dtype=torch.float32).view(8, 7)

    actions = policy.predict_action_chunk(
        {"state": torch.zeros(1, 1, 132)},
        inference_delay=3,
        prev_chunk_left_over=prev_chunk,
    )

    assert actions.shape == (1, 40, 7)
    assert dummy_model.get_action_options == {
        "action_horizon": 8,
        "rtc_overlap_steps": 6,
        "rtc_frozen_steps": 3,
        "rtc_ramp_rate": 6.0,
    }
    assert dummy_model.forward_inputs["action"].shape == (1, 8, 132)
    torch.testing.assert_close(dummy_model.forward_inputs["action"][0, :, :7], prev_chunk)
    torch.testing.assert_close(dummy_model.forward_inputs["action"][0, :, 7:], torch.zeros(8, 125))


def test_groot_predict_action_chunk_strips_padded_n1_7_rtc_prefix(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    dummy_model = _DummyGrootModel()
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: dummy_model))
    config = _groot_config(GROOT_N1_7)
    policy = GrootPolicy(config)
    policy.config.rtc_config = SimpleNamespace(execution_horizon=6)

    prev_chunk = torch.cat(
        (
            torch.arange(4 * 7, dtype=torch.float32).view(4, 7) + 1.0,
            torch.zeros(2, 7),
        )
    )

    policy.predict_action_chunk(
        {"state": torch.zeros(1, 1, 132)},
        inference_delay=5,
        prev_chunk_left_over=prev_chunk,
    )

    assert dummy_model.get_action_options == {
        "action_horizon": 4,
        "rtc_overlap_steps": 4,
        "rtc_frozen_steps": 4,
        "rtc_ramp_rate": 6.0,
    }
    assert dummy_model.forward_inputs["action"].shape == (1, 4, 132)
    torch.testing.assert_close(dummy_model.forward_inputs["action"][0, :, :7], prev_chunk[:4])
    torch.testing.assert_close(dummy_model.forward_inputs["action"][0, :, 7:], torch.zeros(4, 125))


def test_groot_n1_7_predict_action_chunk_truncates_to_checkpoint_valid_horizon(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)

    class HorizonModel(_DummyGrootModel):
        def get_action(self, inputs, options=None):
            del options
            batch_size = inputs["state"].shape[0]
            steps = torch.arange(40, dtype=torch.float32).view(1, 40, 1).expand(batch_size, 40, 132)
            return {"action_pred": steps}

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: HorizonModel()))
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    config = GrootConfig(
        model_version=GROOT_N1_7,
        base_model_path=str(model_path),
        embodiment_tag="libero_sim",
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        chunk_size=40,
        n_action_steps=40,
    )
    policy = GrootPolicy(config)

    actions = policy.predict_action_chunk({"state": torch.zeros(1, 1, 132)})

    assert actions.shape == (1, 16, 7)
    torch.testing.assert_close(actions[0, :, 0], torch.arange(16, dtype=torch.float32))


def _write_n1_5_marked_checkpoint(model_path):
    """Write a generically named local dir whose config.json carries N1.5 content markers."""
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "gr00t_n1_5", "architectures": ["GR00T_N1_5"]})
    )


def test_groot_from_pretrained_rejects_n1_5_checkpoint_against_n1_7_caller_config(tmp_path):
    model_path = tmp_path / "local-checkpoint"
    _write_n1_5_marked_checkpoint(model_path)
    config = _groot_config(GROOT_N1_7)

    # The caller config is valid on its own; from_pretrained overrides its
    # base_model_path with the pretrained path, detects the N1.5 checkpoint from
    # the local config.json content, and must reject the mismatch before any
    # model weights are loaded.
    with pytest.raises(ValueError, match="does not match base_model_path"):
        GrootPolicy.from_pretrained(model_path, config=config)


def test_groot_from_pretrained_keeps_matching_caller_config(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "GR00T-N1.7-local"
    model_path.mkdir()
    config = _groot_config(GROOT_N1_7)

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path, config=config)

    assert policy.config.model_version == GROOT_N1_7
    assert policy.config.base_model_path == str(model_path)


def test_groot_from_pretrained_infers_n1_7_from_ambiguous_local_config(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "local-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "Gr00tN1d7"}')

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path)

    assert policy.config.model_version == GROOT_N1_7
    assert policy.config.base_model_path == str(model_path)


def test_raw_n1_7_libero_checkpoint_processors_use_checkpoint_assets(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)

    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))

    pack_inputs = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    vlm_encode = next(step for step in preprocessor.steps if isinstance(step, GrootN17VLMEncodeStep))
    decode_actions = next(step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep))

    assert pack_inputs.embodiment_tag == "libero_sim"
    assert pack_inputs.embodiment_mapping["libero_sim"] == 42
    assert pack_inputs.formalize_language is True
    assert pack_inputs.valid_action_horizon == 16
    assert pack_inputs.action_horizon == 40
    assert pack_inputs.max_state_dim == 132
    assert pack_inputs.max_action_dim == 132
    assert pack_inputs.clip_outliers is True
    assert pack_inputs.video_modality_keys == ["image", "wrist_image"]
    assert pack_inputs.stats[OBS_STATE]["min"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert pack_inputs.stats[OBS_STATE]["max"] == [
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
    ]
    assert pack_inputs.stats[ACTION]["min"] == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    assert vlm_encode.image_crop_size == [230, 230]
    assert vlm_encode.image_target_size == [256, 256]
    assert vlm_encode.shortest_image_edge == 256
    assert vlm_encode.crop_fraction == 0.95
    assert vlm_encode.use_albumentations is True
    assert decode_actions.raw_stats["action"]["gripper"]["q99"] == [115.0]
    assert decode_actions.env_action_dim == 7
    assert decode_actions.use_percentiles is True
    assert decode_actions.use_relative_action is True
    assert decode_actions.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO


def test_raw_n1_7_checkpoint_requires_percentile_stats_when_config_uses_percentiles(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    statistics = json.loads((model_path / "statistics.json").read_text())
    del statistics["libero_sim"]["state"]["x"]["q01"]
    (model_path / "statistics.json").write_text(json.dumps(statistics))
    config = _raw_n1_7_libero_config(model_path)

    with pytest.raises(KeyError, match="q01.*state.x"):
        make_pre_post_processors(config, pretrained_path=str(model_path))


def test_raw_n1_7_checkpoint_processors_prefer_checkpoint_stats_when_dataset_stats_supplied(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    dataset_stats = {
        OBS_STATE: {
            "min": torch.full((8,), -8.0),
            "max": torch.full((8,), 8.0),
        },
        ACTION: {
            "min": torch.full((7,), -7.0),
            "max": torch.full((7,), 7.0),
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_path),
        dataset_stats=dataset_stats,
    )

    pack_inputs = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    decode_actions = next(step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep))
    torch.testing.assert_close(
        torch.as_tensor(pack_inputs.stats[OBS_STATE]["min"]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    torch.testing.assert_close(
        torch.as_tensor(pack_inputs.stats[ACTION]["max"]),
        torch.tensor([109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]),
    )
    assert decode_actions.raw_stats["action"]["gripper"]["q99"] == [115.0]
    assert decode_actions.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO


def test_groot_n1_7_saved_processors_round_trip_checkpoint_specific_fields(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "saved_processors"

    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        save_dir,
        config_filename="policy_preprocessor.json",
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        save_dir,
        config_filename="policy_postprocessor.json",
    )
    pack_inputs = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    decode_actions = next(
        step for step in loaded_postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)
    )

    assert pack_inputs.valid_action_horizon == 16
    assert pack_inputs.action_horizon == 40
    assert pack_inputs.video_modality_keys == ["image", "wrist_image"]
    assert pack_inputs.clip_outliers is True
    torch.testing.assert_close(
        pack_inputs.stats[OBS_STATE]["min"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    assert decode_actions.env_action_dim == 7
    assert decode_actions.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO
    assert decode_actions.raw_stats["action"]["gripper"]["q99"] == [115.0]


def test_converted_raw_n1_7_processors_load_without_legacy_action_unpack_override(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "converted_pretrained_model"

    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(save_dir),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    assert any(isinstance(step, GrootN17PackInputsStep) for step in loaded_preprocessor.steps)
    assert any(isinstance(step, GrootN17ActionDecodeStep) for step in loaded_postprocessor.steps)


def test_converted_raw_n1_7_absolute_action_processors_load_without_relative_steps(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "absolute_pretrained_model"

    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(save_dir),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    assert any(isinstance(step, GrootN17PackInputsStep) for step in loaded_preprocessor.steps)
    assert any(isinstance(step, GrootN17ActionDecodeStep) for step in loaded_postprocessor.steps)
    assert not any(isinstance(step, RelativeActionsProcessorStep) for step in loaded_preprocessor.steps)
    assert not any(isinstance(step, AbsoluteActionsProcessorStep) for step in loaded_postprocessor.steps)


def test_converted_raw_n1_7_relative_action_processors_reconnect_after_load(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "relative_pretrained_model"
    action_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    preprocessor_config_path = save_dir / "policy_preprocessor.json"
    preprocessor_config = json.loads(preprocessor_config_path.read_text())
    preprocessor_config["steps"].insert(
        2,
        {
            "registry_name": "relative_actions_processor",
            "config": {
                "enabled": True,
                "exclude_joints": ["gripper"],
                "action_names": action_names,
            },
        },
    )
    preprocessor_config_path.write_text(json.dumps(preprocessor_config, indent=4) + "\n")

    postprocessor_config_path = save_dir / "policy_postprocessor.json"
    postprocessor_config = json.loads(postprocessor_config_path.read_text())
    postprocessor_config["steps"].insert(
        -1,
        {
            "registry_name": "absolute_actions_processor",
            "config": {"enabled": True},
        },
    )
    postprocessor_config_path.write_text(json.dumps(postprocessor_config, indent=4) + "\n")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(save_dir),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    relative_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )

    assert relative_step.enabled is True
    assert relative_step.exclude_joints == ["gripper"]
    assert relative_step.action_names == action_names
    assert absolute_step.relative_step is relative_step


def test_groot_n1_7_pack_inputs_rejects_state_dim_above_core_max():
    step = GrootN17PackInputsStep(
        max_state_dim=2,
        max_action_dim=4,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(1, 3),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    with pytest.raises(ValueError, match="State dimension 3 exceeds max_state_dim 2"):
        step(transition)


def test_groot_n1_7_pack_inputs_rejects_action_shape_above_core_limits():
    step = GrootN17PackInputsStep(
        action_horizon=2,
        max_state_dim=2,
        max_action_dim=2,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(1, 2),
        },
        TransitionKey.ACTION: torch.zeros(1, 2, 3),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    with pytest.raises(ValueError, match="Action dimension 3 exceeds max_action_dim 2"):
        step(transition)

    transition[TransitionKey.ACTION] = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="Action horizon 3 exceeds action_horizon 2"):
        step(transition)


def test_groot_n1_7_pack_inputs_clips_and_masks_only_valid_action_horizon():
    step = GrootN17PackInputsStep(
        action_horizon=40,
        valid_action_horizon=16,
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=True,
        clip_outliers=True,
        stats={
            OBS_STATE: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
            ACTION: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[2.0, -1.0]]),
        },
        TransitionKey.ACTION: torch.full((1, 16, 2), 1.0),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    torch.testing.assert_close(
        output[TransitionKey.OBSERVATION]["state"][0, 0, :2],
        torch.tensor([1.0, -1.0]),
    )
    assert output[TransitionKey.ACTION].shape == (1, 40, 4)
    torch.testing.assert_close(output[TransitionKey.ACTION][0, 16:], torch.zeros(24, 4))
    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (1, 40, 4)
    assert action_mask[0, :16, :2].sum().item() == 32
    assert action_mask[0, 16:].sum().item() == 0
    assert action_mask[0, :, 2:].sum().item() == 0


def test_groot_n1_7_pack_inputs_normalizes_state_with_q01_q99_clips_and_pads():
    step = GrootN17PackInputsStep(
        action_horizon=4,
        max_state_dim=6,
        max_action_dim=7,
        normalize_min_max=True,
        clip_outliers=True,
        stats={
            OBS_STATE: {
                "min": [0.0, 10.0, -2.0, 4.0],
                "max": [10.0, 10.0, 2.0, 8.0],
            }
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[5.0, 42.0, -6.0, 10.0]]),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    expected = torch.tensor([[[0.0, 0.0, -1.0, 1.0, 0.0, 0.0]]])
    torch.testing.assert_close(output[TransitionKey.OBSERVATION]["state"], expected)


def test_groot_n1_7_libero_open_gripper_state_normalizes_near_core_oracle():
    step = GrootN17PackInputsStep(
        action_horizon=40,
        max_state_dim=132,
        max_action_dim=7,
        normalize_min_max=True,
        clip_outliers=True,
        stats={
            OBS_STATE: {
                "min": [
                    -0.27276572585105896,
                    -0.237214133143425,
                    0.916006326675415,
                    2.779496669769287,
                    -1.3187512159347534,
                    -0.4198998212814331,
                    0.001503719249740243,
                    -0.03989770635962486,
                ],
                "max": [
                    0.1352936029434204,
                    0.362916499376297,
                    1.286232590675354,
                    3.2829697132110596,
                    0.9332759976387024,
                    0.6325722336769104,
                    0.03993396461009979,
                    -0.0016719202976673841,
                ],
            }
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor(
                [
                    [
                        -0.20846466720104218,
                        0.0,
                        1.1732795238494873,
                        3.1403393745422363,
                        0.0007735038525424898,
                        -0.0892220064997673,
                        0.020833000540733337,
                        -0.020833000540733337,
                    ]
                ]
            ),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    normalized = output[TransitionKey.OBSERVATION]["state"][0, 0, :8]
    expected = torch.tensor(
        [
            -0.6848445534706116,
            -0.2094583511352539,
            0.3898160457611084,
            0.4334142208099365,
            0.17185509204864502,
            -0.3716168999671936,
            0.005941033363342285,
            -0.002521216869354248,
        ]
    )
    torch.testing.assert_close(normalized, expected, atol=1e-6, rtol=1e-6)
    assert normalized[6:].abs().max().item() < 0.01


def test_groot_n1_7_pack_inputs_normalizes_action_chunk_per_dimension_before_padding():
    step = GrootN17PackInputsStep(
        action_horizon=5,
        valid_action_horizon=3,
        max_state_dim=4,
        max_action_dim=5,
        normalize_min_max=True,
        clip_outliers=True,
        stats={
            OBS_STATE: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
            ACTION: {
                "min": [-2.0, 10.0, 100.0],
                "max": [2.0, 30.0, 101.0],
            },
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[0.5, 0.5]]),
        },
        TransitionKey.ACTION: torch.tensor(
            [
                [
                    [-2.0, 30.0, 100.25],
                    [0.0, 20.0, 101.0],
                    [2.0, 10.0, 100.0],
                ]
            ]
        ),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    expected_actions = torch.tensor(
        [
            [
                [-1.0, 1.0, -0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(output[TransitionKey.ACTION], expected_actions)
    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (1, 5, 5)
    assert action_mask[0, :3, :3].sum().item() == 9
    assert action_mask[0, 3:].sum().item() == 0
    assert action_mask[0, :, 3:].sum().item() == 0


def test_groot_n1_7_pack_inputs_adds_inference_action_horizon_mask():
    step = GrootN17PackInputsStep(
        action_horizon=40,
        valid_action_horizon=16,
        max_state_dim=8,
        max_action_dim=7,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(2, 8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move", "Place"]},
    }

    output = step(transition)

    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (2, 40)
    assert action_mask[:, :16].sum().item() == 32
    assert action_mask[:, 16:].sum().item() == 0
    assert output[TransitionKey.COMPLEMENTARY_DATA]["embodiment_id"].dtype == torch.int32


def test_groot_n1_7_pack_inputs_orders_video_by_checkpoint_modality_keys():
    step = GrootN17PackInputsStep(
        normalize_min_max=False,
        video_modality_keys=["image", "wrist_image"],
    )
    transition = {
        TransitionKey.OBSERVATION: {
            f"{OBS_IMAGES}.zz_extra": torch.full((1, 3, 2, 2), 33, dtype=torch.uint8),
            f"{OBS_IMAGES}.image2": torch.full((1, 3, 2, 2), 22, dtype=torch.uint8),
            f"{OBS_IMAGES}.image": torch.full((1, 3, 2, 2), 11, dtype=torch.uint8),
            OBS_STATE: torch.zeros(1, 8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    video = output[TransitionKey.OBSERVATION]["video"]
    assert video.shape == (1, 1, 2, 2, 2, 3)
    assert np.unique(video[0, 0, 0]).tolist() == [11]
    assert np.unique(video[0, 0, 1]).tolist() == [22]
    assert f"{OBS_IMAGES}.zz_extra" not in output[TransitionKey.OBSERVATION]
    assert f"{OBS_IMAGES}.image" not in output[TransitionKey.OBSERVATION]
    assert f"{OBS_IMAGES}.image2" not in output[TransitionKey.OBSERVATION]


def test_groot_n1_7_postprocessor_clips_normalized_action_before_unnormalizing():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=3,
        normalize_min_max=True,
        clip_normalized_action=True,
        stats={
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [10.0, 10.0, 10.0],
            }
        },
    )
    transition = {
        TransitionKey.ACTION: torch.tensor([[-2.0, 0.0, 2.0]]),
    }

    output = step(transition)

    torch.testing.assert_close(output[TransitionKey.ACTION], torch.tensor([[0.0, 5.0, 10.0]]))


def test_groot_n1_7_action_decode_applies_named_libero_transform_from_modality_key():
    unit_stats = {
        "min": [0.0],
        "max": [1.0],
        "mean": [0.5],
        "std": [1.0],
        "q01": [0.0],
        "q99": [1.0],
    }
    step = GrootN17ActionDecodeStep(
        env_action_dim=3,
        raw_stats={
            "action": {
                "x": unit_stats,
                "gripper": unit_stats,
                "y": unit_stats,
            }
        },
        modality_config={
            "action": {
                "modality_keys": ["x", "gripper", "y"],
                "action_configs": [{}, {}, {}],
            }
        },
        action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    )
    action = torch.tensor(
        [
            [
                [-1.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )

    output = step({TransitionKey.ACTION: action})

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0],
                [1.0, -0.0, 0.0],
                [0.5, -1.0, 0.5],
            ]
        ]
    )
    torch.testing.assert_close(output[TransitionKey.ACTION], expected)


def test_groot_n1_7_action_decode_truncates_to_valid_horizon_for_relative_stats():
    arm_min = [[-1.0] * 5 for _ in range(16)]
    arm_max = [[1.0] * 5 for _ in range(16)]
    raw_stats = {
        "state": {
            "single_arm": _stats([0.0] * 5),
            "gripper": _stats([0.0]),
        },
        "action": {
            "single_arm": _stats([0.0] * 5),
            "gripper": {
                "min": [0.0],
                "max": [10.0],
                "mean": [5.0],
                "std": [1.0],
                "q01": [0.0],
                "q99": [10.0],
            },
        },
        "relative_action": {
            "single_arm": {
                "min": arm_min,
                "max": arm_max,
                "mean": [[0.0] * 5 for _ in range(16)],
                "std": [[1.0] * 5 for _ in range(16)],
                "q01": arm_min,
                "q99": arm_max,
            },
        },
    }
    modality_config = {
        "state": {
            "modality_keys": ["single_arm", "gripper"],
        },
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
            ],
        },
    }
    pack_step = GrootN17PackInputsStep(
        raw_stats=raw_stats,
        modality_config=modality_config,
        normalize_min_max=False,
    )
    pack_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 6)},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=6,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
        pack_step=pack_step,
    )

    output = decode_step({TransitionKey.ACTION: torch.zeros(1, 40, 6)})

    decoded = output[TransitionKey.ACTION]
    assert decoded.shape == (1, 16, 6)
    torch.testing.assert_close(decoded[..., :5], torch.zeros(1, 16, 5))
    torch.testing.assert_close(decoded[..., 5], torch.full((1, 16), 5.0))


def test_groot_n1_7_action_decode_requires_gripper_key_for_libero_transform():
    step = GrootN17ActionDecodeStep(
        env_action_dim=1,
        raw_stats={
            "action": {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                },
            }
        },
        modality_config={
            "action": {
                "modality_keys": ["x"],
                "action_configs": [{}],
            }
        },
        action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    )

    with pytest.raises(KeyError, match="gripper"):
        step({TransitionKey.ACTION: torch.zeros(1, 1, 1)})


def test_groot_n1_7_postprocessor_converts_libero_gripper_convention():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=7,
        normalize_min_max=True,
        stats={
            ACTION: {
                "min": [0.0] * 7,
                "max": [1.0] * 7,
            }
        },
        libero_gripper_action=True,
    )
    transition = {
        TransitionKey.ACTION: torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
    }

    output = step(transition)

    torch.testing.assert_close(output[TransitionKey.ACTION][:, -1], torch.tensor([1.0, -1.0]))


def test_groot_n1_7_postprocessor_decodes_selected_action_and_gripper_thresholds():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=7,
        normalize_min_max=True,
        clip_normalized_action=True,
        stats={
            ACTION: {
                "min": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 0.0],
                "max": [2.0, 14.0, 26.0, 38.0, 50.0, 62.0, 1.0],
            }
        },
        libero_gripper_action=True,
    )
    selected_actions = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, -0.5],
            [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.0],
            [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.5],
        ]
    )

    output = step({TransitionKey.ACTION: selected_actions})

    expected_prefix = torch.tensor([0.0, 11.0, 23.0, 36.0, 50.0, 62.0])
    torch.testing.assert_close(output[TransitionKey.ACTION][:, :6], expected_prefix.expand(3, 6))
    torch.testing.assert_close(output[TransitionKey.ACTION][:, -1], torch.tensor([1.0, -0.0, -1.0]))


def test_groot_n1_7_postprocessor_decodes_action_chunks_without_dropping_timesteps():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=7,
        normalize_min_max=True,
        clip_normalized_action=True,
        stats={
            ACTION: {
                "min": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 0.0],
                "max": [2.0, 14.0, 26.0, 38.0, 50.0, 62.0, 1.0],
            }
        },
        libero_gripper_action=True,
    )
    action_chunk = torch.tensor(
        [
            [
                [-1.0, 0.0, 1.0, -0.5, 0.5, 2.0, -1.0, 99.0],
                [0.25, -0.25, 0.75, -0.75, 1.0, -1.0, 0.0, 99.0],
                [1.0, -1.0, 0.0, 0.5, -0.5, 0.0, 0.5, 99.0],
            ]
        ]
    )

    output = step({TransitionKey.ACTION: action_chunk})

    expected_prefix = torch.tensor(
        [
            [
                [0.0, 12.0, 26.0, 32.0, 47.5, 62.0],
                [1.25, 11.5, 25.25, 31.0, 50.0, 50.0],
                [2.0, 10.0, 23.0, 36.0, 42.5, 56.0],
            ]
        ]
    )
    assert output[TransitionKey.ACTION].shape == (1, 3, 7)
    torch.testing.assert_close(output[TransitionKey.ACTION][..., :6], expected_prefix)
    torch.testing.assert_close(output[TransitionKey.ACTION][..., -1], torch.tensor([[1.0, -0.0, -1.0]]))


def test_groot_from_pretrained_rejects_n1_5_checkpoint_without_caller_config(tmp_path):
    model_path = tmp_path / "local-checkpoint"
    _write_n1_5_marked_checkpoint(model_path)

    # Without a caller config, from_pretrained infers the model version from the
    # local config.json content ('n1.5') and must fail with the removal guidance
    # instead of silently treating the N1.5 checkpoint as N1.7.
    with pytest.raises(ValueError, match="Unsupported GR00T model_version") as exc_info:
        GrootPolicy.from_pretrained(model_path)

    assert GROOT_N1_5_REMOVAL_GUIDANCE in str(exc_info.value)


def test_groot_n1_7_processors_are_registered_lazily_without_external_gr00t():
    sys.modules.pop("gr00t", None)
    config = _groot_config(GROOT_N1_7)

    preprocessor, _ = make_groot_pre_post_processors(config)
    step_types = {type(step) for step in preprocessor.steps}

    assert GrootN17PackInputsStep in step_types
    assert GrootN17VLMEncodeStep in step_types
    assert "gr00t" not in sys.modules


def test_groot_n1_7_pack_inputs_preserves_per_sample_language():
    step = GrootN17PackInputsStep(
        action_horizon=2,
        max_state_dim=4,
        max_action_dim=3,
        formalize_language=True,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        TransitionKey.ACTION: torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Pick Red Block!", "Place Blue Cube."],
        },
    }

    output = step(transition)

    assert output[TransitionKey.COMPLEMENTARY_DATA]["language"] == [
        "pick red block",
        "place blue cube",
    ]
    torch.testing.assert_close(
        output[TransitionKey.OBSERVATION]["state"][:, 0, :2],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_groot_n1_7_language_formalization_preserves_core_task_identifier_and_batch():
    step = GrootN17PackInputsStep(
        action_horizon=2,
        max_state_dim=8,
        max_action_dim=7,
        formalize_language=True,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(2, 8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": [
                "Pick_Up_The_Black_Bowl_Next_To_The_Ramekin_And_Place_It_On_The_Plate!!!",
                "MOVE, the YELLOW mug -- to Zone_2.",
            ],
        },
    }

    output = step(transition)

    assert output[TransitionKey.COMPLEMENTARY_DATA]["language"] == [
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "move the yellow mug  to zone_2",
    ]


def test_groot_n1_7_vlm_encode_uses_per_sample_language():
    class FakeProcessor:
        def __init__(self):
            self.rendered_texts = []
            self.encoded_texts = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            text = conversation[0]["content"][-1]["text"]
            self.rendered_texts.append(text)
            return f"rendered:{text}"

        def __call__(self, text, images, return_tensors, padding):
            self.encoded_texts = text
            return {
                "input_ids": torch.arange(len(text)).view(len(text), 1),
                "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
            }

    fake_proc = FakeProcessor()
    step = GrootN17VLMEncodeStep()
    step._proc = fake_proc
    transition = {
        TransitionKey.OBSERVATION: {
            "video": np.zeros((2, 1, 1, 2, 2, 3), dtype=np.uint8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "language": ["first task", "second task"],
        },
    }

    output = step(transition)

    assert fake_proc.rendered_texts == ["first task", "second task"]
    assert fake_proc.encoded_texts == ["rendered:first task", "rendered:second task"]
    assert "video" not in output[TransitionKey.OBSERVATION]
    torch.testing.assert_close(
        output[TransitionKey.COMPLEMENTARY_DATA]["input_ids"],
        torch.tensor([[0], [1]]),
    )


def test_groot_n1_7_vlm_encode_packs_images_time_major_then_camera_order():
    class FakeProcessor:
        def __init__(self):
            self.add_generation_prompts = []
            self.conversation_image_values = []
            self.conversation_texts = []
            self.encoded_texts = None
            self.encoded_image_values = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            assert tokenize is False
            self.add_generation_prompts.append(add_generation_prompt)
            content = conversation[0]["content"]
            self.conversation_image_values.append(
                [int(np.asarray(item["image"])[0, 0, 0]) for item in content if item["type"] == "image"]
            )
            text = content[-1]["text"]
            self.conversation_texts.append(text)
            return f"rendered:{text}"

        def __call__(self, text, images, return_tensors, padding):
            assert return_tensors == "pt"
            assert padding is True
            self.encoded_texts = text
            self.encoded_image_values = [int(np.asarray(image)[0, 0, 0]) for image in images]
            return {
                "input_ids": torch.arange(len(text)).view(len(text), 1),
                "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
                "pixel_values": torch.arange(len(images)).view(len(images), 1),
                "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
            }

    fake_proc = FakeProcessor()
    step = GrootN17VLMEncodeStep()
    step._proc = fake_proc
    video = np.zeros((2, 2, 2, 2, 2, 3), dtype=np.uint8)
    image_id = 1
    for batch_idx in range(2):
        for timestep in range(2):
            for view_idx in range(2):
                video[batch_idx, timestep, view_idx, :, :, :] = image_id
                image_id += 1
    transition = {
        TransitionKey.OBSERVATION: {"video": video},
        TransitionKey.COMPLEMENTARY_DATA: {"language": ["task a", "task b"]},
    }

    output = step(transition)

    assert fake_proc.conversation_image_values == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert fake_proc.encoded_image_values == [1, 2, 3, 4, 5, 6, 7, 8]
    assert fake_proc.conversation_texts == ["task a", "task b"]
    assert fake_proc.encoded_texts == ["rendered:task a", "rendered:task b"]
    assert fake_proc.add_generation_prompts == [False, False]
    assert "video" not in output[TransitionKey.OBSERVATION]
    assert set(output[TransitionKey.COMPLEMENTARY_DATA]) >= {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }


def test_groot_n1_7_vlm_image_transform_matches_albumentations_eval_path():
    cv2 = pytest.importorskip("cv2", exc_type=ImportError)
    from PIL import Image

    image_np = (np.arange(360 * 360 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(360, 360, 3)

    transformed = _transform_n1_7_image_for_vlm(
        Image.fromarray(image_np),
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
    )

    expected = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_AREA)
    crop_edge = int(256 * 0.95)
    crop_start = (256 - crop_edge) // 2
    expected = expected[crop_start : crop_start + crop_edge, crop_start : crop_start + crop_edge]
    expected = cv2.resize(expected, (256, 256), interpolation=cv2.INTER_AREA)

    assert transformed.size == (256, 256)
    np.testing.assert_array_equal(np.asarray(transformed), expected)


def test_groot_n1_7_vlm_encode_transforms_non_square_two_camera_sample_like_core_albumentations():
    cv2 = pytest.importorskip("cv2", exc_type=ImportError)

    class FakeProcessor:
        def __init__(self):
            self.images = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            return conversation[0]["content"][-1]["text"]

        def __call__(self, text, images, return_tensors, padding):
            self.images = images
            return {
                "input_ids": torch.ones(len(text), 1, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
            }

    camera_a = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    camera_b = (np.arange(3 * 5 * 3, dtype=np.uint16).reshape(3, 5, 3) * 3 % 251).astype(np.uint8)
    video = np.stack([camera_a, camera_b], axis=0).reshape(1, 1, 2, 3, 5, 3)
    fake_proc = FakeProcessor()
    step = GrootN17VLMEncodeStep(
        image_target_size=[8, 8],
        shortest_image_edge=10,
        crop_fraction=0.6,
        use_albumentations=True,
    )
    step._proc = fake_proc

    step(
        {
            TransitionKey.OBSERVATION: {"video": video},
            TransitionKey.COMPLEMENTARY_DATA: {"language": ["move"]},
        }
    )

    assert fake_proc.images is not None
    assert len(fake_proc.images) == 2
    np.testing.assert_array_equal(
        np.asarray(fake_proc.images[0]),
        _expected_albumentations_eval_image(
            camera_a,
            cv2,
            target_size=[8, 8],
            shortest_edge=10,
            crop_fraction=0.6,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(fake_proc.images[1]),
        _expected_albumentations_eval_image(
            camera_b,
            cv2,
            target_size=[8, 8],
            shortest_edge=10,
            crop_fraction=0.6,
        ),
    )


def test_groot_n1_7_vlm_encode_config_round_trips_model_name():
    step = GrootN17VLMEncodeStep(
        model_name="local-cosmos",
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
    )

    restored = GrootN17VLMEncodeStep(**step.get_config())

    assert restored.model_name == "local-cosmos"
    assert restored.image_crop_size == [230, 230]
    assert restored.image_target_size == [256, 256]
    assert restored.shortest_image_edge == 256
    assert restored.crop_fraction == 0.95
    assert restored.use_albumentations is True


def test_groot_n1_7_processor_uses_qwen_component_assets(monkeypatch):
    pytest.importorskip("transformers")

    import transformers

    from lerobot.policies.groot import processor_groot

    calls = []

    class FakeTokenizer:
        chat_template = "fake-chat-template"
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return cls()

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("image_processor", model_name, kwargs))
            return cls()

    class FakeVideoProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("video_processor", model_name, kwargs))
            return cls()

    class FakeProcessor:
        from_pretrained_called = False

        def __init__(self, *, image_processor, tokenizer, video_processor, chat_template):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_called = True
            raise AssertionError("Cosmos does not publish processor_config.json")

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "Qwen2VLImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(transformers, "Qwen3VLVideoProcessor", FakeVideoProcessor)
    monkeypatch.setattr(transformers, "Qwen3VLProcessor", FakeProcessor)

    processor = processor_groot._build_n1_7_processor("nvidia/Cosmos-Reason2-2B")

    assert [call[:2] for call in calls] == [
        ("tokenizer", "nvidia/Cosmos-Reason2-2B"),
        ("image_processor", "nvidia/Cosmos-Reason2-2B"),
        ("video_processor", "nvidia/Cosmos-Reason2-2B"),
    ]
    assert all(call[2] == {"trust_remote_code": True} for call in calls)
    assert processor.tokenizer.padding_side == "left"
    assert processor.chat_template == "fake-chat-template"
    assert not FakeProcessor.from_pretrained_called


def test_groot_n1_7_saved_processors_reload_through_factory(tmp_path):
    config = _groot_config(GROOT_N1_7)
    dataset_stats = {
        OBS_STATE: {
            "min": torch.zeros(8),
            "max": torch.ones(8),
        },
        ACTION: {
            "min": torch.zeros(7),
            "max": torch.ones(7),
        },
    }
    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=dataset_stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        dataset_stats=dataset_stats,
    )

    pack_step = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    unpack_step = loaded_postprocessor.steps[0]
    assert pack_step.normalize_min_max
    torch.testing.assert_close(pack_step.stats[OBS_STATE]["min"], dataset_stats[OBS_STATE]["min"])
    torch.testing.assert_close(pack_step.stats[ACTION]["max"], dataset_stats[ACTION]["max"])
    torch.testing.assert_close(unpack_step.stats[OBS_STATE]["min"], dataset_stats[OBS_STATE]["min"])
    torch.testing.assert_close(unpack_step.stats[ACTION]["max"], dataset_stats[ACTION]["max"])
    assert unpack_step.env_action_dim == 7


def test_groot_n1_7_saved_processors_reload_through_factory_preserves_saved_stats(tmp_path):
    config = _groot_config(GROOT_N1_7)
    saved_stats = {
        OBS_STATE: {
            "min": torch.full((8,), -2.0),
            "max": torch.full((8,), 2.0),
        },
        ACTION: {
            "min": torch.full((7,), -3.0),
            "max": torch.full((7,), 3.0),
        },
    }
    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=saved_stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
    )

    pack_step = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    unpack_step = loaded_postprocessor.steps[0]
    assert pack_step.normalize_min_max
    torch.testing.assert_close(pack_step.stats[OBS_STATE]["min"], saved_stats[OBS_STATE]["min"])
    torch.testing.assert_close(pack_step.stats[ACTION]["max"], saved_stats[ACTION]["max"])
    torch.testing.assert_close(unpack_step.stats[OBS_STATE]["min"], saved_stats[OBS_STATE]["min"])
    torch.testing.assert_close(unpack_step.stats[ACTION]["max"], saved_stats[ACTION]["max"])
    assert unpack_step.env_action_dim == 7


def test_groot_policy_selects_n1_7_model_class(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    called = {}

    def fake_from_pretrained(cls, **kwargs):
        called.update(kwargs)
        return _DummyGrootModel()

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(fake_from_pretrained))

    policy = GrootPolicy(_groot_config(GROOT_N1_7))

    assert called["pretrained_model_name_or_path"] == GROOT_N1_7_BASE_MODEL
    assert isinstance(policy._groot_model, _DummyGrootModel)


def test_groot_policy_forwards_n1_7_qwen_inputs(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    dummy_model = _DummyGrootModel()
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: dummy_model))
    policy = GrootPolicy(_groot_config(GROOT_N1_7))

    batch = {
        "state": torch.zeros(2, 1, 132),
        "action": torch.zeros(2, 40, 132),
        "action_mask": torch.ones(2, 40, 132),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "input_ids": torch.ones(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "pixel_values": torch.zeros(4, 3, 16, 16),
        "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(2, 8, dtype=torch.int32),
        "pixel_values_videos": torch.zeros(1, 3, 16, 16),
        "video_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "next.state": torch.ones(2, 1, 132),
        "info": {"ignored": True},
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(1.0)
    assert metrics == {"loss": pytest.approx(1.0)}
    assert set(dummy_model.forward_inputs) == {
        "state",
        "action",
        "action_mask",
        "embodiment_id",
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
        "pixel_values_videos",
        "video_grid_thw",
    }


def test_groot_n1_7_libero_execution_horizon_uses_core_eight_action_cadence(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)

    assert infer_groot_n1_7_action_horizon(model_path, "libero_sim") == 16
    assert infer_groot_n1_7_action_execution_horizon(model_path, "libero_sim") == 8


def test_groot_n1_7_select_action_uses_checkpoint_valid_horizon(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)

    class HorizonModel(_DummyGrootModel):
        def get_action(self, inputs):
            assert inputs["action_mask"].shape == (1, 40)
            assert inputs["action_mask"][0, :16].sum().item() == 16
            assert inputs["action_mask"][0, 16:].sum().item() == 0
            batch_size = inputs["state"].shape[0]
            steps = torch.arange(40, dtype=torch.float32).view(1, 40, 1).expand(batch_size, 40, 132)
            return {"action_pred": steps}

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: HorizonModel()))
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    config = GrootConfig(
        model_version=GROOT_N1_7,
        base_model_path=str(model_path),
        embodiment_tag="libero_sim",
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        n_action_steps=40,
    )
    policy = GrootPolicy(config)
    batch = {
        "state": torch.zeros(1, 1, 132),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "action_mask": torch.cat((torch.ones(1, 16), torch.zeros(1, 24)), dim=1),
    }

    first_action = policy.select_action(batch)

    assert policy._action_queue_steps == 8
    assert len(policy._action_queue) == 7
    torch.testing.assert_close(first_action[0, 0], torch.tensor(0.0))

    for expected_step in range(1, 8):
        action = policy.select_action(batch)
        torch.testing.assert_close(action[0, 0], torch.tensor(float(expected_step)))

    refreshed_action = policy.select_action(batch)
    torch.testing.assert_close(refreshed_action[0, 0], torch.tensor(0.0))


def test_qwen3_backbone_uses_nested_transformers_model_contract(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.feature_extraction_utils import BatchFeature

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])

    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.visual = FakeVisual()

    class FakeQwenForConditionalGeneration(nn.Module):
        config = SimpleNamespace(image_token_id=42)

        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            super().eval()
            return self

        def forward(self, **kwargs):
            batch_size, sequence_length = kwargs["input_ids"].shape
            features = torch.arange(batch_size * sequence_length * 4, dtype=torch.float32).view(
                batch_size, sequence_length, 4
            )
            return SimpleNamespace(hidden_states=[features, features + 1])

    monkeypatch.setattr(
        groot_n1_7,
        "Qwen3VLForConditionalGeneration",
        FakeQwenForConditionalGeneration,
    )
    backbone = groot_n1_7.Qwen3Backbone(
        model_name="fake-qwen",
        select_layer=2,
        tune_llm=False,
        tune_visual=False,
        use_flash_attention=False,
    )

    assert not hasattr(backbone.model, "language_model")
    assert len(backbone.language_model.layers) == 2
    assert not any(parameter.requires_grad for parameter in backbone.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.visual.parameters())

    output = backbone.forward(
        BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 42, 2], [42, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                "pixel_values": torch.zeros(2, 3, 2, 2),
                "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            }
        )
    )

    assert output["backbone_features"].shape == (2, 3, 4)
    torch.testing.assert_close(
        output["image_mask"],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    torch.testing.assert_close(
        output["backbone_attention_mask"],
        torch.tensor([[True, True, False], [True, True, True]]),
    )


def test_qwen3_backbone_can_initialize_from_config_without_downloading_weights(monkeypatch):
    pytest.importorskip("transformers")

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])

    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.visual = FakeVisual()

    class FakeQwenForConditionalGeneration(nn.Module):
        config = SimpleNamespace(image_token_id=42)
        from_pretrained_called = False
        from_config_called = False

        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_called = True
            raise AssertionError("Qwen backbone weights should not be loaded separately")

        @classmethod
        def _from_config(cls, config, **kwargs):
            cls.from_config_called = True
            return cls()

        def eval(self):
            super().eval()
            return self

    monkeypatch.setattr(groot_n1_7, "Qwen3VLForConditionalGeneration", FakeQwenForConditionalGeneration)

    backbone = groot_n1_7.Qwen3Backbone(
        model_name="nvidia/Cosmos-Reason2-2B",
        select_layer=2,
        load_pretrained_weights=False,
    )

    assert isinstance(backbone.model, FakeQwenForConditionalGeneration)
    assert FakeQwenForConditionalGeneration.from_config_called
    assert not FakeQwenForConditionalGeneration.from_pretrained_called


def test_gr00t_n1_7_from_pretrained_defers_backbone_weight_loading(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from huggingface_hub.errors import HFValidationError

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    called = {}

    class FakeLoadedModel:
        def __init__(self):
            self.config = SimpleNamespace(tune_top_llm_layers=0)
            self.backbone = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)
            self.action_head = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)

    def fake_snapshot_download(*args, **kwargs):
        raise HFValidationError("local path")

    def fake_super_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        called["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        called.update(kwargs)
        return FakeLoadedModel()

    monkeypatch.setattr(groot_n1_7, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        groot_n1_7.PreTrainedModel,
        "from_pretrained",
        classmethod(fake_super_from_pretrained),
    )

    loaded = groot_n1_7.GR00TN17.from_pretrained(str(tmp_path))

    assert isinstance(loaded, FakeLoadedModel)
    assert called["pretrained_model_name_or_path"] == str(tmp_path)
    assert called["load_backbone_weights"] is False


def test_gr00t_n1_7_action_head_meta_init_defers_beta_distribution():
    pytest.importorskip("diffusers")
    # GR00TN17Config subclasses transformers.PretrainedConfig (object fallback otherwise).
    pytest.importorskip("transformers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        state_history_length=1,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        use_vlln=False,
        add_pos_embed=False,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
    )

    with torch.device("meta"):
        meta_action_head = GR00TN17ActionHead(config)

    assert meta_action_head._beta_dist is None
    assert any(parameter.is_meta for parameter in meta_action_head.parameters())

    action_head = GR00TN17ActionHead(config)
    sample = action_head.sample_time(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)

    assert action_head._beta_dist is not None
    assert sample.shape == (3,)
    assert torch.isfinite(sample).all()


def test_gr00t_n1_7_model_forward_with_mocked_backbone():
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")

    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        use_vlln=True,
        vl_self_attention_cfg={"num_layers": 0},
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
    )

    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def prepare_input(self, inputs):
            return BatchFeature(data=inputs)

        def forward(self, inputs):
            batch_size = inputs["state"].shape[0]
            return BatchFeature(
                data={
                    "backbone_features": torch.randn(batch_size, 3, config.backbone_embedding_dim),
                    "backbone_attention_mask": torch.ones(batch_size, 3, dtype=torch.bool),
                    "image_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
                }
            )

        def set_trainable_parameters(self, *args, **kwargs):
            return None

    with patch(
        "lerobot.policies.groot.groot_n1_7.get_backbone_cls",
        return_value=lambda **kwargs: MockBackbone(),
    ):
        model = GR00TN17(config)

    inputs = {
        "state": torch.randn(2, config.state_history_length, config.max_state_dim),
        "action": torch.randn(2, config.action_horizon, config.max_action_dim),
        "action_mask": torch.ones(2, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }

    output = model.forward(inputs)
    assert output["loss"].dim() == 0
    assert torch.isfinite(output["loss"])

    inference_inputs = {key: value for key, value in inputs.items() if key != "action"}
    action_output = model.get_action(inference_inputs)
    assert action_output["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)


# ---------------------------------------------------------------------------
# GR00T N1.5 removal: every detection point must fail with the canonical guidance
# ---------------------------------------------------------------------------


def test_groot_config_rejects_legacy_n1_5_tokenizer_assets_repo():
    with pytest.raises(ValueError, match="tokenizer_assets_repo") as exc_info:
        GrootConfig(tokenizer_assets_repo="nvidia/GR00T-N1.5-3B", device="cpu")

    assert GROOT_N1_5_REMOVAL_GUIDANCE in str(exc_info.value)


def test_groot_legacy_n1_5_checkpoint_config_fails_with_removal_guidance(tmp_path):
    # config.json layout serialized by lerobot<=0.5.1 groot checkpoints: legacy
    # N1.5 defaults plus the N1.5-only 'tokenizer_assets_repo' field.
    legacy_config = {
        "type": "groot",
        "n_obs_steps": 1,
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 64,
        "max_action_dim": 32,
        "model_version": "n1.5",
        "base_model_path": "nvidia/GR00T-N1.5-3B",
        "tokenizer_assets_repo": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "gr1",
        "video_backend": "decord",
        "output_dir": "./tmp/gr00t",
        "device": "cpu",
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy_config))

    with pytest.raises(ParsingError) as exc_info:
        PreTrainedConfig.from_pretrained(tmp_path)

    # draccus wraps the dataclass error in a generic ParsingError; the clear N1.5
    # removal message must be the root cause instead of an opaque DecodingError
    # about unknown config fields.
    messages = []
    error: BaseException | None = exc_info.value
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    assert any(
        "tokenizer_assets_repo" in message and GROOT_N1_5_REMOVAL_GUIDANCE in message for message in messages
    )


@pytest.mark.parametrize(
    "config_payload",
    [
        {"model_type": "gr00t_n1_5"},
        {"architectures": ["GR00T_N1_5"]},
        {"model_version": "n1_5"},
        {"backbone_cfg": {"eagle_path": "eagle"}},
    ],
)
def test_groot_config_rejects_generic_local_dir_with_n1_5_content_markers(tmp_path, config_payload):
    # A renamed local snapshot has no N1.5 hint in its path; the config.json
    # content markers (as shipped by nvidia/GR00T-N1.5-3B) must still be detected.
    model_path = tmp_path / "renamed-snapshot"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config_payload))

    assert infer_groot_model_version(str(model_path)) == GROOT_N1_5

    with pytest.raises(ValueError, match="does not match base_model_path") as exc_info:
        GrootConfig(base_model_path=str(model_path), device="cpu")

    assert GROOT_N1_5_REMOVAL_GUIDANCE in str(exc_info.value)


@pytest.mark.parametrize(
    "registry_name",
    [
        "groot_pack_inputs_v3",
        "groot_eagle_encode_v3",
        "groot_eagle_collate_v3",
        "groot_action_unpack_unnormalize_v1",
    ],
)
def test_removed_n1_5_processor_steps_fail_with_removal_guidance(tmp_path, registry_name):
    (tmp_path / "processor.json").write_text(
        json.dumps(
            {"name": "legacy_groot_processor", "steps": [{"registry_name": registry_name, "config": {}}]}
        )
    )

    with pytest.raises(ValueError, match=re.escape(GROOT_N1_5_REMOVAL_GUIDANCE)):
        PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="processor.json")


def test_groot_action_unpack_step_registers_and_serializes_as_v2(tmp_path):
    # The action-chunk semantics changed vs. the N1.5-era v1 step, so the registry
    # name was bumped: v1 must never silently load into the new implementation.
    assert ProcessorStepRegistry.get("groot_action_unpack_unnormalize_v2") is GrootActionUnpackUnnormalizeStep

    config = _groot_config(GROOT_N1_7)
    dataset_stats = {
        OBS_STATE: {"min": torch.zeros(8), "max": torch.ones(8)},
        ACTION: {"min": torch.zeros(7), "max": torch.ones(7)},
    }
    _, postprocessor = make_groot_pre_post_processors(config, dataset_stats=dataset_stats)
    postprocessor.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "policy_postprocessor.json").read_text())
    assert saved["steps"][0]["registry_name"] == "groot_action_unpack_unnormalize_v2"


# ---------------------------------------------------------------------------
# Legacy N1.5-era default remapping warns instead of silently rewriting values
# ---------------------------------------------------------------------------


def test_groot_default_config_uses_n1_7_values_without_warnings(caplog):
    with caplog.at_level(logging.WARNING, logger="lerobot.policies.groot.configuration_groot"):
        config = GrootConfig(device="cpu")

    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.chunk_size == 40
    assert config.n_action_steps == 40
    assert tuple(config.image_size) == (256, 256)
    assert not any("legacy GR00T N1.5-era default" in record.getMessage() for record in caplog.records)


def test_groot_legacy_default_remap_emits_warnings(caplog):
    with caplog.at_level(logging.WARNING, logger="lerobot.policies.groot.configuration_groot"):
        config = GrootConfig(
            chunk_size=50,
            n_action_steps=50,
            max_state_dim=64,
            max_action_dim=32,
            image_size=(224, 224),
            device="cpu",
        )

    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.chunk_size == 40
    assert config.n_action_steps == 40
    assert tuple(config.image_size) == (256, 256)
    remap_messages = [
        record.getMessage()
        for record in caplog.records
        if "legacy GR00T N1.5-era default" in record.getMessage()
    ]
    assert any("chunk_size=50" in message and "40" in message for message in remap_messages)
    assert any("n_action_steps=50" in message and "40" in message for message in remap_messages)
    assert any("max_state_dim=64" in message and "132" in message for message in remap_messages)
    assert any("max_action_dim=32" in message and "132" in message for message in remap_messages)
    assert any("image_size=(224, 224)" in message and "(256, 256)" in message for message in remap_messages)


def test_groot_non_legacy_values_are_not_remapped(caplog):
    with caplog.at_level(logging.WARNING, logger="lerobot.policies.groot.configuration_groot"):
        config = GrootConfig(
            chunk_size=48,
            n_action_steps=20,
            max_state_dim=100,
            max_action_dim=65,
            image_size=(225, 225),
            device="cpu",
        )

    assert config.chunk_size == 48
    assert config.n_action_steps == 20
    assert config.max_state_dim == 100
    assert config.max_action_dim == 65
    assert tuple(config.image_size) == (225, 225)
    assert not any("legacy GR00T N1.5-era default" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# action_decode_transform: explicit 'none' wins over the embodiment default
# ---------------------------------------------------------------------------


def test_groot_explicit_none_action_decode_transform_overrides_libero_default():
    config = GrootConfig(embodiment_tag="libero_sim", action_decode_transform="none", device="cpu")

    assert config.action_decode_transform is None


@pytest.mark.parametrize("auto_value", ["auto", "AUTO"])
def test_groot_auto_action_decode_transform_resolves_to_embodiment_default(auto_value):
    libero = GrootConfig(embodiment_tag="libero_sim", action_decode_transform=auto_value, device="cpu")
    other = GrootConfig(embodiment_tag="new_embodiment", action_decode_transform=auto_value, device="cpu")

    assert libero.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO
    assert other.action_decode_transform is None


def test_groot_action_decode_transform_opt_out_survives_save_load_round_trip(tmp_path):
    explicit_none = GrootConfig(embodiment_tag="libero_sim", action_decode_transform="none", device="cpu")
    explicit_dir = tmp_path / "explicit_none"
    explicit_none.save_pretrained(explicit_dir)
    reloaded_none = PreTrainedConfig.from_pretrained(explicit_dir)

    assert reloaded_none.action_decode_transform is None

    unset = GrootConfig(embodiment_tag="libero_sim", device="cpu")
    unset_dir = tmp_path / "unset"
    unset.save_pretrained(unset_dir)
    reloaded_unset = PreTrainedConfig.from_pretrained(unset_dir)

    assert reloaded_unset.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO


# ---------------------------------------------------------------------------
# GrootN17ActionDecodeStep: 2-D (B, D) actions from the sync select_action path
# ---------------------------------------------------------------------------


def _symmetric_unit_stats(dim: int) -> dict[str, list[float]]:
    return {
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def test_groot_n1_7_action_decode_handles_2d_relative_non_eef_actions():
    raw_stats = {
        "state": {"single_arm": _symmetric_unit_stats(5)},
        "action": {"single_arm": _symmetric_unit_stats(5)},
        "relative_action": {"single_arm": {"min": [-1.0] * 5, "max": [1.0] * 5}},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm"]},
        "action": {
            "modality_keys": ["single_arm"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            ],
        },
    }
    pack_step = GrootN17PackInputsStep(
        raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, max_state_dim=8
    )
    reference = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
    pack_step({TransitionKey.OBSERVATION: {OBS_STATE: reference}, TransitionKey.COMPLEMENTARY_DATA: {}})
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=5,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
        pack_step=pack_step,
    )

    output = decode_step({TransitionKey.ACTION: torch.zeros(2, 5)})

    # Relative stats span [-1, 1], so the normalized 0 decodes to a 0 delta and the
    # absolute action equals the cached reference state, preserving the (B, D) rank.
    assert output[TransitionKey.ACTION].shape == (2, 5)
    torch.testing.assert_close(output[TransitionKey.ACTION], reference)

    # 3-D chunks keep their (B, T, D) rank and decode identically per timestep.
    chunk_output = decode_step({TransitionKey.ACTION: torch.zeros(2, 3, 5)})
    assert chunk_output[TransitionKey.ACTION].shape == (2, 3, 5)
    torch.testing.assert_close(chunk_output[TransitionKey.ACTION], reference[:, None, :].expand(2, 3, 5))


def test_groot_n1_7_action_decode_handles_2d_mixed_relative_and_absolute_groups():
    raw_stats = {
        "state": {"single_arm": {"mean": [0.0] * 5}, "gripper": {"mean": [0.0]}},
        "action": {
            "single_arm": _symmetric_unit_stats(5),
            "gripper": {
                "min": [0.0],
                "max": [10.0],
                "mean": [5.0],
                "std": [1.0],
                "q01": [0.0],
                "q99": [10.0],
            },
        },
        "relative_action": {"single_arm": {"min": [-1.0] * 5, "max": [1.0] * 5}},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm", "gripper"]},
        "action": {
            "modality_keys": ["single_arm", "gripper"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
            ],
        },
    }
    pack_step = GrootN17PackInputsStep(
        raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, max_state_dim=8
    )
    pack_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 9.0]])},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=6,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
        pack_step=pack_step,
    )

    output = decode_step({TransitionKey.ACTION: torch.zeros(1, 6)})

    # Arm group: 0 delta added to the cached reference state [0..4]. Gripper group is
    # absolute: a normalized 0 unnormalizes to the [0, 10] midpoint 5.0 (not the raw
    # reference state 9.0).
    assert output[TransitionKey.ACTION].shape == (1, 6)
    torch.testing.assert_close(output[TransitionKey.ACTION], torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]))


def test_groot_n1_7_action_decode_handles_2d_relative_eef_xyz_rot6d_actions():
    raw_stats = {
        "state": {"eef": {"mean": [0.0] * 9}},
        "action": {"eef": _symmetric_unit_stats(9)},
        "relative_action": {"eef": {"min": [-1.0] * 9, "max": [1.0] * 9}},
    }
    modality_config = {
        "state": {"modality_keys": ["eef"]},
        "action": {
            "modality_keys": ["eef"],
            "action_configs": [{"rep": "RELATIVE", "type": "EEF", "format": "XYZ+ROT6D", "state_key": None}],
        },
    }
    pack_step = GrootN17PackInputsStep(
        raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, max_state_dim=16
    )
    # Reference pose: translation (1, 2, 3) with the identity rotation in rot6d form.
    identity_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pack_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[1.0, 2.0, 3.0, *identity_rot6d]])},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=9,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
        pack_step=pack_step,
    )

    # Relative stats span [-1, 1], so the normalized values below ARE the decoded
    # deltas: translate +0.5 along x with no rotation change.
    output = decode_step({TransitionKey.ACTION: torch.tensor([[0.5, 0.0, 0.0, *identity_rot6d]])})

    assert output[TransitionKey.ACTION].shape == (1, 9)
    torch.testing.assert_close(output[TransitionKey.ACTION], torch.tensor([[1.5, 2.0, 3.0, *identity_rot6d]]))


def test_groot_n1_7_action_decode_uses_first_stat_row_for_2d_per_timestep_relative_stats():
    per_step_min = [[-float(step + 1)] * 5 for step in range(16)]
    per_step_max = [[float(step + 1)] * 5 for step in range(16)]
    raw_stats = {
        "state": {"single_arm": {"mean": [0.0] * 5}},
        "action": {"single_arm": _symmetric_unit_stats(5)},
        "relative_action": {"single_arm": {"min": per_step_min, "max": per_step_max}},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm"]},
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            ],
        },
    }
    pack_step = GrootN17PackInputsStep(
        raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, max_state_dim=8
    )
    pack_step(
        {TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 5)}, TransitionKey.COMPLEMENTARY_DATA: {}}
    )
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=5,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
        pack_step=pack_step,
    )

    output = decode_step({TransitionKey.ACTION: torch.full((1, 5), 0.5)})

    # A popped (B, D) action decodes as chunk step 0: row 0 of the per-timestep stats
    # spans [-1, 1], so 0.5 unnormalizes to 0.5 (row 1 would give 1.0) and the zero
    # reference leaves it unchanged.
    assert output[TransitionKey.ACTION].shape == (1, 5)
    torch.testing.assert_close(output[TransitionKey.ACTION], torch.full((1, 5), 0.5))


# ---------------------------------------------------------------------------
# Raw checkpoint stats fallback and per-instance raw-state cache
# ---------------------------------------------------------------------------


def _raw_n1_7_unknown_embodiment_config(model_path) -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    return GrootConfig(
        model_version=GROOT_N1_7,
        base_model_path=str(model_path),
        embodiment_tag="new_embodiment",
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
    )


def test_raw_n1_7_checkpoint_missing_embodiment_stats_falls_back_to_dataset_stats(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_unknown_embodiment_config(model_path)
    dataset_stats = {
        OBS_STATE: {"min": torch.zeros(8), "max": torch.full((8,), 30.0)},
        ACTION: {"min": torch.zeros(7), "max": torch.full((7,), 30.0)},
    }

    _, postprocessor = make_groot_pre_post_processors(config, dataset_stats=dataset_stats)

    assert isinstance(postprocessor.steps[0], GrootActionUnpackUnnormalizeStep)
    # The decode must invert the dataset-stats normalization applied by the pack step:
    # a normalized 0 decodes to the [0, 30] midpoint 15.0 instead of passing through.
    decoded = postprocessor(torch.zeros(2, 7))
    torch.testing.assert_close(decoded, torch.full((2, 7), 15.0))


def test_raw_n1_7_checkpoint_missing_embodiment_stats_without_dataset_stats_raises(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_unknown_embodiment_config(model_path)

    with pytest.raises(ValueError, match="has no statistics for embodiment tag"):
        make_groot_pre_post_processors(config, dataset_stats=None)


def test_groot_n1_7_raw_state_cache_is_per_instance():
    from lerobot.policies.groot import processor_groot

    # The process-global raw-state cache was removed: a second pipeline's preprocess
    # must not leak its reference state into the first pipeline's decode step.
    assert not hasattr(processor_groot, "_N1_7_RAW_STATE_CACHE")

    raw_stats = {
        "state": {"single_arm": _symmetric_unit_stats(5)},
        "action": {"single_arm": _symmetric_unit_stats(5)},
        "relative_action": {"single_arm": {"min": [-1.0] * 5, "max": [1.0] * 5}},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm"]},
        "action": {
            "modality_keys": ["single_arm"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            ],
        },
    }

    def build_pair():
        pack = GrootN17PackInputsStep(
            raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, max_state_dim=8
        )
        decode = GrootN17ActionDecodeStep(
            env_action_dim=5,
            raw_stats=raw_stats,
            modality_config=modality_config,
            use_relative_action=True,
            pack_step=pack,
        )
        return pack, decode

    first_pack, first_decode = build_pair()
    second_pack, _ = build_pair()
    first_reference = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    first_pack(
        {TransitionKey.OBSERVATION: {OBS_STATE: first_reference}, TransitionKey.COMPLEMENTARY_DATA: {}}
    )
    second_pack(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.full((1, 5), 99.0)},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )

    output = first_decode({TransitionKey.ACTION: torch.zeros(1, 5)})

    torch.testing.assert_close(output[TransitionKey.ACTION], first_reference)


def test_groot_n1_7_action_decode_without_connected_pack_step_raises():
    raw_stats = {
        "state": {"single_arm": _symmetric_unit_stats(5)},
        "action": {"single_arm": _symmetric_unit_stats(5)},
        "relative_action": {"single_arm": {"min": [-1.0] * 5, "max": [1.0] * 5}},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm"]},
        "action": {
            "modality_keys": ["single_arm"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            ],
        },
    }
    orphan_decode = GrootN17ActionDecodeStep(
        env_action_dim=5,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=True,
    )

    with pytest.raises(RuntimeError, match="connected GrootN17PackInputsStep"):
        orphan_decode({TransitionKey.ACTION: torch.zeros(1, 5)})


def test_groot_n1_7_loaded_processors_reconnect_pack_and_decode_steps(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "saved"
    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(save_dir),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    # The pack/decode link is not serialized, so the factory must re-link the loaded
    # decode step to the loaded pack step's per-instance raw-state cache.
    pack_step = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    decode_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)
    )
    assert decode_step.pack_step is pack_step


# ---------------------------------------------------------------------------
# Raw checkpoint factory branch: caller overrides and hub repo loading
# ---------------------------------------------------------------------------


def test_raw_n1_7_checkpoint_processors_apply_caller_overrides(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    rename_map = {f"{OBS_IMAGES}.cam": f"{OBS_IMAGES}.image"}

    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": rename_map}},
    )

    rename_step = next(
        step for step in preprocessor.steps if isinstance(step, RenameObservationsProcessorStep)
    )
    assert rename_step.rename_map == rename_map


def test_raw_n1_7_checkpoint_processors_reject_unknown_override_key(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)

    with pytest.raises(KeyError, match="does not match any step"):
        make_pre_post_processors(
            config,
            pretrained_path=str(model_path),
            preprocessor_overrides={"missing_step": {"enabled": True}},
        )


def test_raw_n1_7_checkpoint_processors_reject_unknown_override_field(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)

    with pytest.raises(TypeError, match="is not a config field"):
        make_pre_post_processors(
            config,
            pretrained_path=str(model_path),
            preprocessor_overrides={"rename_observations_processor": {"bogus_field": 1}},
        )


def test_converted_n1_7_processors_load_from_hub_repo_id_without_legacy_override_error(tmp_path, monkeypatch):
    import lerobot.processor.pipeline as pipeline_module
    from lerobot.policies.groot import processor_groot

    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "hub_repo"
    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    def fake_hf_hub_download(repo_id=None, filename=None, **kwargs):
        assert repo_id == "user/groot-finetune"
        file_path = save_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(filename)
        return str(file_path)

    monkeypatch.setattr(processor_groot, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(pipeline_module, "hf_hub_download", fake_hf_hub_download)

    # Loading from a hub repo id must inspect the serialized postprocessor and skip
    # the legacy action-unpack override instead of raising
    # KeyError("Override keys ['groot_action_unpack_unnormalize_v1'] ...").
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path="user/groot-finetune",
    )

    assert any(isinstance(step, GrootN17PackInputsStep) for step in loaded_preprocessor.steps)
    assert any(isinstance(step, GrootN17ActionDecodeStep) for step in loaded_postprocessor.steps)


def test_converted_n1_7_processors_retry_without_legacy_overrides_when_inspection_fails(
    tmp_path, monkeypatch
):
    from lerobot.policies.groot import processor_groot

    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = _raw_n1_7_libero_config(model_path)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))
    save_dir = tmp_path / "converted"
    config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    # Simulate the config inspection failing (e.g. offline without a cached copy):
    # the injected legacy overrides then miss the serialized steps, and loading must
    # fall back to the caller overrides instead of surfacing the KeyError.
    monkeypatch.setattr(processor_groot, "_pretrained_processor_config_has_step", lambda *a, **k: False)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(save_dir),
        preprocessor_overrides={"rename_observations_processor": {"rename_map": {}}},
    )

    assert any(isinstance(step, GrootN17PackInputsStep) for step in loaded_preprocessor.steps)
    assert any(isinstance(step, GrootN17ActionDecodeStep) for step in loaded_postprocessor.steps)


# ---------------------------------------------------------------------------
# Camera matching warnings
# ---------------------------------------------------------------------------


def test_groot_n1_7_pack_inputs_warns_once_for_unmatched_and_dropped_cameras(caplog):
    step = GrootN17PackInputsStep(normalize_min_max=False, video_modality_keys=["image", "side_image"])
    observation = {
        f"{OBS_IMAGES}.image": torch.full((1, 3, 2, 2), 11, dtype=torch.uint8),
        f"{OBS_IMAGES}.wrist": torch.full((1, 3, 2, 2), 22, dtype=torch.uint8),
        OBS_STATE: torch.zeros(1, 8),
    }

    with caplog.at_level(logging.WARNING):
        step(
            {
                TransitionKey.OBSERVATION: dict(observation),
                TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
            }
        )

    unmatched_warnings = [
        record.getMessage() for record in caplog.records if "no matching camera" in record.getMessage()
    ]
    dropped_warnings = [
        record.getMessage() for record in caplog.records if "Dropping camera(s)" in record.getMessage()
    ]
    assert len(unmatched_warnings) == 1
    assert "side_image" in unmatched_warnings[0]
    assert len(dropped_warnings) == 1
    assert f"{OBS_IMAGES}.wrist" in dropped_warnings[0]

    # The warnings are emitted once per step instance, not once per frame.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        step(
            {
                TransitionKey.OBSERVATION: dict(observation),
                TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
            }
        )
    assert not any("camera" in record.getMessage() for record in caplog.records)


def test_groot_n1_7_pack_inputs_does_not_warn_on_full_camera_match(caplog):
    step = GrootN17PackInputsStep(normalize_min_max=False, video_modality_keys=["image", "wrist_image"])
    observation = {
        f"{OBS_IMAGES}.image": torch.full((1, 3, 2, 2), 11, dtype=torch.uint8),
        # LIBERO conversions expose the wrist camera as image2; the alias must match silently.
        f"{OBS_IMAGES}.image2": torch.full((1, 3, 2, 2), 22, dtype=torch.uint8),
        OBS_STATE: torch.zeros(1, 8),
    }

    with caplog.at_level(logging.WARNING):
        output = step(
            {TransitionKey.OBSERVATION: observation, TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]}}
        )

    assert output[TransitionKey.OBSERVATION]["video"].shape == (1, 1, 2, 2, 2, 3)
    assert not any("camera" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# GrootPolicy bf16 handling and GR00TN17 backbone loading kwargs
# ---------------------------------------------------------------------------


def test_groot_policy_use_bf16_false_does_not_touch_model_compute_dtype(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    # _DummyGrootModel deliberately defines no compute_dtype, like the real GR00TN17:
    # with use_bf16=False the policy must not read or set the attribute.
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy(_groot_config(GROOT_N1_7))

    assert not hasattr(policy._groot_model, "compute_dtype")
    assert not hasattr(policy._groot_model.config, "compute_dtype")


def test_groot_policy_use_bf16_true_sets_model_compute_dtype(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    config = GrootConfig(
        model_version=GROOT_N1_7,
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=True,
    )

    policy = GrootPolicy(config)

    assert policy._groot_model.compute_dtype == "bfloat16"
    assert policy._groot_model.config.compute_dtype == "bfloat16"


def _stub_gr00t_n1_7_loading(monkeypatch, called, snapshot_kwargs):
    from huggingface_hub.errors import HFValidationError

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLoadedModel:
        def __init__(self):
            self.config = SimpleNamespace(tune_top_llm_layers=0)
            self.backbone = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)
            self.action_head = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)

    def fake_snapshot_download(*args, **kwargs):
        snapshot_kwargs.update(kwargs)
        raise HFValidationError("local path")

    def fake_super_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        called["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        called.update(kwargs)
        return FakeLoadedModel()

    monkeypatch.setattr(groot_n1_7, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        groot_n1_7.PreTrainedModel,
        "from_pretrained",
        classmethod(fake_super_from_pretrained),
    )
    return groot_n1_7


def test_gr00t_n1_7_from_pretrained_does_not_forward_revision_to_backbone(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    called: dict = {}
    snapshot_kwargs: dict = {}
    groot_n1_7 = _stub_gr00t_n1_7_loading(monkeypatch, called, snapshot_kwargs)

    groot_n1_7.GR00TN17.from_pretrained(
        str(tmp_path),
        revision="deadbeefcafe",
        cache_dir=str(tmp_path / "cache"),
        token="hf_dummy",
        local_files_only=False,
    )

    # 'revision' pins the GR00T checkpoint repo and must not leak into the unrelated
    # backbone repo load; the repo-agnostic hub kwargs are still forwarded.
    transformers_loading_kwargs = called["transformers_loading_kwargs"]
    assert "revision" not in transformers_loading_kwargs
    assert transformers_loading_kwargs["cache_dir"] == str(tmp_path / "cache")
    assert transformers_loading_kwargs["local_files_only"] is False
    assert transformers_loading_kwargs["token"] == "hf_dummy"
    assert snapshot_kwargs["revision"] == "deadbeefcafe"


def test_gr00t_n1_7_from_pretrained_preserves_explicit_backbone_revision(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    called: dict = {}
    snapshot_kwargs: dict = {}
    groot_n1_7 = _stub_gr00t_n1_7_loading(monkeypatch, called, snapshot_kwargs)

    groot_n1_7.GR00TN17.from_pretrained(
        str(tmp_path),
        revision="deadbeefcafe",
        transformers_loading_kwargs={"revision": "backbone-tag", "trust_remote_code": True},
    )

    assert called["transformers_loading_kwargs"]["revision"] == "backbone-tag"


def test_get_backbone_cls_warns_only_for_unrecognized_qwen_backbone_names(caplog):
    pytest.importorskip("transformers")
    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    with caplog.at_level(logging.WARNING, logger=groot_n1_7.__name__):
        recognized = groot_n1_7.get_backbone_cls(
            groot_n1_7.GR00TN17Config(model_name="nvidia/Cosmos-Reason2-2B")
        )
    assert recognized is groot_n1_7.Qwen3Backbone
    assert not any("Unrecognized" in record.getMessage() for record in caplog.records)

    # Local snapshot paths carry no recognized repo marker; with the default
    # backbone_model_type='qwen' they must load with a loud assumption warning.
    with caplog.at_level(logging.WARNING, logger=groot_n1_7.__name__):
        local = groot_n1_7.get_backbone_cls(groot_n1_7.GR00TN17Config(model_name="/local/backbone/snapshot"))
    assert local is groot_n1_7.Qwen3Backbone
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Unrecognized GR00T N1.7 backbone model name" in record.getMessage()
    ]
    assert len(warnings) == 1

    with pytest.raises(ValueError, match="Unsupported GR00T N1.7 backbone model"):
        groot_n1_7.get_backbone_cls(
            groot_n1_7.GR00TN17Config(model_name="totally/bogus", backbone_model_type=None)
        )
