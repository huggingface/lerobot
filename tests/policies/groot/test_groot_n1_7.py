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
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.policies.groot.configuration_groot import (
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    GROOT_N1_7,
    GROOT_N1_7_BASE_MODEL,
    GrootConfig,
    infer_groot_n1_7_action_execution_horizon,
    infer_groot_n1_7_action_horizon,
    normalize_groot_model_version,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    N1_7_NATIVE_ACTION_HORIZON,
    GrootActionUnpackUnnormalizeStep,
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _make_relative_action_training_stats,
    _transform_n1_7_image_for_vlm_albumentations,
    _transform_n1_7_image_for_vlm_torch,
    make_groot_pre_post_processors,
)
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
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


def _groot_config() -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    kwargs = {"action_decode_transform": GROOT_ACTION_DECODE_TRANSFORM_LIBERO}
    return GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        **kwargs,
    )


def _native_action_chunk(rows: list[list[float]]) -> torch.Tensor:
    chunk = torch.tensor(rows, dtype=torch.float32)
    if chunk.shape[0] >= N1_7_NATIVE_ACTION_HORIZON:
        return chunk[:N1_7_NATIVE_ACTION_HORIZON]
    tail = chunk[-1:].repeat(N1_7_NATIVE_ACTION_HORIZON - chunk.shape[0], 1)
    return torch.cat([chunk, tail], dim=0)


def _raw_n1_7_libero_config(model_path) -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    return GrootConfig(
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
    pytest.importorskip("transformers")

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
                    "letter_box_transform": False,
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
    del target_size

    def resize_shortest_edge(frame):
        height, width = frame.shape[:2]
        scale = shortest_edge / float(min(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    image_np = resize_shortest_edge(image_np)
    height, width = image_np.shape[:2]
    crop_h = max(1, int(height * crop_fraction))
    crop_w = max(1, int(width * crop_fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return resize_shortest_edge(image_np[top : top + crop_h, left : left + crop_w])


class _DummyGrootModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(compute_dtype="float32")
        self.compute_dtype = "float32"
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

    assert config.base_model_path == GROOT_N1_7_BASE_MODEL
    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.chunk_size == 40
    assert config.n_action_steps == 40
    assert len(config.action_delta_indices) == 40


@pytest.mark.parametrize("legacy_version", ["n1.5", "n1_5", "n15", "1.5"])
def test_groot_normalize_model_version_rejects_n1_5_aliases(legacy_version):
    # model_version is no longer a GrootConfig field, but normalize_groot_model_version is still
    # live (e.g. via infer_groot_model_version) and must keep rejecting N1.5 with removal guidance.
    with pytest.raises(ValueError, match="Unsupported GR00T model_version"):
        normalize_groot_model_version(legacy_version)


def test_groot_normalize_model_version_accepts_n1_7():
    assert normalize_groot_model_version(GROOT_N1_7) == GROOT_N1_7


def test_groot_n1_7_accepts_named_action_decode_transform():
    config = GrootConfig(
        action_decode_transform="libero",
        device="cpu",
    )

    assert config.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO


@pytest.mark.parametrize("legacy_transform", ["libero_gripper", "libero-gripper"])
def test_groot_n1_7_rejects_legacy_libero_gripper_action_decode_transform(legacy_transform):
    with pytest.raises(ValueError, match="Unsupported GR00T N1.7 action decode transform"):
        GrootConfig(
            action_decode_transform=legacy_transform,
            device="cpu",
        )


def test_groot_config_rejects_mismatched_n1_5_path_for_n1_7():
    with pytest.raises(ValueError, match="does not match base_model_path"):
        GrootConfig(
            base_model_path="nvidia/GR00T-N1.5-3B",
            device="cpu",
        )


def test_groot_n1_7_can_be_selected_from_policy_config_factory_without_external_gr00t():
    sys.modules.pop("gr00t", None)

    config = make_policy_config("groot", device="cpu")

    assert isinstance(config, GrootConfig)
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
    config = _groot_config()
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
    config = _groot_config()
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


def test_groot_from_pretrained_rejects_mismatched_caller_config(tmp_path):
    model_path = tmp_path / "GR00T-N1.7-local"
    model_path.mkdir()
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)

    # An N1.7 config paired with a legacy N1.5 base path is a mismatch and must be
    # rejected. The mismatch is detected during config validation (__post_init__),
    # so construction itself raises before from_pretrained is reached.
    with pytest.raises(ValueError, match="does not match base_model_path"):
        config = GrootConfig(
            base_model_path="nvidia/GR00T-N1.5-3B",
            input_features=input_features,
            output_features=output_features,
            device="cpu",
            use_bf16=False,
            action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
        )
        GrootPolicy.from_pretrained(model_path, config=config)


def test_groot_from_pretrained_keeps_matching_caller_config(tmp_path, monkeypatch):
    pytest.importorskip("transformers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "GR00T-N1.7-local"
    model_path.mkdir()
    config = _groot_config()

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path, config=config)

    assert policy.config.base_model_path == str(model_path)


def test_groot_from_pretrained_infers_n1_7_from_ambiguous_local_config(tmp_path, monkeypatch):
    pytest.importorskip("transformers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "local-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "Gr00tN1d7"}')

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path)

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
    assert vlm_encode.letter_box_transform is False
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
    vlm_encode = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17VLMEncodeStep))
    decode_actions = next(
        step for step in loaded_postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)
    )

    assert pack_inputs.valid_action_horizon == 16
    assert pack_inputs.action_horizon == 40
    assert pack_inputs.video_modality_keys == ["image", "wrist_image"]
    assert pack_inputs.clip_outliers is True
    assert vlm_encode.letter_box_transform is False
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


def test_groot_n1_7_pack_inputs_raises_when_relative_groups_cannot_normalize():
    # Relative groups carry per-chunk-timestep stats; if the action horizon exceeds the available
    # stat rows, grouped normalization cannot apply and the flat fallback would silently wrongly scale.
    step = GrootN17PackInputsStep(
        action_horizon=3,
        valid_action_horizon=3,
        max_state_dim=2,
        max_action_dim=2,
        normalize_min_max=True,
        raw_stats={
            "state": {"single_arm": {"min": [0.0, 0.0], "max": [1.0, 1.0]}},
            "action": {"single_arm": {"min": [0.0, 0.0], "max": [1.0, 1.0]}},
            # only one horizon row, but the action chunk has horizon 3
            "relative_action": {"single_arm": {"min": [[-1.0, -1.0]], "max": [[1.0, 1.0]]}},
        },
        modality_config={
            "state": {"modality_keys": ["single_arm"]},
            "action": {
                "modality_keys": ["single_arm"],
                "action_configs": [
                    {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
                ],
                "delta_indices": [0, 1, 2],
            },
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 2)},
        TransitionKey.ACTION: torch.zeros(1, 3, 2),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    with pytest.raises(ValueError, match="could not apply native grouped normalization"):
        step(transition)


def test_groot_n1_7_pack_inputs_trains_native_relative_groups_with_absolute_gripper():
    step = GrootN17PackInputsStep(
        action_horizon=2,
        valid_action_horizon=2,
        max_state_dim=6,
        max_action_dim=6,
        normalize_min_max=True,
        clip_outliers=False,
        stats={
            OBS_STATE: {
                "min": [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
                "max": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            },
            ACTION: {
                "min": [-10.0, -10.0, -10.0, -10.0, -10.0, 0.0],
                "max": [10.0, 10.0, 10.0, 10.0, 10.0, 100.0],
            },
        },
        raw_stats={
            "state": {
                "single_arm": {"min": [-100.0] * 5, "max": [100.0] * 5},
                "gripper": {"min": [0.0], "max": [100.0]},
            },
            "action": {
                "single_arm": {"min": [-100.0] * 5, "max": [100.0] * 5},
                "gripper": {"min": [0.0], "max": [100.0]},
            },
            "relative_action": {
                "single_arm": {"min": [-10.0] * 5, "max": [10.0] * 5},
            },
        },
        modality_config={
            "state": {"modality_keys": ["single_arm", "gripper"]},
            "action": {
                "modality_keys": ["single_arm", "gripper"],
                "action_configs": [
                    {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                    {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                ],
                "delta_indices": [0, 1],
            },
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 25.0]]),
        },
        TransitionKey.ACTION: torch.tensor(
            [
                [
                    [12.0, 18.0, 35.0, 30.0, 55.0, 0.0],
                    [9.0, 21.0, 27.0, 43.0, 50.0, 100.0],
                ]
            ]
        ),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    expected_actions = torch.tensor(
        [
            [
                [0.2, -0.2, 0.5, -1.0, 0.5, -1.0],
                [-0.1, 0.1, -0.3, 0.3, 0.0, 1.0],
            ]
        ]
    )
    torch.testing.assert_close(output[TransitionKey.ACTION], expected_actions)


def test_groot_policy_ignores_rtc_leftovers_for_relative_actions():
    policy = object.__new__(GrootPolicy)
    policy.config = SimpleNamespace(use_relative_actions=True)
    policy._warned_native_relative_rtc_prefix_disabled = False
    inputs = {"state": torch.zeros(1, 1, 132)}

    output_inputs, options = policy._prepare_n1_7_rtc_inputs(
        inputs,
        inference_delay=1,
        prev_chunk_left_over=torch.ones(8, 6),
    )

    assert output_inputs is inputs
    assert options is None


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


def test_groot_n1_7_pack_inputs_masks_padded_action_horizons():
    step = GrootN17PackInputsStep(
        action_horizon=4,
        valid_action_horizon=4,
        max_state_dim=3,
        max_action_dim=5,
        normalize_min_max=False,
    )
    action = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    action_is_pad = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, False],
        ]
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(2, 3),
        },
        TransitionKey.ACTION: action.clone(),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Move", "Place"],
            "action_is_pad": action_is_pad,
        },
    }

    output = step(transition)

    expected_valid = (~action_is_pad).float()
    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (2, 4, 5)
    torch.testing.assert_close(action_mask[..., :3], expected_valid.unsqueeze(-1).expand(-1, -1, 3))
    assert action_mask[..., 3:].sum().item() == 0

    packed_action = output[TransitionKey.ACTION]
    assert packed_action.shape == (2, 4, 5)
    torch.testing.assert_close(packed_action[0, 0, :3], action[0, 0])
    torch.testing.assert_close(packed_action[0, 2, :3], action[0, 2])
    assert packed_action[0, 1].abs().sum().item() == 0
    assert packed_action[0, 3].abs().sum().item() == 0
    assert packed_action[1, 0].abs().sum().item() == 0


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


def test_groot_n1_7_action_decode_rejects_stepwise_native_relative_actions():
    raw_stats = {
        "state": {
            "single_arm": _stats([0.0] * 5),
            "gripper": _stats([0.0]),
        },
        "action": {
            "single_arm": _stats([0.0] * 5),
            "gripper": _stats([0.0]),
        },
        "relative_action": {
            "single_arm": _stats([0.0] * 5),
        },
    }
    modality_config = {
        "state": {
            "modality_keys": ["single_arm", "gripper"],
        },
        "action": {
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

    with pytest.raises(NotImplementedError, match="cannot decode native relative actions one step at a time"):
        decode_step({TransitionKey.ACTION: torch.zeros(1, 6)})


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


def test_groot_n1_7_fallback_processors_wire_libero_transform_to_postprocessor():
    config = _groot_config()
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

    _, postprocessor = make_groot_pre_post_processors(config, dataset_stats=dataset_stats)

    action_decode_step = next(
        step for step in postprocessor.steps if isinstance(step, GrootActionUnpackUnnormalizeStep)
    )
    assert action_decode_step.libero_gripper_action is True


def test_groot_n1_7_loaded_fallback_postprocessor_honors_config_action_decode_transform(tmp_path):
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
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
    disabled_config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=None,
    )
    preprocessor, postprocessor = make_groot_pre_post_processors(
        disabled_config,
        dataset_stats=dataset_stats,
    )
    save_dir = tmp_path / "saved_fallback_processors"
    disabled_config.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    saved_postprocessor = json.loads((save_dir / "policy_postprocessor.json").read_text())
    saved_decode_config = next(
        step["config"]
        for step in saved_postprocessor["steps"]
        if step["registry_name"] == "groot_action_unpack_unnormalize_v2"
    )
    assert saved_decode_config["libero_gripper_action"] is False

    enabled_config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    )
    _, loaded_postprocessor = make_pre_post_processors(enabled_config, pretrained_path=str(save_dir))
    action_decode_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, GrootActionUnpackUnnormalizeStep)
    )

    assert action_decode_step.libero_gripper_action is True
    output = action_decode_step({TransitionKey.ACTION: torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])})
    torch.testing.assert_close(output[TransitionKey.ACTION][0, -1], torch.tensor(1.0))


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


def test_groot_from_pretrained_rejects_caller_config_mismatch_from_local_config(tmp_path):
    model_path = tmp_path / "local-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "Gr00tN1d7"}')
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)

    # An N1.7 config paired with a legacy N1.5 base path is a mismatch and must be
    # rejected. The mismatch is detected during config validation (__post_init__),
    # so construction itself raises before from_pretrained is reached.
    with pytest.raises(ValueError, match="does not match base_model_path"):
        config = GrootConfig(
            base_model_path="nvidia/GR00T-N1.5-3B",
            input_features=input_features,
            output_features=output_features,
            device="cpu",
            use_bf16=False,
            action_decode_transform=GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
        )
        GrootPolicy.from_pretrained(model_path, config=config)


def test_groot_n1_7_processors_are_registered_lazily_without_external_gr00t():
    sys.modules.pop("gr00t", None)
    config = _groot_config()

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
            content = conversation[0]["content"]
            assert [item["type"] for item in content] == ["image", "text"]
            text = content[-1]["text"]
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
            self.conversation_content_types = []
            self.conversation_image_values = []
            self.conversation_texts = []
            self.encoded_texts = None
            self.encoded_image_values = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            assert tokenize is False
            self.add_generation_prompts.append(add_generation_prompt)
            content = conversation[0]["content"]
            self.conversation_content_types.append([item["type"] for item in content])
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
    assert fake_proc.conversation_content_types == [
        ["image", "image", "image", "image", "text"],
        ["image", "image", "image", "image", "text"],
    ]
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

    transformed = _transform_n1_7_image_for_vlm_albumentations(
        Image.fromarray(image_np),
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
    )

    expected = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_AREA)
    crop_edge = int(256 * 0.95)
    crop_start = (256 - crop_edge) // 2
    expected = expected[crop_start : crop_start + crop_edge, crop_start : crop_start + crop_edge]
    expected = cv2.resize(expected, (256, 256), interpolation=cv2.INTER_AREA)

    assert transformed.shape == (256, 256, 3)
    np.testing.assert_array_equal(np.asarray(transformed), expected)


def test_groot_n1_7_albumentations_letterbox_is_opt_in():
    pytest.importorskip("cv2", exc_type=ImportError)

    image = np.full((3, 5, 3), 255, dtype=np.uint8)

    default = _transform_n1_7_image_for_vlm_albumentations(
        image,
        image_crop_size=None,
        image_target_size=[10, 10],
        shortest_image_edge=10,
        crop_fraction=None,
    )
    letterboxed = _transform_n1_7_image_for_vlm_albumentations(
        image,
        image_crop_size=None,
        image_target_size=[10, 10],
        shortest_image_edge=10,
        crop_fraction=None,
        letter_box_transform=True,
    )

    assert default.shape == (10, 17, 3)
    assert default.min() == 255
    assert letterboxed.shape == (10, 10, 3)
    assert letterboxed.min() < 255


def test_groot_n1_7_torch_letterbox_is_opt_in():
    image = torch.full((3, 3, 5), 255, dtype=torch.uint8)

    default = _transform_n1_7_image_for_vlm_torch(
        image,
        image_crop_size=None,
        image_target_size=[10, 10],
        shortest_image_edge=10,
        crop_fraction=None,
    )
    letterboxed = _transform_n1_7_image_for_vlm_torch(
        image,
        image_crop_size=None,
        image_target_size=[10, 10],
        shortest_image_edge=10,
        crop_fraction=None,
        letter_box_transform=True,
    )

    assert tuple(default.shape) == (3, 10, 10)
    assert int(default.min()) == 255
    assert tuple(letterboxed.shape) == (3, 10, 10)
    assert int(letterboxed.min()) < 255


def test_groot_n1_7_vlm_encode_transforms_non_square_two_camera_sample_like_core_albumentations():
    cv2 = pytest.importorskip("cv2", exc_type=ImportError)

    class FakeProcessor:
        def __init__(self):
            self.images = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            content = conversation[0]["content"]
            assert [item["type"] for item in content] == ["image", "image", "text"]
            return content[-1]["text"]

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
        letter_box_transform=True,
    )

    restored = GrootN17VLMEncodeStep(**step.get_config())

    assert restored.model_name == "local-cosmos"
    assert restored.image_crop_size == [230, 230]
    assert restored.image_target_size == [256, 256]
    assert restored.shortest_image_edge == 256
    assert restored.crop_fraction == 0.95
    assert restored.use_albumentations is True
    assert restored.letter_box_transform is True


def test_groot_n1_7_processor_uses_qwen_component_assets(monkeypatch):
    pytest.importorskip("transformers")

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

    monkeypatch.setattr(processor_groot, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(processor_groot, "Qwen2VLImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(processor_groot, "Qwen3VLVideoProcessor", FakeVideoProcessor)
    monkeypatch.setattr(processor_groot, "Qwen3VLProcessor", FakeProcessor)

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
    config = _groot_config()
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
    config = _groot_config()
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


def test_groot_n1_7_relative_action_training_processors_save_native_grouped_stats(tmp_path):
    input_features, output_features = _groot_features(state_dim=6, action_dim=6)
    action_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=None,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
    )
    absolute_dataset_stats = {
        OBS_STATE: {
            "min": torch.tensor([-50.0, -60.0, -70.0, -80.0, -90.0, 0.0]),
            "max": torch.tensor([50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
        },
        ACTION: {
            "min": torch.tensor([-100.0, -110.0, -120.0, -130.0, -140.0, 0.0]),
            "max": torch.tensor([100.0, 110.0, 120.0, 130.0, 140.0, 100.0]),
        },
    }
    samples = [
        {
            OBS_STATE: torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 0.0]),
            ACTION: _native_action_chunk(
                [
                    [8.0, 17.0, 26.0, 35.0, 44.0, 0.0],
                    [12.0, 23.0, 34.0, 45.0, 56.0, 100.0],
                ]
            ),
        },
        {
            OBS_STATE: torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
            ACTION: _native_action_chunk(
                [
                    [-1.0, -2.0, -3.0, -4.0, -5.0, 25.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 75.0],
                ]
            ),
        },
    ]

    class _RelativeStatsDataset:
        meta = SimpleNamespace(
            stats=absolute_dataset_stats,
            features={ACTION: {"names": action_names}},
        )

        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    relative_dataset_stats = _make_relative_action_training_stats(
        _RelativeStatsDataset(),
        exclude_joints=["gripper"],
        action_names=action_names,
        preserve_action_horizon=True,
    )
    expected_relative_action_min_prefix = torch.tensor(
        [-2.0, -3.0, -4.0, -5.0, -6.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )
    expected_relative_action_max_prefix = torch.tensor(
        [-1.0, -2.0, -3.0, -4.0, -5.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    preprocessor, postprocessor = make_groot_pre_post_processors(
        config, dataset_stats=relative_dataset_stats, dataset_meta=_RelativeStatsDataset.meta
    )
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    preprocessor_config = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    assert not any(
        step.get("registry_name") == "relative_actions_processor" for step in preprocessor_config["steps"]
    )
    pack_entry = next(
        step
        for step in preprocessor_config["steps"]
        if step.get("registry_name") == "groot_n1_7_pack_inputs_v1"
    )
    pack_config = pack_entry["config"]
    assert pack_config["modality_config"]["action"]["modality_keys"] == ["single_arm", "gripper"]
    assert pack_config["modality_config"]["action"]["action_configs"] == [
        {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
        {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
    ]
    pack_relative_min = pack_config["raw_stats"]["relative_action"]["single_arm"]["min"]
    assert pack_relative_min[:2] == [
        [-2.0, -3.0, -4.0, -5.0, -6.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ]
    assert len(pack_relative_min) == N1_7_NATIVE_ACTION_HORIZON
    assert (
        pack_config["raw_stats"]["relative_action"]["single_arm"]["count"] == [2] * N1_7_NATIVE_ACTION_HORIZON
    )
    assert pack_config["raw_stats"]["action"]["gripper"]["min"] == [0.0]
    assert pack_config["raw_stats"]["action"]["gripper"]["max"] == [100.0]

    pack_state = load_file(tmp_path / pack_entry["state_file"])
    expected_flat_dim = N1_7_NATIVE_ACTION_HORIZON * 5 + 1
    assert pack_state[f"{ACTION}.min"].shape == (expected_flat_dim,)
    assert pack_state[f"{ACTION}.max"].shape == (expected_flat_dim,)
    torch.testing.assert_close(pack_state[f"{ACTION}.min"][:10], expected_relative_action_min_prefix)
    torch.testing.assert_close(pack_state[f"{ACTION}.max"][:10], expected_relative_action_max_prefix)
    assert pack_state[f"{ACTION}.min"][-1].item() == 0.0
    assert pack_state[f"{ACTION}.max"][-1].item() == 100.0

    postprocessor_config = json.loads((tmp_path / "policy_postprocessor.json").read_text())
    assert not any(
        step.get("registry_name") == "absolute_actions_processor" for step in postprocessor_config["steps"]
    )
    decode_entry = next(
        step
        for step in postprocessor_config["steps"]
        if step.get("registry_name") == "groot_n1_7_action_decode_v1"
    )
    decode_config = decode_entry["config"]
    assert decode_config["use_relative_action"] is True
    decode_relative_max = decode_config["raw_stats"]["relative_action"]["single_arm"]["max"]
    assert decode_relative_max[:2] == [
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    assert len(decode_relative_max) == N1_7_NATIVE_ACTION_HORIZON
    assert (
        decode_config["raw_stats"]["relative_action"]["single_arm"]["count"]
        == [2] * N1_7_NATIVE_ACTION_HORIZON
    )
    assert decode_config["raw_stats"]["action"]["gripper"]["max"] == [100.0]


def test_groot_n1_7_relative_action_processors_compute_stats_from_runtime_dataset_meta(monkeypatch, tmp_path):
    pytest.importorskip("datasets")

    input_features, output_features = _groot_features(state_dim=6, action_dim=6)
    action_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=None,
        chunk_size=2,
        n_action_steps=2,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
    )
    absolute_dataset_stats = {
        OBS_STATE: {
            "min": torch.tensor([-50.0, -60.0, -70.0, -80.0, -90.0, 0.0]),
            "max": torch.tensor([50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
        },
        ACTION: {
            "min": torch.tensor([-100.0, -110.0, -120.0, -130.0, -140.0, 0.0]),
            "max": torch.tensor([100.0, 110.0, 120.0, 130.0, 140.0, 100.0]),
        },
    }
    samples = [
        {
            OBS_STATE: torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 0.0]),
            ACTION: _native_action_chunk(
                [
                    [8.0, 17.0, 26.0, 35.0, 44.0, 0.0],
                    [12.0, 23.0, 34.0, 45.0, 56.0, 100.0],
                ]
            ),
        },
        {
            OBS_STATE: torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
            ACTION: _native_action_chunk(
                [
                    [-1.0, -2.0, -3.0, -4.0, -5.0, 25.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 75.0],
                ]
            ),
        },
    ]
    runtime_meta = SimpleNamespace(
        repo_id="local/relative",
        root=tmp_path,
        revision="main",
        fps=30,
        stats=absolute_dataset_stats,
        features={ACTION: {"names": action_names}},
    )

    class _RelativeStatsDataset:
        meta = runtime_meta

        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    def _fake_lerobot_dataset(repo_id, **kwargs):
        assert repo_id == runtime_meta.repo_id
        assert kwargs["root"] == runtime_meta.root
        assert kwargs["revision"] == runtime_meta.revision
        assert kwargs["download_videos"] is False
        assert kwargs["delta_timestamps"][ACTION] == [
            index / runtime_meta.fps for index in range(N1_7_NATIVE_ACTION_HORIZON)
        ]
        return _RelativeStatsDataset()

    monkeypatch.setattr("lerobot.policies.groot.processor_groot.LeRobotDataset", _fake_lerobot_dataset)
    config._runtime_dataset_meta = runtime_meta

    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=absolute_dataset_stats)

    assert not any(isinstance(step, RelativeActionsProcessorStep) for step in preprocessor.steps)
    assert isinstance(postprocessor.steps[0], GrootN17ActionDecodeStep)
    pack_step = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    assert pack_step.action_horizon == N1_7_NATIVE_ACTION_HORIZON
    assert pack_step.valid_action_horizon == 2
    pack_relative_min = pack_step.raw_stats["relative_action"]["single_arm"]["min"]
    assert pack_relative_min[:2] == [
        [-2.0, -3.0, -4.0, -5.0, -6.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ]
    assert len(pack_relative_min) == N1_7_NATIVE_ACTION_HORIZON
    assert pack_step.raw_stats["relative_action"]["single_arm"]["count"] == [2] * N1_7_NATIVE_ACTION_HORIZON
    assert pack_step.raw_stats["action"]["gripper"]["max"] == [100.0]


def test_groot_n1_7_generated_relative_stats_match_oss_gr00t_reference_numbers():
    input_features, output_features = _groot_features(state_dim=6, action_dim=6)
    action_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=None,
        chunk_size=3,
        n_action_steps=3,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
    )
    absolute_dataset_stats = {
        OBS_STATE: {
            "min": torch.tensor([-20.0, -30.0, -40.0, -50.0, -60.0, 0.0]),
            "max": torch.tensor([80.0, 70.0, 60.0, 50.0, 40.0, 100.0]),
            "mean": torch.tensor([30.0, 20.0, 10.0, 0.0, -10.0, 50.0]),
            "std": torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            "q01": torch.tensor([-10.0, -20.0, -30.0, -40.0, -50.0, 10.0]),
            "q99": torch.tensor([70.0, 60.0, 50.0, 40.0, 30.0, 90.0]),
        },
        ACTION: {
            "min": torch.tensor([-5.0, -20.0, 0.0, -25.0, 10.0, 20.0]),
            "max": torch.tensor([20.0, 30.0, 45.0, 60.0, 70.0, 90.0]),
            "mean": torch.tensor([5.0, 5.0, 20.0, 20.0, 40.0, 55.0]),
            "std": torch.tensor([5.0, 10.0, 10.0, 20.0, 20.0, 25.0]),
            "q01": torch.tensor([-4.0, -19.0, 1.0, -24.0, 11.0, 20.0]),
            "q99": torch.tensor([19.0, 29.0, 44.0, 59.0, 69.0, 90.0]),
        },
    }
    state_a = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 25.0])
    state_b = torch.tensor([0.0, -10.0, 10.0, -20.0, 20.0, 75.0])
    action_a = _native_action_chunk(
        [
            [11.0, 22.0, 33.0, 44.0, 55.0, 20.0],
            [12.0, 24.0, 36.0, 48.0, 60.0, 80.0],
            [13.0, 26.0, 39.0, 52.0, 65.0, 90.0],
        ]
    )
    action_b = _native_action_chunk(
        [
            [-1.0, -8.0, 13.0, -16.0, 25.0, 30.0],
            [-2.0, -6.0, 16.0, -12.0, 30.0, 40.0],
            [-3.0, -4.0, 19.0, -8.0, 35.0, 50.0],
        ]
    )
    samples = [
        {OBS_STATE: state_a, ACTION: action_a},
        {OBS_STATE: state_b, ACTION: action_b},
    ]

    class _Dataset:
        meta = SimpleNamespace(
            stats=absolute_dataset_stats,
            features={ACTION: {"names": action_names}},
        )

        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    relative_dataset_stats = _make_relative_action_training_stats(
        _Dataset(),
        exclude_joints=["gripper"],
        action_names=action_names,
        preserve_action_horizon=True,
    )

    # Static reference values from OSS GR00T's JointActionChunk.relative_chunking +
    # calculate_stats_for_key path: stats are computed per chunk timestep, not
    # flattened over all timesteps.
    oss_arm_min = torch.tensor(
        [
            [-1.0, 2.0, 3.0, 4.0, 5.0],
            [-2.0, 4.0, 6.0, 8.0, 10.0],
            [-3.0, 6.0, 9.0, 12.0, 15.0],
        ]
    )
    oss_arm_max = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [3.0, 6.0, 9.0, 12.0, 15.0],
        ]
    )
    oss_arm_mean = torch.tensor(
        [
            [0.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 4.0, 6.0, 8.0, 10.0],
            [0.0, 6.0, 9.0, 12.0, 15.0],
        ]
    )
    oss_arm_std = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    oss_arm_q01 = torch.tensor(
        [
            [-0.98, 2.0, 3.0, 4.0, 5.0],
            [-1.96, 4.0, 6.0, 8.0, 10.0],
            [-2.94, 6.0, 9.0, 12.0, 15.0],
        ]
    )
    oss_arm_q99 = torch.tensor(
        [
            [0.98, 2.0, 3.0, 4.0, 5.0],
            [1.96, 4.0, 6.0, 8.0, 10.0],
            [2.94, 6.0, 9.0, 12.0, 15.0],
        ]
    )

    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["min"][:3, :5]), oss_arm_min)
    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["max"][:3, :5]), oss_arm_max)
    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["mean"][:3, :5]), oss_arm_mean)
    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["std"][:3, :5]), oss_arm_std)
    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["q01"][:3, :5]), oss_arm_q01)
    torch.testing.assert_close(torch.as_tensor(relative_dataset_stats[ACTION]["q99"][:3, :5]), oss_arm_q99)
    assert torch.as_tensor(relative_dataset_stats[ACTION]["min"]).shape[0] == N1_7_NATIVE_ACTION_HORIZON

    preprocessor, postprocessor = make_groot_pre_post_processors(
        config,
        dataset_stats=relative_dataset_stats,
        dataset_meta=_Dataset.meta,
    )
    pack_step = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    decode_step = next(step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep))

    assert pack_step.use_percentiles is True
    pack_relative_min = torch.as_tensor(pack_step.raw_stats["relative_action"]["single_arm"]["min"])
    pack_relative_q99 = torch.as_tensor(pack_step.raw_stats["relative_action"]["single_arm"]["q99"])
    assert pack_relative_min.shape == (N1_7_NATIVE_ACTION_HORIZON, 5)
    assert pack_relative_q99.shape == (N1_7_NATIVE_ACTION_HORIZON, 5)
    torch.testing.assert_close(pack_relative_min[:3], oss_arm_min)
    torch.testing.assert_close(pack_relative_q99[:3], oss_arm_q99)
    assert pack_step.stats[ACTION]["min"][:15] == pytest.approx(oss_arm_min.flatten().tolist())
    assert pack_step.stats[ACTION]["max"][:15] == pytest.approx(oss_arm_max.flatten().tolist())
    assert pack_step.stats[ACTION]["min"][-1] == pytest.approx(20.0)
    assert pack_step.stats[ACTION]["max"][-1] == pytest.approx(90.0)

    packed = pack_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state_a.unsqueeze(0)},
            TransitionKey.ACTION: action_a.unsqueeze(0),
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move the vial"]},
        }
    )
    expected_normalized = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 5.0 / 7.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(packed[TransitionKey.ACTION][0, :3, :6], expected_normalized)

    decoded = decode_step({TransitionKey.ACTION: packed[TransitionKey.ACTION]})
    assert decoded[TransitionKey.ACTION].shape == (1, N1_7_NATIVE_ACTION_HORIZON, 6)
    torch.testing.assert_close(
        decoded[TransitionKey.ACTION][:, :3],
        action_a.unsqueeze(0)[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )


def test_groot_n1_7_relative_action_stats_skip_padded_tail_chunks():
    samples = [
        {
            OBS_STATE: torch.tensor([10.0, 100.0]),
            ACTION: torch.tensor([[11.0, 101.0], [12.0, 102.0], [13.0, 103.0]]),
            f"{ACTION}_is_pad": torch.tensor([False, False, False]),
        },
        {
            OBS_STATE: torch.tensor([20.0, 200.0]),
            ACTION: torch.tensor([[18.0, 198.0], [16.0, 196.0], [14.0, 194.0]]),
            f"{ACTION}_is_pad": torch.tensor([False, False, False]),
        },
        {
            OBS_STATE: torch.tensor([0.0, 0.0]),
            ACTION: torch.tensor([[999.0, 999.0], [888.0, 888.0], [777.0, 777.0]]),
            f"{ACTION}_is_pad": torch.tensor([False, False, True]),
        },
    ]

    class _Dataset:
        meta = SimpleNamespace(stats={})

        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    relative_dataset_stats = _make_relative_action_training_stats(
        _Dataset(),
        exclude_joints=[],
        action_names=None,
        preserve_action_horizon=True,
    )

    torch.testing.assert_close(
        torch.as_tensor(relative_dataset_stats[ACTION]["count"]),
        torch.tensor([2, 2, 2]),
    )
    torch.testing.assert_close(
        torch.as_tensor(relative_dataset_stats[ACTION]["min"]),
        torch.tensor([[-2.0, -2.0], [-4.0, -4.0], [-6.0, -6.0]]),
    )
    torch.testing.assert_close(
        torch.as_tensor(relative_dataset_stats[ACTION]["max"]),
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )


def test_groot_policy_selects_n1_7_model_class(monkeypatch):
    pytest.importorskip("transformers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    called = {}

    def fake_from_pretrained(cls, **kwargs):
        called.update(kwargs)
        return _DummyGrootModel()

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(fake_from_pretrained))

    policy = GrootPolicy(_groot_config())

    assert called["pretrained_model_name_or_path"] == GROOT_N1_7_BASE_MODEL
    assert isinstance(policy._groot_model, _DummyGrootModel)


def test_groot_policy_forwards_n1_7_qwen_inputs(monkeypatch):
    pytest.importorskip("transformers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    dummy_model = _DummyGrootModel()
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: dummy_model))
    policy = GrootPolicy(_groot_config())

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


def test_groot_select_action_rejects_relative_action_policies():
    policy = object.__new__(GrootPolicy)
    object.__setattr__(policy, "config", SimpleNamespace(use_relative_actions=True))

    with pytest.raises(NotImplementedError, match="select_action does not support relative-action policies"):
        policy.select_action({})


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
