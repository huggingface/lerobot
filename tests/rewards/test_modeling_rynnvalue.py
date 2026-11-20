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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402

from lerobot.configs.rewards import RewardModelConfig  # noqa: E402
from lerobot.rewards import RemainingTimeOutput  # noqa: E402
from lerobot.rewards.factory import (  # noqa: E402
    get_reward_model_class,
    make_reward_model,
    make_reward_model_config,
    make_reward_pre_post_processors,
)
from lerobot.rewards.rynnvalue import RynnValueConfig  # noqa: E402
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RYNNVALUE_FEATURE_PREFIX  # noqa: E402
from lerobot.rewards.rynnvalue.modeling_rynnvalue import (  # noqa: E402
    RynnValueRewardModel,
    reduce_remaining_time,
)
from lerobot.rewards.rynnvalue.rynn_value_lang.modeling_rynn_value_lang import (  # noqa: E402
    RynnValueLangModel,
)
from lerobot.rewards.rynnvalue.rynn_value_lang.value_heads import BroValueHead  # noqa: E402
from lerobot.rewards.rynnvalue.rynn_value_lang.value_tokenizer import (  # noqa: E402
    ValueTokenizer,
    to_symexp,
    to_symlog,
)


class _FakeNativeConfig:
    _attn_implementation = None

    def __init__(self, **values) -> None:
        self.values = values or {"model_type": "rynn_value_lang", "hidden_size": 16}
        self.vision_config = SimpleNamespace(_attn_implementation=None)

    def to_dict(self):
        return self.values


class _FakeRynnValueModel(torch.nn.Module):
    def __init__(
        self,
        pred_value: torch.Tensor | None = None,
        config: _FakeNativeConfig | None = None,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = config or _FakeNativeConfig()
        self.pred_value = (
            pred_value
            if pred_value is not None
            else torch.tensor([[8.0, 2.0, 5.0, 1.0]], dtype=torch.float32)
        )

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return SimpleNamespace(value=SimpleNamespace(pred_value=self.pred_value))

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, torch.full((input_ids.shape[0], 2), 5)], dim=1)


HUB_MODEL_CONFIG = {"model_type": "rynn_value_lang", "hidden_size": 16}


def _patch_checkpoint_load(
    monkeypatch,
    pred_value: torch.Tensor | None = None,
    native_config: _FakeNativeConfig | None = None,
) -> list[str]:
    """Replace the native model builds and the Hub checkpoint; return one `"model_config"` per build."""
    from lerobot.rewards.rynnvalue import modeling_rynnvalue

    sources: list[str] = []

    class _FakeCheckpointModel(_FakeRynnValueModel):
        @classmethod
        def _from_config(cls, config, **kwargs):  # noqa: ARG003
            sources.append("model_config")
            return cls(pred_value=pred_value, config=native_config or config)

    class _FakeProcessor:
        def save_pretrained(self, directory):
            directory.mkdir(parents=True)
            (directory / "processor_config.json").write_text("{}")

    monkeypatch.setattr(
        modeling_rynnvalue,
        "RewardModelConfig",
        SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_config=HUB_MODEL_CONFIG)
        ),
    )
    monkeypatch.setattr(
        modeling_rynnvalue,
        "RynnValueLangProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeProcessor()),
    )
    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_dict",
        classmethod(lambda cls, values: _FakeNativeConfig(**values)),
    )
    monkeypatch.setattr(modeling_rynnvalue, "RynnValueLangModel", _FakeCheckpointModel)
    return sources


def _encoded_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        f"{RYNNVALUE_FEATURE_PREFIX}input_ids": torch.ones(batch_size, 8, dtype=torch.long),
        f"{RYNNVALUE_FEATURE_PREFIX}attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
        f"{RYNNVALUE_FEATURE_PREFIX}pixel_values": torch.zeros(batch_size, 4),
        f"{RYNNVALUE_FEATURE_PREFIX}image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
    }


def test_rynnvalue_registered_with_reward_factory():
    assert RewardModelConfig.get_choice_class("rynnvalue") is RynnValueConfig
    assert isinstance(make_reward_model_config("rynnvalue", device="cpu"), RynnValueConfig)
    assert get_reward_model_class("rynnvalue") is RynnValueRewardModel


def test_rynnvalue_config_validation():
    with pytest.raises(ValueError, match="max_frames"):
        RynnValueConfig(device="cpu", max_frames=0)
    with pytest.raises(ValueError, match="max_image_side"):
        RynnValueConfig(device="cpu", max_image_side=0)
    with pytest.raises(ValueError, match="inference_fps must be > 0"):
        RynnValueConfig(device="cpu", inference_fps=0)
    with pytest.raises(ValueError, match="pretrained_path"):
        RynnValueConfig(device="cpu", pretrained_path=None)


@pytest.mark.parametrize("repo_id", ["lerobot/RynnValue-4B", "lerobot/RynnValue-8B"])
def test_published_rynnvalue_configs_load(repo_id: str) -> None:
    # The model tests use fake configs, so only this catches a published config the class rejects.
    try:
        config_file = hf_hub_download(repo_id, CONFIG_NAME)
    except (HfHubHTTPError, OSError) as error:
        pytest.skip(f"Could not download {repo_id}: {error}")

    config = RewardModelConfig.from_pretrained(Path(config_file).parent)

    assert isinstance(config, RynnValueConfig)


def test_value_tokenizer_two_hot_symlog_roundtrip():
    tokenizer = ValueTokenizer(
        bins=256,
        min_value=-256,
        max_value=256,
        forward_transform=to_symlog,
        inverse_transform=to_symexp,
        support_transform="symlog",
    )
    values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    target = tokenizer.encode(values)
    decoded = tokenizer.decode_from_bins(torch.log(target.clamp_min(1e-8)))
    assert target.shape == (5, 256)
    assert torch.allclose(target.sum(dim=-1), torch.ones(5))
    assert torch.allclose(decoded, values, atol=0.2)


def test_bro_head_preserves_prefix_shape():
    head = BroValueHead(input_dim=12, output_dim=256, hidden_dims=16, depth=2)
    assert head(torch.zeros(3, 12)).shape == (3, 256)


def test_bro_head_runs_in_the_model_dtype_once_cast(monkeypatch):
    head = BroValueHead(input_dim=12, output_dim=256, hidden_dims=16, depth=2)
    assert next(head.parameters()).dtype == torch.float32

    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu", torch_dtype="bfloat16"))
    assert next(model.model.parameters()).dtype == torch.bfloat16

    head = head.to(dtype=torch.bfloat16)
    assert head(torch.zeros(3, 12, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_grouped_query_tokens_are_concatenated():
    model = RynnValueLangModel.__new__(RynnValueLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(value_token_id=7, value_token_repeat=2)
    model.value_heads = torch.nn.ModuleList([torch.nn.Identity()])
    model.value_tokenizer = ValueTokenizer(bins=12, min_value=0.0, max_value=1.0)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)

    output = model._compute_value_outputs(hidden, torch.tensor([[7, 7, 7, 7]]), value=None)

    assert output.logits.shape == (1, 2, 12)
    assert torch.equal(output.logits[0, 0], hidden[0, :2].reshape(-1))


def test_prediction_slot_ids_isolate_repeated_groups():
    model = RynnValueLangModel.__new__(RynnValueLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(value_token_id=7, relative_value_token_id=8)
    input_ids = torch.tensor([[1, 7, 7, 2, 8, 8, 7, 7, 3]])
    extras, cached = model._build_pred_slot_extras(input_ids)
    assert extras["is_pred_key"].tolist() == [[False, True, True, False, True, True, True, True, False]]
    assert extras["pred_slot_id"].tolist() == [[-1, 0, 0, -1, 1, 1, 2, 2, -1]]
    assert torch.equal(cached, extras["is_pred_key"])


def test_reduce_remaining_time_averages_heads_and_selects_final_slot():
    predictions = torch.tensor(
        [
            [9.0, 3.0, 8.0, 2.0],
            [7.0, 1.0, 6.0, 0.0],
        ]
    )
    assert torch.equal(reduce_remaining_time(predictions, batch_size=2), torch.tensor([2.0, 1.0]))


def test_reduce_remaining_time_supports_uneven_sample_slot_counts():
    predictions = torch.tensor(
        [
            [9.0, 8.0, 3.0],
            [7.0, 6.0, 1.0],
        ]
    )
    remaining_time = reduce_remaining_time(predictions, batch_size=2, slot_counts=[1, 2])
    assert torch.equal(remaining_time, torch.tensor([8.0, 2.0]))


def test_predict_remaining_time_uses_value_tokens_to_split_uneven_samples(monkeypatch):
    native_config = _FakeNativeConfig()
    native_config.value_token_id = 99
    native_config.value_token_repeat = 2
    _patch_checkpoint_load(
        monkeypatch, pred_value=torch.tensor([[8.0, 5.0, 1.0]]), native_config=native_config
    )
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    batch = _encoded_batch(batch_size=2)
    batch[f"{RYNNVALUE_FEATURE_PREFIX}input_ids"] = torch.tensor(
        [
            [99, 99, 0, 0],
            [99, 99, 99, 99],
        ]
    )

    prediction = model.predict_remaining_time(batch)
    assert torch.equal(prediction.remaining_time_s, torch.tensor([8.0, 1.0]))


def test_predict_remaining_time_returns_native_seconds(monkeypatch):
    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    prediction = model.predict_remaining_time(_encoded_batch())
    assert isinstance(prediction, RemainingTimeOutput)
    assert torch.equal(prediction.remaining_time_s, torch.tensor([2.0, 1.0]))
    assert model.model.forward_kwargs["use_cache"] is False
    assert model.is_trainable is False
    with pytest.raises(NotImplementedError, match="predict_remaining_time"):
        model.compute_reward({})


def test_value_slot_attention_applies_to_the_language_model_only(monkeypatch):
    _patch_checkpoint_load(monkeypatch)
    native_config = RynnValueRewardModel(RynnValueConfig(device="cpu")).model.config
    assert native_config._attn_implementation == "pred_slot_isolated_eager"
    assert native_config.vision_config._attn_implementation == "eager"


def test_predict_remaining_time_requires_encoded_inputs(monkeypatch):
    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    with pytest.raises(KeyError, match="observation.rynnvalue.input_ids"):
        model.predict_remaining_time({})


def test_predict_success_parses_the_generated_analysis(monkeypatch):
    from lerobot.rewards.rynnvalue import modeling_rynnvalue

    _patch_checkpoint_load(monkeypatch)
    decoded = []
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 99,
        batch_decode=lambda sequences, skip_special_tokens: (
            decoded.append(sequences)
            or [
                "- Video Description: The arm picks up the cube.\n- Match: Yes\n- Success: no",
                "- Match: No",
            ]
        ),
    )
    loaded = []
    monkeypatch.setattr(
        modeling_rynnvalue,
        "RynnValueLangProcessor",
        SimpleNamespace(
            from_pretrained=lambda path, **kwargs: (
                loaded.append((path, kwargs["subfolder"])) or SimpleNamespace(tokenizer=tokenizer)
            )
        ),
    )
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    batch = {
        f"{RYNNVALUE_FEATURE_PREFIX}input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]]),
        f"{RYNNVALUE_FEATURE_PREFIX}attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        f"{RYNNVALUE_FEATURE_PREFIX}pixel_values": torch.zeros(2, 4),
        f"{RYNNVALUE_FEATURE_PREFIX}image_grid_thw": torch.ones(2, 3, dtype=torch.long),
    }

    output = model.predict_success(batch)

    torch.testing.assert_close(output.success, torch.tensor([0.0, float("nan")]), equal_nan=True)
    torch.testing.assert_close(output.match, torch.tensor([1.0, 0.0]))
    assert output.description == ["The arm picks up the cube.", None]
    assert loaded == [("lerobot/RynnValue-4B", "processor")]
    generate_kwargs = model.model.generate_kwargs
    assert generate_kwargs["input_ids"].tolist() == [[1, 2, 3, 4], [0, 0, 1, 2]]
    assert generate_kwargs["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert generate_kwargs["max_new_tokens"] == 128
    assert generate_kwargs["do_sample"] is False
    assert generate_kwargs["eos_token_id"] == generate_kwargs["pad_token_id"] == 99
    assert decoded[0].tolist() == [[5, 5], [5, 5]]


def test_default_config_builds_the_empty_architecture_for_the_lerobot_checkpoint(monkeypatch):
    sources = _patch_checkpoint_load(monkeypatch)
    config = RynnValueConfig(device="cpu")
    RynnValueRewardModel(config)

    assert config.pretrained_path == "lerobot/RynnValue-4B"
    assert config.model_config == HUB_MODEL_CONFIG
    assert sources == ["model_config"]


def test_lerobot_checkpoint_rebuilds_from_its_embedded_model_config(monkeypatch, tmp_path):
    sources = _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    model.save_pretrained(tmp_path)
    assert (tmp_path / CONFIG_NAME).exists()
    assert (tmp_path / SAFETENSORS_SINGLE_FILE).exists()
    assert (tmp_path / "processor" / "processor_config.json").exists()

    loaded = RynnValueRewardModel.from_pretrained(tmp_path)
    assert loaded.config.model_config == HUB_MODEL_CONFIG
    assert sources == ["model_config", "model_config"]


def test_checkpoint_roundtrip_through_reward_factories(monkeypatch, tmp_path):
    from lerobot.rewards.rynnvalue import processor_rynnvalue

    model_sources = _patch_checkpoint_load(monkeypatch)
    RynnValueRewardModel(RynnValueConfig(device="cpu")).save_pretrained(tmp_path)
    model_sources.clear()

    class _FakeProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        use_meta = False

        def process_episode(self, instruction, images, **kwargs):  # noqa: ARG002
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(1, len(images), 4),
                "image_grid_thw": torch.ones(1, len(images), 3, dtype=torch.long),
            }

    processor_sources = []
    monkeypatch.setattr(
        processor_rynnvalue.RynnValueLangProcessor,
        "from_pretrained",
        classmethod(
            lambda cls, path, **kwargs: (
                processor_sources.append((path, kwargs.get("subfolder"))) or _FakeProcessor()
            )
        ),
    )

    config = RewardModelConfig.from_pretrained(tmp_path)
    config.pretrained_path = str(tmp_path)
    model = make_reward_model(config)
    preprocessor, _ = make_reward_pre_post_processors(config)
    encoded = preprocessor(
        {
            config.image_key: torch.zeros(3, 8, 8),
            config.task_key: "pick up the cube",
        }
    )
    prediction = model.predict_remaining_time(encoded)

    assert model_sources == ["model_config"]
    assert processor_sources == [(str(tmp_path), "processor")]
    assert prediction.remaining_time_s.shape == (1,)
    assert torch.isfinite(prediction.remaining_time_s).all()
