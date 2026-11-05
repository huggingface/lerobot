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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE  # noqa: E402

from lerobot.configs.rewards import RewardModelConfig  # noqa: E402
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

    def forward(self, **kwargs):  # noqa: ARG002
        return SimpleNamespace(value=SimpleNamespace(pred_value=self.pred_value))


def _patch_checkpoint_load(monkeypatch, pred_value: torch.Tensor | None = None) -> None:
    from lerobot.rewards.rynnvalue import modeling_rynnvalue

    class _FakeCheckpointModel(_FakeRynnValueModel):
        def __init__(self, config=None) -> None:
            super().__init__(pred_value=pred_value, config=config)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
            return cls()

    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _FakeNativeConfig()),
    )
    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_dict",
        classmethod(lambda cls, values: _FakeNativeConfig(**values)),
    )
    monkeypatch.setattr(modeling_rynnvalue, "RynnValueLangModel", _FakeCheckpointModel)


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
    with pytest.raises(ValueError, match="reward_output"):
        RynnValueConfig(device="cpu", reward_output="progress")
    with pytest.raises(ValueError, match="pred_slot_isolated_eager"):
        RynnValueConfig(device="cpu", attn_implementation="sdpa")


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


def test_bro_head_inherits_active_model_dtype():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        head = BroValueHead(input_dim=12, output_dim=256, hidden_dims=16, depth=2)
    finally:
        torch.set_default_dtype(previous_dtype)

    hidden_states = torch.zeros(3, 12, dtype=torch.bfloat16)
    assert next(head.parameters()).dtype == torch.bfloat16
    assert head(hidden_states).dtype == torch.bfloat16


def test_grouped_query_tokens_are_concatenated():
    hidden = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    concatenated, prefix = RynnValueLangModel._concat_slot_tokens(
        hidden, hidden.shape[:-1], repeat=2, token_name="<value>"
    )
    assert prefix == torch.Size([2])
    assert concatenated.shape == (2, 12)
    assert torch.equal(concatenated[0], hidden[:2].reshape(-1))


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


def test_predict_remaining_time_uses_value_tokens_to_split_uneven_samples():
    native_config = _FakeNativeConfig()
    native_config.value_token_id = 99
    native_config.value_token_repeat = 2
    native_model = _FakeRynnValueModel(
        pred_value=torch.tensor([[8.0, 5.0, 1.0]]),
        config=native_config,
    )
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"), model=native_model)
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
    assert torch.equal(prediction.remaining_time_s, torch.tensor([2.0, 1.0]))
    assert model.is_trainable is False
    assert not hasattr(model, "compute_reward")


def test_legacy_reward_output_does_not_change_native_prediction(monkeypatch):
    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu", reward_output="potential"))
    prediction = model.predict_remaining_time(_encoded_batch())
    assert torch.equal(prediction.remaining_time_s, torch.tensor([2.0, 1.0]))


def test_predict_remaining_time_requires_encoded_inputs(monkeypatch):
    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(RynnValueConfig(device="cpu"))
    with pytest.raises(KeyError, match="observation.rynnvalue.input_ids"):
        model.predict_remaining_time({})


def test_embedded_model_config_avoids_upstream_checkpoint_load(monkeypatch):
    from lerobot.rewards.rynnvalue import modeling_rynnvalue

    source_load_called = False

    def fail_source_load(*args, **kwargs):  # noqa: ARG001
        nonlocal source_load_called
        source_load_called = True
        raise AssertionError("upstream checkpoint should not be loaded")

    class _ConstructedModel(_FakeRynnValueModel):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config

        from_pretrained = classmethod(fail_source_load)

    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_dict",
        classmethod(lambda cls, config: SimpleNamespace(**config)),
    )
    monkeypatch.setattr(modeling_rynnvalue, "RynnValueLangModel", _ConstructedModel)

    model = RynnValueRewardModel(
        RynnValueConfig(
            device="cpu",
            torch_dtype="float32",
            model_config={"hidden_size": 16},
        )
    )
    assert model.model.config.hidden_size == 16
    assert source_load_called is False


def test_lerobot_checkpoint_without_embedded_model_config_fails_before_upstream_load(monkeypatch):
    from lerobot.rewards.rynnvalue import modeling_rynnvalue

    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: pytest.fail("upstream config load should not be attempted")),
    )
    with pytest.raises(ValueError, match="missing `model_config`"):
        RynnValueRewardModel(
            RynnValueConfig(
                device="cpu",
                pretrained_path="/tmp/legacy-rynnvalue",
            )
        )


def test_save_and_load_lerobot_checkpoint(monkeypatch, tmp_path):
    _patch_checkpoint_load(monkeypatch)
    model = RynnValueRewardModel(
        RynnValueConfig(
            device="cpu",
            model_id="Alibaba-DAMO-Academy/RynnValue-8B",
            model_revision="abc123",
        )
    )
    model.save_pretrained(tmp_path)
    assert (tmp_path / CONFIG_NAME).exists()
    assert (tmp_path / SAFETENSORS_SINGLE_FILE).exists()

    loaded = RynnValueRewardModel.from_pretrained(tmp_path)
    assert loaded.config.model_id == "Alibaba-DAMO-Academy/RynnValue-8B"
    assert loaded.config.model_revision == "abc123"
    assert loaded.config.model_config == {"model_type": "rynn_value_lang", "hidden_size": 16}
    assert loaded.config.pretrained_path == str(tmp_path)


def test_conversion_writes_self_contained_lerobot_checkpoint(monkeypatch, tmp_path):
    from lerobot.rewards.rynnvalue import convert_rynnvalue_checkpoint as conversion

    class _SourceConfig:
        _attn_implementation = None

        @staticmethod
        def to_dict():
            return {"model_type": "rynn_value_lang", "hidden_size": 16}

    class _Processor:
        @staticmethod
        def save_pretrained(output_dir):
            (output_dir / "processor_config.json").write_text("{}")

    monkeypatch.setattr(
        conversion.RynnValueLangConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _SourceConfig()),
    )
    monkeypatch.setattr(
        conversion.RynnValueLangModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _FakeRynnValueModel()),
    )
    monkeypatch.setattr(
        conversion.RynnValueLangProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _Processor()),
    )

    conversion.convert_rynnvalue_checkpoint(
        tmp_path,
        source_model_id="upstream/rynnvalue",
        revision="abc123",
        torch_dtype="float32",
    )

    assert (tmp_path / CONFIG_NAME).exists()
    assert (tmp_path / SAFETENSORS_SINGLE_FILE).exists()
    assert (tmp_path / "processor_config.json").exists()
    config = RewardModelConfig.from_pretrained(tmp_path)
    assert config.model_id == "upstream/rynnvalue"
    assert config.model_revision == "abc123"
    assert config.model_config == {"model_type": "rynn_value_lang", "hidden_size": 16}


def test_offline_checkpoint_roundtrip_through_reward_factories(monkeypatch, tmp_path):
    from lerobot.rewards.rynnvalue import modeling_rynnvalue, processor_rynnvalue

    original = RynnValueRewardModel(
        RynnValueConfig(device="cpu", torch_dtype="float32"),
        model=_FakeRynnValueModel(),
    )
    original.save_pretrained(tmp_path)

    def fail_upstream_load(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("offline LeRobot checkpoint must not load the upstream model")

    class _OfflineModel(_FakeRynnValueModel):
        def __init__(self, config) -> None:
            super().__init__(config=config)

        from_pretrained = classmethod(fail_upstream_load)

    class _OfflineProcessor:
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
        modeling_rynnvalue.RynnValueLangConfig,
        "from_pretrained",
        classmethod(fail_upstream_load),
    )
    monkeypatch.setattr(
        modeling_rynnvalue.RynnValueLangConfig,
        "from_dict",
        classmethod(lambda cls, values: _FakeNativeConfig(**values)),
    )
    monkeypatch.setattr(modeling_rynnvalue, "RynnValueLangModel", _OfflineModel)
    monkeypatch.setattr(
        processor_rynnvalue.RynnValueLangProcessor,
        "from_pretrained",
        classmethod(
            lambda cls, model_id, **kwargs: (
                processor_sources.append((model_id, kwargs.get("revision"))) or _OfflineProcessor()
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

    assert processor_sources == [(str(tmp_path), None)]
    assert prediction.remaining_time_s.shape == (1,)
    assert torch.isfinite(prediction.remaining_time_s).all()
