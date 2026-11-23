# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""MolmoAct2 model and policy on a tiny random Molmo2 (see `conftest.py`)."""

import logging
import re
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

pytest.importorskip("transformers", reason="transformers is required (install lerobot[molmoact2])")

from transformers.models.molmo2.modeling_molmo2 import Molmo2RMSNorm  # noqa: E402

from lerobot.policies.molmoact2.modeling_molmoact2 import (  # noqa: E402
    ActionExpert,
    MolmoAct2Policy,
    _attention_bias,
    _decoder_layer,
    _get_molmo2_config,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    _ACTION_END_TOKEN_ID,
    _ACTION_START_TOKEN_ID,
    _ACTION_TOKEN_START_ID,
    EOS_TOKEN_ID,
    UniversalActionProcessor,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


def make_batch(sizes, image_patch_id: int, action_span: bool = True) -> dict[str, Tensor]:
    """Build a preprocessed batch in the layout of `MolmoAct2PackInputsProcessorStep`, left padded.

    Each row is BOS (`<|im_end|>`), `<im_start><im_patch><im_end>` for each camera, text tokens (6, then 3) and,
    with `action_span`, `<action_start>`, three action tokens, `<action_end>` and `<|im_end|>`, which are the labels.
    The image start and end tokens are extra-vocabulary rows of the token embedding. The action expert attends to
    every valid token; the masking of the pack step is tested with the processor.
    """
    generator = torch.Generator().manual_seed(0)
    image_tokens = [sizes.base_vocab_size, image_patch_id, sizes.base_vocab_size + 1] * len(sizes.cameras)
    rows, span_lengths = [], []
    for length in (6, 3):
        text = torch.randint(0, 1000, (length,), generator=generator).tolist()
        span = []
        if action_span:
            bins = torch.randint(0, 100, (3,), generator=generator) + _ACTION_TOKEN_START_ID
            span = [_ACTION_START_TOKEN_ID, *bins.tolist(), _ACTION_END_TOKEN_ID, EOS_TOKEN_ID]
        rows.append([EOS_TOKEN_ID, *image_tokens, *text, *span])
        span_lengths.append(len(span))
    batch_size, width = len(rows), max(len(row) for row in rows)
    input_ids = torch.zeros(batch_size, width, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, width, dtype=torch.long)
    labels = torch.full((batch_size, width), -100)
    for index, (row, span_length) in enumerate(zip(rows, span_lengths, strict=True)):
        input_ids[index, width - len(row) :] = torch.tensor(row)
        attention_mask[index, width - len(row) :] = 1
        if span_length:
            labels[index, width - span_length :] = input_ids[index, width - span_length :]
    num_crops = batch_size * len(sizes.cameras)
    action = torch.zeros(batch_size, sizes.chunk_size, sizes.max_action_dim)
    action[..., : sizes.action_dim] = (
        torch.rand(batch_size, sizes.chunk_size, sizes.action_dim, generator=generator) * 2 - 1
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
        "encoder_attention_mask": attention_mask.bool(),
        "mm_token_type_ids": torch.isin(input_ids, torch.tensor(image_tokens)).long() * attention_mask,
        "pixel_values": torch.randn(
            num_crops, sizes.patches_per_crop, sizes.pixels_per_patch, generator=generator
        ),
        "image_token_pooling": torch.arange(num_crops * sizes.patches_per_crop).view(
            num_crops, sizes.patches_per_crop
        ),
        "image_grids": torch.tensor([[1, 1, 0, 0]] * num_crops),
        ACTION: action,
        "action_dim_is_pad": (torch.arange(sizes.max_action_dim) >= sizes.action_dim).repeat(batch_size, 1),
    }
    if action_span:
        batch["labels"] = labels
    return batch


def row_batch(
    sizes, batch: dict[str, Tensor], row: int, extra_tokens: Sequence[int] = ()
) -> dict[str, Tensor]:
    """Return the prompt of one row of `make_batch` without its padding, followed by `extra_tokens`."""
    valid = batch["attention_mask"][row].bool()
    extra = torch.tensor(extra_tokens, dtype=torch.long)
    input_ids = torch.cat((batch["input_ids"][row, valid], extra))[None]
    image_tokens = torch.cat((batch["mm_token_type_ids"][row, valid], torch.zeros_like(extra)))
    crops = slice(row * len(sizes.cameras), (row + 1) * len(sizes.cameras))
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1])[None],
        "encoder_attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "mm_token_type_ids": image_tokens[None],
        "pixel_values": batch["pixel_values"][crops],
        "image_token_pooling": batch["image_token_pooling"][crops] - crops.start * sizes.patches_per_crop,
        "image_grids": batch["image_grids"][crops],
        "action_dim_is_pad": batch["action_dim_is_pad"][row : row + 1],
    }


@pytest.fixture
def policy(make_policy):
    return make_policy()


@pytest.fixture
def batch(sizes, policy):
    return make_batch(sizes, policy.model.vlm.config.image_patch_id)


def _flow_inputs(sizes) -> dict[str, Tensor]:
    """Fixed flow-matching noise and times for a batch of 2 with 2 flow samples each."""
    generator = torch.Generator().manual_seed(1)
    return {
        "noise": torch.randn(2, 2, sizes.chunk_size, sizes.max_action_dim, generator=generator),
        "time": torch.rand(2, 2, generator=generator),
    }


def _noise(sizes) -> Tensor:
    return torch.randn(2, sizes.chunk_size, sizes.max_action_dim, generator=torch.Generator().manual_seed(2))


def _causal_prefill(model, batch):
    """The decoder loop of `MolmoAct2Model.forward`: causal attention, image tokens included."""
    hidden = model.embed_inputs(batch)
    cos, sin = model._rope(batch["position_ids"], hidden.dtype)
    bias = _attention_bias(batch["attention_mask"], None, hidden.dtype)
    key_values = []
    for layer in model.layers:
        hidden, key, value = _decoder_layer(layer, hidden, cos, sin, bias, dropout=0.0)
        key_values.append((key, value))
    return model._final_norm(hidden), key_values


def test_get_molmo2_config_is_the_molmo2_4b_layout(sizes):
    config = _get_molmo2_config()
    text = config.text_config

    assert (text.hidden_size, text.num_hidden_layers, text.vocab_size) == (2560, 36, sizes.base_vocab_size)
    assert text.additional_vocab_size == 128
    # The action expert's default context projection reads 8 KV heads of 128 dims.
    assert text.num_key_value_heads * text.head_dim == 1024
    # The decoder loop runs pre-norm layers with qwen3-style q/k norm.
    assert text.qk_norm_type == "qwen3" and not text.norm_after
    assert config.image_patch_id == sizes.base_vocab_size + 2


@pytest.mark.parametrize("image_block", [False, True], ids=["causal", "image_block"])
def test_decoder_loop_matches_transformers_molmo2(policy, batch, image_block):
    """Our decoder loop must stay equal to `Molmo2Model.forward`: this guards against transformers changes."""
    model = policy.model.eval()
    with torch.no_grad():
        reference = model.vlm.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            pixel_values=batch["pixel_values"],
            image_token_pooling=batch["image_token_pooling"],
            image_grids=batch["image_grids"],
            mm_token_type_ids=batch["mm_token_type_ids"] if image_block else None,
            use_cache=True,
        )
        causal = _causal_prefill(model, batch)
        hidden, key_values = model.prefill(batch) if image_block else causal

    valid = batch["attention_mask"].bool()
    torch.testing.assert_close(hidden[valid], reference.last_hidden_state[valid], rtol=1e-5, atol=1e-5)
    if image_block:  # the image tokens of the first camera see those of the second one
        assert not torch.allclose(hidden[valid], causal[0][valid], atol=1e-3)
    for (key, value), layer_cache in zip(key_values, reference.past_key_values.layers, strict=True):
        torch.testing.assert_close(key.transpose(1, 2)[valid], layer_cache.keys.transpose(1, 2)[valid])
        torch.testing.assert_close(value.transpose(1, 2)[valid], layer_cache.values.transpose(1, 2)[valid])


def test_prefill_of_a_left_padded_row_matches_the_row_alone(sizes, policy, batch):
    model = policy.model.eval()
    with torch.no_grad():
        hidden, key_values = model.prefill(batch)
        single, single_key_values = model.prefill(row_batch(sizes, batch, 1))

    length = single.shape[1]
    assert batch["attention_mask"][1, 0] == 0
    torch.testing.assert_close(hidden[1, -length:], single[0], rtol=1e-5, atol=1e-5)
    # Attention only sees relative positions; the post-RoPE keys read by the action expert see absolute ones.
    for (key, _), (single_key, _) in zip(key_values, single_key_values, strict=True):
        torch.testing.assert_close(key[1, :, -length:], single_key[0], rtol=1e-5, atol=1e-5)


def test_action_expert_has_the_released_588_key_layout():
    with torch.device("meta"):
        state = ActionExpert(num_layers=36, llm_kv_dim=1024).state_dict()

    sublayers = ["self_attn.qkv", "self_attn.out_proj", "cross_attn.q_proj", "cross_attn.out_proj"]
    sublayers += ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj", "modulation.linear"]
    sublayers = [f"blocks.{index}.{name}" for index in range(36) for name in sublayers]
    linears = [*sublayers, "time_embed.1", "time_embed.3", "action_embed", "final_layer.modulation.linear"]
    linears.append("final_layer.linear")
    expected = {f"{name}.{kind}" for name in linears for kind in ("weight", "bias")}
    expected |= {"context_k_proj.weight", "context_v_proj.weight"}
    assert len(state) == 588 and set(state) == expected
    assert state["blocks.0.self_attn.qkv.weight"].shape == (3 * 768, 768)
    assert state["blocks.0.mlp.up_proj.weight"].shape == (3072, 768)
    assert state["blocks.0.modulation.linear.weight"].shape == (9 * 768, 768)
    assert state["context_k_proj.weight"].shape == (768, 1024)
    assert state["time_embed.1.weight"].shape == (768, 256)
    assert state["final_layer.linear.weight"].shape == (32, 768)


def _expert_inputs(sizes, expert: ActionExpert):
    """Noisy actions, timesteps, and the projected keys and values of 2 LLM layers (5 tokens, `llm_kv_dim=8`)."""
    generator = torch.Generator().manual_seed(0)
    kv_states = [tuple(torch.randn(2, 5, 8, generator=generator) for _ in range(2)) for _ in range(2)]
    with torch.no_grad():
        context = expert.prepare_kv_context(kv_states)
    return (
        torch.randn(2, sizes.chunk_size, sizes.max_action_dim, generator=generator),
        torch.rand(2, generator=generator),
        context,
    )


def test_action_expert_starts_at_zero_velocity(sizes):
    expert = ActionExpert(**sizes.action_expert, num_layers=2, llm_kv_dim=8)
    noisy_actions, timesteps, context = _expert_inputs(sizes, expert)

    velocity = expert(noisy_actions, timesteps, context, torch.ones(2, 5, dtype=torch.bool))

    assert velocity.shape == noisy_actions.shape
    assert torch.count_nonzero(velocity) == 0


def test_action_expert_ignores_masked_llm_tokens(sizes):
    torch.manual_seed(0)
    expert = ActionExpert(**sizes.action_expert, num_layers=2, llm_kv_dim=8)
    for param in expert.parameters():
        nn.init.normal_(param, std=0.1)
    noisy_actions, timesteps, context = _expert_inputs(sizes, expert)
    changed = [(key.clone(), value.clone()) for key, value in context]
    for key, value in changed:
        key[1, 3:], value[1, 3:] = 100.0, 100.0
    encoder_attention_mask = torch.tensor([[True] * 5, [True, True, True, False, False]])

    with torch.no_grad():
        velocity = expert(noisy_actions, timesteps, context, encoder_attention_mask)
        changed_velocity = expert(noisy_actions, timesteps, changed, encoder_attention_mask)
        unmasked = torch.ones_like(encoder_attention_mask)
        unmasked_changed = expert(noisy_actions, timesteps, changed, unmasked)

    assert torch.count_nonzero(velocity) > 0
    torch.testing.assert_close(changed_velocity, velocity)
    assert not torch.allclose(unmasked_changed, expert(noisy_actions, timesteps, context, unmasked))


def test_sample_actions_keeps_padded_dims_at_zero(sizes, policy, batch):
    noise = _noise(sizes)

    actions = policy.model.eval().sample_actions(batch, noise=noise)

    assert actions.shape == (2, sizes.chunk_size, sizes.max_action_dim)
    assert torch.count_nonzero(actions[..., sizes.action_dim :]) == 0
    assert torch.count_nonzero(actions[..., : sizes.action_dim]) > 0
    assert torch.equal(policy.model.sample_actions(batch, noise=noise), actions)


def test_flow_loss_averages_the_squared_error_over_valid_dims(sizes, make_policy, monkeypatch):
    model = make_policy(action_mode="continuous").model
    batch = make_batch(sizes, model.vlm.config.image_patch_id, action_span=False)
    batch["action_dim_is_pad"][1, 3:] = True  # row 1 has 3 action dims
    batch[ACTION][..., sizes.action_dim :] = 5.0  # padded dims never count
    flow = _flow_inputs(sizes)
    # With a zero prediction the loss is the mean squared target (actions - noise) over the valid dims.
    monkeypatch.setattr(
        model.action_expert.final_layer,
        "forward",
        lambda hidden, conditioning: torch.zeros(*hidden.shape[:2], sizes.max_action_dim),
    )

    per_example = model(batch, reduction="none", **flow)["flow_loss"]
    mean = model(batch, **flow)["flow_loss"]

    valid = ~batch["action_dim_is_pad"]
    expected = torch.stack(
        [
            ((batch[ACTION][row][:, valid[row]] - flow["noise"][row][..., valid[row]]) ** 2).mean()
            for row in range(2)
        ]
    )
    torch.testing.assert_close(per_example, expected)
    torch.testing.assert_close(mean, expected.mean())


def test_discrete_loss_equals_cross_entropy_for_equal_token_counts(policy):
    hidden = torch.randn(2, 6, 16)
    labels = torch.tensor([[-100, 5, 6, -100, -100, -100], [-100, -100, -100, 7, 8, -100]])

    ce_loss, _ = policy.model._cross_entropy_loss(hidden, labels, "mean")

    logits = policy.model.vlm.lm_head(hidden).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    expected = F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(), ignore_index=-100)
    torch.testing.assert_close(ce_loss, expected)


def test_discrete_loss_weights_examples_by_inverse_root_token_count(policy):
    hidden = torch.randn(3, 5, 16)
    labels = torch.tensor(
        [[-100, -100, 1, -100, -100], [-100, 2, 3, 4, -100], [-100, -100, -100, -100, -100]]
    )

    ce_loss, z_loss = policy.model._cross_entropy_loss(hidden, labels, "mean")
    ce_none, z_none = policy.model._cross_entropy_loss(hidden, labels, "none")

    logits = policy.model.vlm.lm_head(hidden).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    valid = shift_labels != -100
    log_z = logits.logsumexp(-1)
    token_ce = log_z - logits.gather(-1, shift_labels.clamp_min(0)[..., None]).squeeze(-1)
    counts = valid.sum(1, keepdim=True).float()
    weights = torch.where(valid, 2.0 / counts.clamp_min(1).sqrt(), 0.0)
    torch.testing.assert_close(ce_loss, (token_ce * weights).sum() / weights.sum())
    torch.testing.assert_close(z_loss, 1e-4 * (log_z.pow(2) * weights).sum() / weights.sum())
    # Per-example losses average to the batch loss; an example without action tokens contributes zero.
    assert ce_none.shape == z_none.shape == (3,)
    assert ce_none[2] == 0 and z_none[2] == 0
    torch.testing.assert_close(ce_none.mean(), ce_loss)
    torch.testing.assert_close(z_none.mean(), z_loss)
    with pytest.raises(ValueError, match="no action tokens"):
        policy.model._cross_entropy_loss(hidden, torch.full_like(labels, -100), "mean")


@pytest.mark.parametrize(
    ("action_mode", "inference_action_mode", "loss_names"),
    [
        ("continuous", "continuous", {"flow_loss"}),
        ("discrete", "discrete", {"ce_loss", "z_loss"}),
        ("both", "continuous", {"flow_loss", "ce_loss", "z_loss"}),
    ],
)
def test_policy_loss_is_the_sum_of_the_action_mode_losses(
    sizes, make_policy, action_mode, inference_action_mode, loss_names
):
    # In eval mode the residual dropout is off, so that every call computes the same losses.
    policy = make_policy(action_mode=action_mode, inference_action_mode=inference_action_mode).eval()
    batch = make_batch(sizes, policy.model.vlm.config.image_patch_id)
    flow = _flow_inputs(sizes)

    losses = policy.model(batch, **flow)
    loss, log = policy.forward(batch, **flow)
    per_example, log_none = policy.forward(batch, reduction="none", **flow)

    assert losses.keys() == loss_names and log.keys() == loss_names | {"loss"}
    torch.testing.assert_close(loss, sum(losses.values()))
    assert per_example.shape == (2,)
    torch.testing.assert_close(per_example.mean(), loss)
    assert log_none["loss"] == pytest.approx(log["loss"])
    with pytest.raises(ValueError, match="reduction"):
        policy.forward(batch, reduction="sum")


def test_gradient_checkpointing_gives_the_same_losses_and_gradients(sizes, make_policy, batch):
    flow = _flow_inputs(sizes)
    results = []
    for gradient_checkpointing in (False, True):
        policy = make_policy(gradient_checkpointing=gradient_checkpointing).train()
        torch.manual_seed(1)  # the same residual dropout masks
        loss, log = policy.forward(batch, **flow)
        loss.backward()
        grads = {name: param.grad for name, param in policy.named_parameters() if param.grad is not None}
        results.append((log, grads))

    assert (
        policy.model.gradient_checkpointing_enabled
        and policy.model.vlm.model.vision_tower.is_gradient_checkpointing
    )
    (log, grads), (checkpointed_log, checkpointed_grads) = results
    assert log == checkpointed_log
    assert grads.keys() == checkpointed_grads.keys()
    assert any("action_expert" in name for name in grads) and any("vision_tower" in name for name in grads)
    for name, grad in grads.items():
        torch.testing.assert_close(checkpointed_grads[name], grad, msg=name)


def test_train_expert_only_freezes_the_vlm(sizes, make_policy, batch):
    policy = make_policy(action_mode="continuous", train_expert_only=True).train()
    vlm, expert = policy.model.vlm, policy.model.action_expert

    # Upstream only sets requires_grad: the frozen VLM stays in train mode and keeps its residual dropout.
    assert all(module.training for module in policy.modules())
    assert not any(param.requires_grad for param in vlm.parameters())
    assert [group["lr"] for group in policy.get_optim_params()] == [policy.config.optimizer_action_expert_lr]
    loss, _ = policy.forward(batch, **_flow_inputs(sizes))
    loss.backward()
    assert all(param.grad is None for param in vlm.parameters())
    assert all(param.grad is not None for param in expert.parameters())


def test_freeze_vision_encoder_freezes_only_the_vision_tower(make_policy):
    policy = make_policy(freeze_vision_encoder=True)
    config = policy.config

    for name, param in policy.named_parameters():
        assert param.requires_grad == ("vision_tower" not in name), name
    lrs = [group["lr"] for group in policy.get_optim_params()]
    assert lrs == [config.optimizer_lr, config.optimizer_connector_lr, config.optimizer_action_expert_lr]


@pytest.mark.parametrize("freeze_embedding", [True, False])
def test_freeze_embedding_trains_only_the_extra_image_token_rows(sizes, make_policy, batch, freeze_embedding):
    policy = make_policy(freeze_embedding=freeze_embedding)

    loss, _ = policy.forward(batch, **_flow_inputs(sizes))
    loss.backward()

    grad = policy.model.vlm.model.language_model.embed_tokens.weight.grad
    # The batch uses the base rows (text, BOS, actions) and the first two extra rows (image start and end).
    assert torch.count_nonzero(grad[sizes.base_vocab_size : sizes.base_vocab_size + 2].abs().sum(-1)) == 2
    assert (torch.count_nonzero(grad[: sizes.base_vocab_size]) == 0) == freeze_embedding


def test_optimizer_groups_use_upstreams_lora_learning_rate_with_peft(make_policy):
    policy = make_policy(optimizer_lr=1.0, optimizer_vit_lr=2.0, optimizer_connector_lr=3.0)
    policy.config.use_peft = True

    lrs = [group["lr"] for group in policy.get_optim_params()]

    # Upstream's LoRA recipe trains the LLM, ViT and connector groups at 5e-5; the action expert keeps its own.
    assert lrs == [5e-5, 5e-5, 5e-5, policy.config.optimizer_action_expert_lr]


def test_optimizer_groups_split_the_parameters_by_component(make_policy):
    policy = make_policy(
        optimizer_lr=1.0, optimizer_vit_lr=2.0, optimizer_connector_lr=3.0, optimizer_action_expert_lr=4.0
    )
    names = {param: name for name, param in policy.named_parameters()}

    groups = policy.get_optim_params()

    assert [group["lr"] for group in groups] == [1.0, 2.0, 3.0, 4.0]
    llm, vit, connector, action_expert = ([names[param] for param in group["params"]] for group in groups)
    assert sorted(llm + vit + connector + action_expert) == sorted(names.values())
    assert {"model.vlm.lm_head.weight", "model.vlm.model.language_model.norm.weight"} <= set(llm)
    assert all("vision_tower" in name for name in vit)
    assert "model.vlm.model.language_model.embed_tokens.weight" in connector
    assert all("multi_modal_projector" in name or "embed_tokens" in name for name in connector)
    assert all(name.startswith("model.action_expert.") for name in action_expert)


def test_bfloat16_policy_stores_only_the_llm_in_bfloat16(sizes, make_policy, batch):
    policy = make_policy(dtype=torch.bfloat16)
    vlm = policy.model.vlm
    float32_modules = [policy.model.action_expert, vlm.model.vision_tower, vlm.model.multi_modal_projector]
    float32_modules += [
        module for module in vlm.modules() if isinstance(module, (nn.LayerNorm, Molmo2RMSNorm))
    ]
    float32_params = {param for module in float32_modules for param in module.parameters()}

    for name, param in policy.named_parameters():
        assert param.dtype == (torch.float32 if param in float32_params else torch.bfloat16), name
    loss, _ = policy.forward(batch, **_flow_inputs(sizes))
    loss.backward()
    actions = policy.predict_action_chunk(batch)
    assert torch.isfinite(loss) and actions.dtype == torch.float32


def test_generate_tokens_with_the_cache_matches_a_full_forward(sizes, policy):
    model = policy.model.eval()
    batch = make_batch(sizes, model.vlm.config.image_patch_id, action_span=False)
    step_logits = []

    def select_token(logits, generated):
        step_logits.append(logits)
        step = len(step_logits)
        # Greedy over the first ids; row 0 stops after 2 tokens and row 1 after 4.
        return torch.where(torch.tensor([step >= 2, step >= 4]), EOS_TOKEN_ID, logits[:, :1000].argmax(-1))

    generated = model.generate_tokens(batch, 10, select_token, EOS_TOKEN_ID)

    assert len(step_logits) == 4  # decoding stops once every row emitted EOS
    assert [len(tokens) for tokens in generated] == [2, 4]
    assert all(tokens[-1] == EOS_TOKEN_ID for tokens in generated)
    with torch.no_grad():
        for row, tokens in enumerate(generated):
            for step in range(len(tokens)):
                hidden, _ = model.prefill(row_batch(sizes, batch, row, tokens[:step]))
                expected = model.vlm.lm_head(hidden[0, -1])
                torch.testing.assert_close(step_logits[step][row], expected, rtol=1e-4, atol=1e-4)


def test_discrete_inference_decodes_the_constrained_fast_span(sizes, make_policy, fast_dir, monkeypatch):
    policy = make_policy(inference_action_mode="discrete", action_tokenizer_name=str(fast_dir))
    batch = make_batch(sizes, policy.model.vlm.config.image_patch_id, action_span=False)
    generate_tokens = policy.model.generate_tokens
    outputs = []

    def record_generate_tokens(*args, **kwargs):
        outputs.append(generate_tokens(*args, **kwargs))
        return outputs[-1]

    monkeypatch.setattr(policy.model, "generate_tokens", record_generate_tokens)

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (2, sizes.chunk_size, sizes.action_dim) and actions.dtype == torch.float32
    fast = UniversalActionProcessor.from_pretrained(str(fast_dir))
    for row, tokens in enumerate(outputs[0]):
        assert tokens[0] == _ACTION_START_TOKEN_ID and tokens[-2:] == [_ACTION_END_TOKEN_ID, EOS_TOKEN_ID]
        bins = [token - _ACTION_TOKEN_START_ID for token in tokens[1:-2]]
        assert len(fast.bpe_tokenizer.decode(bins)) == sizes.chunk_size * sizes.action_dim
        expected = fast.decode([bins], time_horizon=sizes.chunk_size, action_dim=sizes.action_dim)[0]
        torch.testing.assert_close(actions[row], torch.as_tensor(expected, dtype=torch.float32))


def test_select_action_pops_actions_from_one_predicted_chunk(sizes, make_policy, batch, monkeypatch):
    policy = make_policy(n_action_steps=2)
    sample_actions = policy.model.sample_actions
    calls = []

    def count_sample_actions(*args, **kwargs):
        calls.append(1)
        return sample_actions(*args, **kwargs)

    monkeypatch.setattr(policy.model, "sample_actions", count_sample_actions)
    noise = _noise(sizes)

    chunk = policy.predict_action_chunk(batch, noise=noise)
    actions = [policy.select_action(batch, noise=noise) for _ in range(3)]

    assert chunk.shape == (
        2,
        sizes.chunk_size,
        sizes.action_dim,
    )  # the whole chunk; select_action plays the first 2
    assert len(calls) == 3  # the chunk above, then one per 2 selected actions
    assert torch.equal(actions[0], chunk[:, 0]) and torch.equal(actions[1], chunk[:, 1])
    assert torch.equal(actions[2], chunk[:, 0])
    policy.reset()
    assert len(policy._action_queue) == 0


def test_rtc_guides_sampling_toward_the_previous_chunk(sizes, make_policy, batch):
    policy = make_policy(rtc_config=RTCConfig(execution_horizon=sizes.chunk_size, max_guidance_weight=1.0))
    noise = _noise(sizes)
    prefix = torch.full((2, sizes.chunk_size, sizes.action_dim), 0.5)

    free = policy.predict_action_chunk(batch, noise=noise)
    guided = policy.predict_action_chunk(batch, noise=noise, prev_chunk_left_over=prefix, inference_delay=2)

    # Without a previous chunk, RTC is plain Euler sampling.
    torch.testing.assert_close(free, make_policy().predict_action_chunk(batch, noise=noise))
    assert (guided[:, :2] - 0.5).abs().mean() < (free[:, :2] - 0.5).abs().mean()
    with pytest.raises(ValueError, match="RTC is not supported for select_action"):
        policy.select_action(batch)


def test_policy_is_randomly_initialized_without_a_checkpoint(make_policy, caplog):
    with caplog.at_level(logging.WARNING):
        make_policy()

    assert "MolmoAct2 is randomly initialized" in caplog.text


def test_policy_round_trips_through_save_pretrained(sizes, make_policy, batch, tmp_path, caplog):
    policy = make_policy(dtype=torch.bfloat16)
    noise = _noise(sizes)
    actions = policy.predict_action_chunk(batch, noise=noise)

    policy.save_pretrained(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        loaded = MolmoAct2Policy.from_pretrained(tmp_path)

    # Built for the checkpoint: the weights are not initialized before it overwrites them.
    assert loaded.config.pretrained_path == tmp_path
    assert "randomly initialized" not in caplog.text
    assert not loaded.training
    assert {name: param.dtype for name, param in loaded.named_parameters()} == {
        name: param.dtype for name, param in policy.named_parameters()
    }
    assert torch.equal(loaded.predict_action_chunk(batch, noise=noise), actions)


def test_default_peft_targets_cover_every_vlm_linear_layer(policy):
    targets = policy._get_default_peft_targets()
    linear_layers = [name for name, module in policy.named_modules() if isinstance(module, nn.Linear)]

    matched = {name for name in linear_layers if re.fullmatch(targets["target_modules"], name)}

    assert matched == {name for name in linear_layers if name.startswith("model.vlm.")}
    assert targets["modules_to_save"] == ["model.action_expert"]
    assert targets["init_lora_weights"] == "gaussian"  # upstream: A ~ N(0, 1/r), B = 0


def test_lora_trains_the_adapters_and_the_whole_action_expert(sizes, policy, batch, tmp_path):
    pytest.importorskip("peft")
    num_expert_params = len(list(policy.model.action_expert.parameters()))

    # LeRobot only wraps policies loaded from a LeRobot checkpoint.
    policy.config.pretrained_path = tmp_path
    peft_policy = policy.wrap_with_peft()

    trainable = {name for name, param in peft_policy.named_parameters() if param.requires_grad}
    expert = {name for name in trainable if ".action_expert." in name}
    assert expert and all("modules_to_save" in name for name in expert)
    assert len(expert) == num_expert_params
    assert trainable - expert and all("lora_" in name for name in trainable - expert)
    loss, _ = peft_policy.forward(batch, **_flow_inputs(sizes))
    loss.backward()
    without_grad = [
        name for name, param in peft_policy.named_parameters() if param.requires_grad and param.grad is None
    ]
    assert not without_grad, f"Trainable parameters that the loss does not reach: {without_grad}"
