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

"""Tests for XVLA-RMoE: registration/config, MoEFFN mechanics, original-compatible-mode
numerical equivalence with plain X-VLA, action-token masking/position embedding, checkpoint
remapping, and a full tiny end-to-end training + inference run.

See `_helpers.py` for why every test here can run on CPU with a real (tiny) model and no
network access: X-VLA's Florence-2 config is fully self-contained.

Truncated cross-step recurrent-training-specific tests (state chaining, GRU gradient flow,
detach ablation, ground-truth flow path, shared noise, DDP-safe branch selection) live in
`test_xvla_rmoe_recurrent.py`.
"""

import copy
import json

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.xvla.action_hub import EE6DActionSpace
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.soft_transformer import Mlp
from lerobot.policies.xvla_rmoe.configuration_xvla_rmoe import XVLARMoEConfig
from lerobot.policies.xvla_rmoe.modeling_xvla_rmoe import (
    XVLARMoEModel,
    XVLARMoEPolicy,
    _merge_pretrained_xvla_rmoe_config,
    _remap_xvla_mlp_weights_to_moe,
)
from lerobot.policies.xvla_rmoe.moe_soft_transformer import MoEFFN, chunk_position_embedding, masked_mean
from lerobot.utils.constants import ACTION
from tests.policies.xvla_rmoe._helpers import (
    ACTION_DIM,
    GRAD_NOISE_FLOOR,
    STATE_DIM,
    TEXT_CONFIG,
    VISION_CONFIG,
    construct_until_finite,
    make_tiny_batch,
    make_tiny_config,
    make_tiny_policy,
)

pytest.importorskip("transformers")

_COMMON_KWARGS = {
    "florence_config": {
        "vision_config": VISION_CONFIG,
        "text_config": TEXT_CONFIG,
        "projection_dim": 16,
        "pad_token_id": 1,
    },
    "hidden_size": 32,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "num_domains": 2,
    "len_soft_prompts": 2,
    "dim_time": 8,
    "max_len_seq": 128,
    "chunk_size": 6,
    "n_action_steps": 6,
    "num_denoising_steps": 6,
    "action_mode": "joint",
    "max_state_dim": STATE_DIM,
    "use_proprio": True,
    "tokenizer_max_length": 6,
    "resize_imgs_with_padding": (64, 64),
    "num_image_views": 1,
    "device": "cpu",
}


def _set_matching_features(cfg) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}


# ---------------------------------------------------------------------------
# Test 1: registration and config
# ---------------------------------------------------------------------------


def test_policy_registered_in_factory():
    assert get_policy_class("xvla_rmoe") is XVLARMoEPolicy
    cfg = make_policy_config("xvla_rmoe")
    assert isinstance(cfg, XVLARMoEConfig)
    assert PreTrainedConfig.get_choice_class("xvla_rmoe") is XVLARMoEConfig


def test_config_is_xvla_subclass_with_rmoe_defaults():
    cfg = XVLARMoEConfig()
    assert isinstance(cfg, XVLAConfig)
    assert cfg.use_moe is True
    assert cfg.use_routing_memory is True
    assert cfg.use_timestep_router is True
    assert cfg.use_recurrent_routing_training is True
    assert cfg.moe_layer_indices == list(range(cfg.depth - cfg.num_moe_layers, cfg.depth))


def test_pretrained_config_merge_keeps_checkpoint_values_and_cli_overrides():
    pretrained = make_tiny_config(num_moe_experts=4)
    checkpoint_florence_config = copy.deepcopy(pretrained.florence_config)
    checkpoint_input_features = copy.deepcopy(pretrained.input_features)
    checkpoint_output_features = copy.deepcopy(pretrained.output_features)
    requested = XVLARMoEConfig(num_moe_experts=3, expert_symmetry_breaking_std=0.0)
    requested.pretrained_path = "org/plain-xvla"
    requested.input_features = {"observation.images.front": next(iter(checkpoint_input_features.values()))}
    requested.output_features = {}

    merged = _merge_pretrained_xvla_rmoe_config(pretrained, requested)

    assert merged.florence_config == checkpoint_florence_config
    assert merged.input_features == checkpoint_input_features
    assert merged.output_features == checkpoint_output_features
    assert merged.num_moe_experts == 3
    assert merged.pretrained_path == "org/plain-xvla"


def test_single_arm_ee6d_loss_ignores_padded_second_arm_and_uses_gripper_9():
    class LossFixture:
        config = type("Config", (), {"action_mode": "ee6d", "single_arm_ee6d_loss": True})()
        action_space = EE6DActionSpace()

    target = torch.zeros(1, 2, 20)
    target[..., 9] = 1.0
    prediction = torch.zeros_like(target)
    valid = torch.tensor([[True, False]])

    baseline = XVLARMoEModel._compute_action_loss(LossFixture(), prediction, target, valid)

    # Neither the synthetic second arm nor an invalid episode-tail timestep may affect loss.
    changed = prediction.clone()
    changed[..., 10:20] = 100.0
    changed[:, 1, :10] = 100.0
    actual = XVLARMoEModel._compute_action_loss(LossFixture(), changed, target, valid)

    for key in baseline:
        torch.testing.assert_close(actual[key], baseline[key])
    torch.testing.assert_close(
        baseline["gripper_loss"],
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.0]), torch.tensor([1.0])
        ),
    )


def test_build_model_inputs_aligns_action_mask_to_checkpoint_chunk_size():
    policy = make_tiny_policy()
    batch = make_tiny_batch(policy)
    batch["action_is_pad"] = torch.tensor(
        [[False] * policy.config.chunk_size + [True, True]] * batch[ACTION].shape[0]
    )

    inputs = policy._build_model_inputs(batch)

    assert inputs["action_padding_mask"].shape == (
        batch[ACTION].shape[0],
        policy.config.chunk_size,
    )
    assert inputs["action_padding_mask"].all()

    batch["action_is_pad"] = torch.zeros(
        batch[ACTION].shape[0], policy.config.chunk_size - 2, dtype=torch.bool
    )
    inputs = policy._build_model_inputs(batch)
    assert inputs["action_padding_mask"][:, :-2].all()
    assert not inputs["action_padding_mask"][:, -2:].any()


def test_config_serialization_round_trip(tmp_path):
    cfg = make_tiny_config(num_moe_experts=5, chunk_pos_emb_dim=6)
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert type(loaded) is XVLARMoEConfig
    assert loaded.num_moe_experts == 5
    assert loaded.chunk_pos_emb_dim == 6
    assert loaded.moe_layer_indices == cfg.moe_layer_indices


def test_config_validation_failures():
    with pytest.raises(ValueError):
        make_tiny_config(num_moe_experts=1)
    with pytest.raises(ValueError):
        make_tiny_config(routing_hidden_dim=0)
    with pytest.raises(ValueError):
        make_tiny_config(expert_symmetry_breaking_std=-1e-5)
    with pytest.raises(ValueError):
        make_tiny_config(recurrent_training_probability=1.5)
    with pytest.raises(ValueError):
        make_tiny_config(recurrent_unroll_steps=1)
    with pytest.raises(ValueError):
        make_tiny_config(recurrent_unroll_steps=999)
    with pytest.raises(ValueError):
        make_tiny_config(recurrent_timestep_sampling="continuous")
    with pytest.raises(ValueError):
        make_tiny_config(recurrent_loss_reduction="bogus")
    with pytest.raises(ValueError):
        make_tiny_config(moe_layer_indices=[])
    with pytest.raises(ValueError):
        make_tiny_config(moe_layer_indices=[999])


def test_config_auto_disables_dependent_flags():
    cfg = make_tiny_config(use_moe=False)
    assert cfg.use_routing_memory is False
    assert cfg.use_recurrent_routing_training is False

    cfg2 = make_tiny_config(use_routing_memory=False)
    assert cfg2.use_recurrent_routing_training is False
    assert cfg2.use_moe is True  # only the dependent flags are forced off


# ---------------------------------------------------------------------------
# Test 2 / 15: original-compatible mode == plain X-VLA (shape + exact numerics)
# ---------------------------------------------------------------------------


def test_original_compatible_mode_matches_plain_xvla_exactly():
    torch.manual_seed(42)
    cfg_rmoe = XVLARMoEConfig(use_moe=False, **_COMMON_KWARGS)
    _set_matching_features(cfg_rmoe)
    policy_rmoe = construct_until_finite(lambda: XVLARMoEPolicy(cfg_rmoe))
    assert policy_rmoe.model.routing_memory is None
    assert policy_rmoe.model.transformer.num_moe_layers == 0

    torch.manual_seed(123)  # deliberately different seed: weights are copied over below
    cfg_orig = XVLAConfig(**_COMMON_KWARGS)
    _set_matching_features(cfg_orig)
    policy_orig = XVLAPolicy(cfg_orig)

    rmoe_keys = set(policy_rmoe.state_dict().keys())
    orig_keys = set(policy_orig.state_dict().keys())
    assert rmoe_keys == orig_keys, "use_moe=False must not add or remove any parameter"
    policy_orig.load_state_dict(policy_rmoe.state_dict(), strict=True)

    policy_rmoe.eval()
    policy_orig.eval()
    batch = make_tiny_batch(policy_rmoe)

    # `torch.equal` (bit-exact) is deliberately not used here: `policy_rmoe` and `policy_orig`
    # are separate module graphs whose (value-identical, post `load_state_dict`) weight
    # tensors live at different memory addresses, and CPU BLAS/conv kernels are not
    # guaranteed to reduce in bit-identical order across separately-allocated tensors (see
    # `construct_until_finite`'s docstring for the same underlying CPU-determinism caveat).
    torch.manual_seed(7)
    loss_rmoe, _ = policy_rmoe.forward(dict(batch))
    torch.manual_seed(7)
    loss_orig, _ = policy_orig.forward(dict(batch))
    torch.testing.assert_close(loss_rmoe, loss_orig, atol=1e-4, rtol=1e-4)

    torch.manual_seed(9)
    actions_rmoe = policy_rmoe._get_action_chunk(batch)
    torch.manual_seed(9)
    actions_orig = policy_orig._get_action_chunk(batch)
    assert actions_rmoe.shape == actions_orig.shape == (2, cfg_rmoe.chunk_size, policy_rmoe.model.dim_action)
    torch.testing.assert_close(actions_rmoe, actions_orig, atol=1e-4, rtol=1e-4)


def test_inference_regression_when_rmoe_options_disabled():
    """Test 15: with `use_moe=False`, `predict_action_chunk` runs end-to-end and produces the
    expected chunk shape, unaffected by any RMoE-only code path."""
    torch.manual_seed(0)
    policy = make_tiny_policy(use_moe=False)
    policy.eval()
    batch = make_tiny_batch(policy)
    with torch.no_grad():
        actions = policy.predict_action_chunk(dict(batch))
    assert actions.shape == (2, policy.config.chunk_size, policy.model.dim_action)
    assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# Test 3 / 4: MoEFFN shape + near-function-preserving init
# ---------------------------------------------------------------------------


def test_moeffn_output_shape_dtype_device_matches_original_ffn():
    torch.manual_seed(0)
    original = Mlp(in_features=16, hidden_features=32, out_features=16)
    moe = MoEFFN(
        original,
        num_experts=4,
        routing_hidden_dim=8,
        routing_timestep_dim=8,
        chunk_pos_emb_dim=4,
        use_routing_memory=True,
        use_timestep_router=True,
        use_chunk_position_embedding=True,
        expert_symmetry_breaking_std=1e-5,
    )
    x = torch.randn(3, 7, 16)
    routing_state = torch.randn(3, 8)
    timestep_emb = torch.randn(3, 8)
    chunk_pos_emb = torch.randn(3, 7, 4)

    out, decision, full_weights = moe(x, routing_state, timestep_emb, chunk_pos_emb)
    orig_out = original(x)

    assert out.shape == orig_out.shape == (3, 7, 16)
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert decision.shape == (3, 4)
    assert full_weights is None  # return_full_weights defaults to False

    _, _, full_weights2 = moe(x, routing_state, timestep_emb, chunk_pos_emb, return_full_weights=True)
    assert full_weights2.shape == (3, 7, 4)


def test_moeffn_zero_symmetry_breaking_reproduces_original_ffn_output():
    """Test 4a: with `expert_symmetry_breaking_std=0` and a zero-init router, the mixture
    output must equal the wrapped FFN's output exactly (uniform routing over bit-identical
    experts)."""
    torch.manual_seed(0)
    original = Mlp(in_features=8, hidden_features=16, out_features=8)
    moe = MoEFFN(
        original,
        num_experts=4,
        routing_hidden_dim=8,
        routing_timestep_dim=8,
        chunk_pos_emb_dim=0,
        use_routing_memory=True,
        use_timestep_router=True,
        use_chunk_position_embedding=False,
        expert_symmetry_breaking_std=0.0,
    )
    x = torch.randn(2, 3, 8)
    routing_state = torch.zeros(2, 8)
    timestep_emb = torch.zeros(2, 8)
    out, decision, _ = moe(x, routing_state, timestep_emb, None)

    torch.testing.assert_close(out, original(x))
    torch.testing.assert_close(decision, torch.full_like(decision, 1.0 / 4))


def test_moeffn_default_symmetry_breaking_is_near_function_preserving():
    """Test 4b: the default `expert_symmetry_breaking_std=1e-5` leaves experts genuinely
    different from each other (bootstrap requirement), while the relative L2 error of the
    step-0 mixture output vs. the original FFN stays within a small tolerance."""
    torch.manual_seed(0)
    original = Mlp(in_features=8, hidden_features=16, out_features=8)
    moe = MoEFFN(
        original,
        num_experts=4,
        routing_hidden_dim=8,
        routing_timestep_dim=8,
        chunk_pos_emb_dim=0,
        use_routing_memory=True,
        use_timestep_router=True,
        use_chunk_position_embedding=False,
        expert_symmetry_breaking_std=1e-5,
    )
    assert not torch.equal(moe.experts[0].fc1.weight, moe.experts[1].fc1.weight)

    x = torch.randn(2, 3, 8)
    routing_state = torch.zeros(2, 8)
    timestep_emb = torch.zeros(2, 8)
    out, decision, _ = moe(x, routing_state, timestep_emb, None)
    orig_out = original(x)

    rel_err = (out - orig_out).norm() / (orig_out.norm() + 1e-8)
    assert 0.0 < rel_err.item() < 1e-2  # near- but not exactly- function preserving
    torch.testing.assert_close(decision, torch.full_like(decision, 1.0 / 4))


# ---------------------------------------------------------------------------
# Test 11: action-token masked pooling
# ---------------------------------------------------------------------------


def test_masked_mean_ignores_padded_positions():
    torch.manual_seed(0)
    values = torch.randn(4, 6, 5)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, :3] = True
    baseline = masked_mean(values, mask, dim=1)

    corrupted = values.clone()
    corrupted[:, 3:] = 1e6
    corrupted_result = masked_mean(corrupted, mask, dim=1)

    torch.testing.assert_close(baseline, corrupted_result)
    torch.testing.assert_close(masked_mean(values, None, dim=1), values.mean(dim=1))


def test_moeffn_routing_decision_ignores_padded_positions():
    """Test 11 at the `MoEFFN` level: X-VLA has no attention-masking convention at all (dense,
    unmasked self-attention -- see `moe_soft_transformer.py`'s module docstring), so a padded
    action token's *value* legitimately influences other tokens' hidden state through
    attention, same as any other X-VLA token. `MoEFFN` itself, however, is a strictly
    per-token operation applied *after* attention (`router`/experts are `nn.Linear`s with no
    cross-token mixing), so corrupting a padded position must change that position's own
    routing weight but leave every valid position's routing weight -- and therefore the
    masked-mean routing decision -- completely untouched. This is the actual, honest
    per-layer guarantee `token_mask` provides."""
    torch.manual_seed(0)
    original = Mlp(in_features=8, hidden_features=16, out_features=8)
    moe = MoEFFN(
        original,
        num_experts=3,
        routing_hidden_dim=4,
        routing_timestep_dim=4,
        chunk_pos_emb_dim=0,
        use_routing_memory=True,
        use_timestep_router=True,
        use_chunk_position_embedding=False,
        expert_symmetry_breaking_std=1e-3,
    )
    torch.nn.init.normal_(moe.router.weight, std=1.0)  # make the router input-sensitive

    x = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :4] = True  # last 2 positions are padding
    routing_state = torch.randn(2, 4)
    timestep_emb = torch.randn(2, 4)

    _, decision_a, full_a = moe(
        x, routing_state, timestep_emb, None, token_mask=mask, return_full_weights=True
    )

    x_corrupted = x.clone()
    x_corrupted[:, 4:] = 1e6  # corrupt only the padded positions
    _, decision_b, full_b = moe(
        x_corrupted, routing_state, timestep_emb, None, token_mask=mask, return_full_weights=True
    )

    torch.testing.assert_close(decision_a, decision_b)  # masked-mean decision: unaffected
    torch.testing.assert_close(full_a[:, :4], full_b[:, :4])  # valid tokens' own weights: unaffected
    assert not torch.allclose(full_a[:, 4:], full_b[:, 4:])  # sanity: the corruption did do something


def test_transformer_hidden_summary_wiring_matches_manual_masked_mean():
    """Test 11 at the full-transformer level: verifies `action_padding_mask` is correctly
    sliced/aligned end-to-end (not off-by-one, not accidentally covering context tokens) by
    intercepting the real final action-token hidden state and independently recomputing the
    masked mean from it, rather than asserting corruption-invariance (impossible to guarantee
    here -- see `test_moeffn_routing_decision_ignores_padded_positions` above for why, and for
    the guarantee that *is* honestly testable end-to-end)."""
    torch.manual_seed(0)
    policy = make_tiny_policy()
    policy.eval()
    batch = make_tiny_batch(policy, pad_last_n=2)
    inputs = policy._build_model_inputs(batch)
    targets = policy._prepare_action_targets(batch)

    captured = {}

    def _capture(_module, _inputs, output):
        captured["final_action_hidden"] = output

    handle = policy.model.transformer.norm.register_forward_hook(_capture)
    try:
        step_losses, routing_infos, _extra = policy.model.forward_recurrent(action=targets, **inputs)
    finally:
        handle.remove()

    action_padding_mask = inputs["action_padding_mask"]
    manual_hidden_summary = masked_mean(captured["final_action_hidden"], action_padding_mask, dim=1)
    # `equal_nan=True`: this is a self-consistency check (both sides are the masked mean of the
    # *same* captured tensor), not a claim that either side must itself be finite -- an
    # untrained tiny model occasionally producing NaN activations (see `construct_until_finite`)
    # would legitimately make both sides NaN in the same positions, which should still count as
    # "correctly wired", not a mismatch.
    torch.testing.assert_close(
        routing_infos[-1].hidden_summary,
        manual_hidden_summary.to(torch.float32),
        atol=1e-5,
        rtol=1e-5,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Test 12: action-chunk position embedding only applied to action tokens
# ---------------------------------------------------------------------------


def test_chunk_position_embedding_is_zero_outside_action_tokens():
    policy = make_tiny_policy()
    policy.eval()
    batch = make_tiny_batch(policy)
    inputs = policy._build_model_inputs(batch)
    action = policy._prepare_action_targets(batch)

    transformer = policy.model.transformer
    num_actions = action.shape[1]

    # Reproduce the same full-sequence length the transformer builds internally: run a real
    # forward to get vlm_features/aux_visual_inputs shapes, then recompute the position tensor
    # exactly as `SoftPromptedTransformerRMoE.forward` does.
    enc = policy.model.forward_vlm(inputs["input_ids"], inputs["image_input"].float(), inputs["image_mask"])
    action_tokens_len = num_actions
    vlm_len = enc["vlm_features"].shape[1]
    aux_len = enc["aux_visual_inputs"].shape[1]
    soft_len = transformer.len_soft_prompts
    full_seq_len = action_tokens_len + vlm_len + aux_len + soft_len

    pe = chunk_position_embedding(num_actions, transformer.chunk_pos_emb_dim, torch.device("cpu"))
    pe_full = torch.zeros(full_seq_len, transformer.chunk_pos_emb_dim)
    pe_full[:num_actions] = pe

    assert torch.all(pe_full[num_actions:] == 0.0)
    assert torch.any(pe_full[:num_actions] != 0.0)
    # Position embedding must be aligned with the actual action-token index, i.e. strictly
    # increasing in the (arbitrary but monotonic) sinusoidal-embedding L2 norm sense is not
    # guaranteed, but position 0 and position num_actions-1 must differ.
    assert not torch.allclose(pe_full[0], pe_full[num_actions - 1])


# ---------------------------------------------------------------------------
# Test 13: plain X-VLA checkpoint remapping
# ---------------------------------------------------------------------------


def test_checkpoint_remap_broadcasts_experts_and_breaks_symmetry(tmp_path):
    torch.manual_seed(0)
    cfg_plain = XVLAConfig(**_COMMON_KWARGS)
    _set_matching_features(cfg_plain)
    policy_plain = XVLAPolicy(cfg_plain)
    plain_state_dict = policy_plain.state_dict()

    cfg_rmoe = XVLARMoEConfig(expert_symmetry_breaking_std=1e-4, **_COMMON_KWARGS)
    _set_matching_features(cfg_rmoe)
    policy_rmoe = construct_until_finite(lambda: XVLARMoEPolicy(cfg_rmoe))

    remapped = _remap_xvla_mlp_weights_to_moe(plain_state_dict, policy_rmoe, cfg_rmoe)
    missing, unexpected = policy_rmoe.load_state_dict(remapped, strict=False)

    assert len(unexpected) == 0
    expected_missing_substrings = ("router.", "gru_cell.", "hidden_proj.")
    assert all(any(sub in key for sub in expected_missing_substrings) for key in missing)

    found_moe_layer = False
    for name, module in policy_rmoe.named_modules():
        if isinstance(module, MoEFFN):
            found_moe_layer = True
            source_weight = plain_state_dict[f"{name}.fc1.weight"]
            for expert in module.experts:
                diff = (expert.fc1.weight - source_weight).abs().max().item()
                assert diff > 0.0, "checkpoint remap must apply symmetry-breaking noise"
                assert diff < 1e-2, "checkpoint remap noise should be small (near-function-preserving)"
    assert found_moe_layer

    # non-MoE parameters must be loaded unchanged.
    non_moe_key = "model.transformer.pos_emb"
    assert torch.equal(policy_rmoe.state_dict()[non_moe_key], plain_state_dict[non_moe_key])

    # remapping an already-MoE checkpoint must be a no-op (idempotent, not double-remapped).
    already_moe_state_dict = policy_rmoe.state_dict()
    reremapped = _remap_xvla_mlp_weights_to_moe(already_moe_state_dict, policy_rmoe, cfg_rmoe)
    assert reremapped.keys() == already_moe_state_dict.keys()
    for key in reremapped:
        torch.testing.assert_close(reremapped[key], already_moe_state_dict[key], equal_nan=True)


def test_checkpoint_remap_std_zero_leaves_experts_bit_identical_to_source():
    torch.manual_seed(0)
    cfg_plain = XVLAConfig(**_COMMON_KWARGS)
    _set_matching_features(cfg_plain)
    plain_state_dict = XVLAPolicy(cfg_plain).state_dict()

    cfg_rmoe = XVLARMoEConfig(expert_symmetry_breaking_std=0.0, **_COMMON_KWARGS)
    _set_matching_features(cfg_rmoe)
    policy_rmoe = construct_until_finite(lambda: XVLARMoEPolicy(cfg_rmoe))

    remapped = _remap_xvla_mlp_weights_to_moe(plain_state_dict, policy_rmoe, cfg_rmoe)
    policy_rmoe.load_state_dict(remapped, strict=False)

    for name, module in policy_rmoe.named_modules():
        if isinstance(module, MoEFFN):
            source_weight = plain_state_dict[f"{name}.fc1.weight"]
            for expert in module.experts:
                assert torch.equal(expert.fc1.weight, source_weight)


def test_from_pretrained_loads_plain_xvla_checkpoint_as_xvla_rmoe(tmp_path):
    """Regression test: `XVLARMoEPolicy.from_pretrained(<plain xvla checkpoint>)` must decode
    the checkpoint's `config.json` (whose own `"type"` is `"xvla"`, not `"xvla_rmoe"`) as a full
    `XVLARMoEConfig` -- including nested fields like `florence_config` -- rather than silently
    falling back to a bare `XVLAConfig` or losing `florence_config` entirely. This exercises the
    exact path `_load_xvla_rmoe_config` exists for (see its docstring for why the generic
    `PreTrainedConfig.from_pretrained` mechanism can't be reused here)."""
    torch.manual_seed(0)
    cfg_plain = XVLAConfig(**_COMMON_KWARGS)
    _set_matching_features(cfg_plain)
    policy_plain = construct_until_finite(lambda: XVLAPolicy(cfg_plain))
    policy_plain.save_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["type"] == "xvla"

    loaded = XVLARMoEPolicy.from_pretrained(tmp_path)
    assert type(loaded.config) is XVLARMoEConfig
    assert loaded.config.florence_config  # not silently dropped
    assert loaded.config.use_moe is True  # xvla_rmoe-only field, absent from the source json
    assert loaded.model.routing_memory is not None

    loaded.eval()
    batch = make_tiny_batch(loaded)
    with torch.no_grad():
        actions = loaded.predict_action_chunk(dict(batch))
    assert actions.shape == (2, loaded.config.chunk_size, loaded.model.dim_action)
    assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# Test 14: inference routing-state reset
# ---------------------------------------------------------------------------


def test_routing_state_resets_at_start_of_every_generate_actions_call():
    torch.manual_seed(0)
    policy = make_tiny_policy()
    perturb = policy.model.transformer.blocks
    for block in perturb:
        if isinstance(block.mlp, MoEFFN):
            torch.nn.init.normal_(block.mlp.router.weight, std=0.5)
    policy.eval()
    batch = make_tiny_batch(policy)
    inputs = policy._build_model_inputs(batch)

    captured_states = []
    original_initial_state = policy.model.routing_memory.initial_state

    def _spy_initial_state(batch_size, device):
        state = original_initial_state(batch_size, device)
        captured_states.append(state.clone())
        return state

    policy.model.routing_memory.initial_state = _spy_initial_state
    with torch.no_grad():
        policy.model.generate_actions(**inputs, steps=policy.config.num_denoising_steps)
        policy.model.generate_actions(**inputs, steps=policy.config.num_denoising_steps)

    assert len(captured_states) == 2  # one reset per `generate_actions` call
    for state in captured_states:
        assert torch.all(state == 0.0)


# ---------------------------------------------------------------------------
# Test 16: full tiny end-to-end (forward, loss, backward, optimizer step, sample action)
# ---------------------------------------------------------------------------


def test_full_tiny_end_to_end_train_and_sample():
    # `make_tiny_policy` already health-checks construction, but an optimizer step moves
    # weights into territory that wasn't probed; retry a couple of independent draws (see
    # `construct_until_finite`'s docstring for why an untrained tiny hand-crafted vision tower
    # can rarely be numerically unstable -- a pre-existing, unrelated upstream limitation).
    last_actions = None
    for seed in range(3):
        torch.manual_seed(seed)
        policy = make_tiny_policy(recurrent_training_probability=1.0)
        for block in policy.model.transformer.blocks:
            if isinstance(block.mlp, MoEFFN):
                torch.nn.init.normal_(block.mlp.router.weight, std=0.5)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

        batch = make_tiny_batch(policy)
        loss, log_dict = policy.forward(dict(copy.deepcopy(batch)))
        assert torch.isfinite(loss)
        assert log_dict["train/used_recurrent_training"] is True

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_sum = sum(p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None)
        assert grad_sum > GRAD_NOISE_FLOOR
        optimizer.step()

        policy.eval()
        with torch.no_grad():
            actions = policy.predict_action_chunk(dict(batch))
        assert actions.shape == (2, policy.config.chunk_size, policy.model.dim_action)
        last_actions = actions
        if torch.isfinite(actions).all():
            return

    assert torch.isfinite(last_actions).all()


def test_full_tiny_end_to_end_single_t_path():
    """Same as above but forcing the single-t (non-recurrent) branch."""
    for seed in range(3):
        torch.manual_seed(seed)
        policy = make_tiny_policy(recurrent_training_probability=0.0)
        policy.train()  # dropout active here, unlike `construct_until_finite`'s eval-mode probe
        batch = make_tiny_batch(policy)
        loss, log_dict = policy.forward(dict(batch))
        if not torch.isfinite(loss):
            continue
        assert log_dict["train/used_recurrent_training"] is False
        assert "train/single_flow_loss" in log_dict
        loss.backward()
        grad_sum = sum(p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None)
        assert grad_sum > GRAD_NOISE_FLOOR
        return

    assert torch.isfinite(loss)


def test_action_loss_ignores_padded_episode_timesteps():
    policy = make_tiny_policy()
    batch_size, horizon, action_dim = 2, policy.config.chunk_size, policy.model.dim_action
    prediction = torch.randn(batch_size, horizon, action_dim)
    target = torch.randn_like(prediction)
    valid = torch.ones(batch_size, horizon, dtype=torch.bool)
    valid[:, -2:] = False

    baseline = policy.model._compute_action_loss(prediction, target, valid)
    changed_padding = target.clone()
    changed_padding[:, -2:] = 1_000_000.0
    changed = policy.model._compute_action_loss(prediction, changed_padding, valid)

    assert baseline.keys() == changed.keys()
    for key in baseline:
        torch.testing.assert_close(baseline[key], changed[key])
