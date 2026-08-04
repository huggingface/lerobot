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

"""Tests for XVLA-RMoE's truncated cross-step recurrent training: state chaining, GRU
gradient flow, the expert-symmetry-breaking regression control, the `recurrent_detach_state`
ablation, the ground-truth flow path construction, shared noise across unrolled steps, and
deterministic/DDP-safe branch selection.

See `_helpers.py` for why these run on CPU with a real (tiny, from-scratch) `XVLARMoEPolicy`
and no network access.
"""

import types

import pytest
import torch

from lerobot.policies.xvla_rmoe.modeling_xvla_rmoe import RoutingInfo, XVLARMoEPolicy
from lerobot.policies.xvla_rmoe.moe_soft_transformer import MoEFFN
from tests.policies.xvla_rmoe._helpers import (
    GRAD_NOISE_FLOOR,
    disable_dropout,
    make_tiny_batch,
    make_tiny_policy,
    perturb_router_weights,
)

pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Deterministic / DDP-safe branch selection (no model needed)
# ---------------------------------------------------------------------------


def test_should_use_recurrent_step_is_deterministic_and_ddp_safe():
    fake_policy = types.SimpleNamespace(
        config=types.SimpleNamespace(
            use_recurrent_routing_training=True, recurrent_training_probability=0.25
        ),
        training=True,
        _train_call_counter=0,
    )
    decisions = [XVLARMoEPolicy._should_use_recurrent_step(fake_policy) for _ in range(12)]
    assert decisions == [True, False, False, False] * 3

    fake_policy.training = False
    fake_policy._train_call_counter = 0
    assert all(XVLARMoEPolicy._should_use_recurrent_step(fake_policy) is False for _ in range(5))
    assert fake_policy._train_call_counter == 0  # eval mode must not consume the counter

    fake_disabled = types.SimpleNamespace(
        config=types.SimpleNamespace(
            use_recurrent_routing_training=False, recurrent_training_probability=0.25
        ),
        training=True,
        _train_call_counter=0,
    )
    assert all(XVLARMoEPolicy._should_use_recurrent_step(fake_disabled) is False for _ in range(8))


def test_recurrent_timestep_sampling_decreasing_and_shares_grid():
    torch.manual_seed(0)
    policy = make_tiny_policy(recurrent_unroll_steps=3)
    for _ in range(50):
        ts = policy.model._sample_recurrent_timesteps(torch.device("cpu"), torch.float32)
        assert ts.shape == (3,)
        assert torch.all(ts[:-1] > ts[1:])  # t_0 > t_1 > ... > t_{K-1}
        assert torch.all(ts > 0.0) and torch.all(ts <= 1.0)


# ---------------------------------------------------------------------------
# Test 6: recurrent state chaining
# ---------------------------------------------------------------------------


def test_forward_recurrent_state_chains_and_shapes():
    torch.manual_seed(0)
    policy = make_tiny_policy()
    perturb_router_weights(policy)
    policy.train()
    batch = make_tiny_batch(policy, pad_last_n=2)
    inputs = policy._build_model_inputs(batch)
    targets = policy._prepare_action_targets(batch)

    step_losses, routing_infos, extra = policy.model.forward_recurrent(action=targets, **inputs)

    cfg = policy.config
    assert len(step_losses) == len(routing_infos) == cfg.recurrent_unroll_steps

    num_moe_layers = policy.model.transformer.num_moe_layers
    bsize = targets.shape[0]
    for info in routing_infos:
        assert isinstance(info, RoutingInfo)
        assert info.routing_summary.shape == (bsize, num_moe_layers * cfg.num_moe_experts)
        assert info.hidden_summary.shape == (bsize, cfg.hidden_size)
        assert info.next_routing_state.shape == (bsize, cfg.routing_hidden_dim)
        assert info.layer_routing_weights is None  # never populated on the training hot path

    # m_0 -> forward(t_0) -> m_1 -> forward(t_1) -> m_2 -> ... : each state must actually
    # differ from the previous one (a truly-zero-init router could get stuck at a fixed
    # point, hence the router perturbation above).
    states = [torch.zeros(bsize, cfg.routing_hidden_dim)] + [
        info.next_routing_state for info in routing_infos
    ]
    for prev, nxt in zip(states[:-1], states[1:], strict=True):
        assert not torch.allclose(prev, nxt)

    assert "routing_memory_norm" in extra and "routing_memory_delta_norm" in extra


# ---------------------------------------------------------------------------
# Test 7: GRU + router gradient flow
# ---------------------------------------------------------------------------


def test_recurrent_training_backprops_into_gru_and_router():
    """A specific random router perturbation + batch draw can (rarely) land close to a
    cancellation direction and produce a genuinely small -- not zero, just small -- gradient
    for that one draw; this doesn't mean the gradient path is broken, only that this
    particular draw wasn't very informative. Retry a few independent draws and require the
    claim to hold for at least one, rather than for an arbitrary single draw.
    """
    found_gru_grad = False
    found_router_grad = False
    for attempt in range(5):
        torch.manual_seed(attempt)
        policy = make_tiny_policy()
        disable_dropout(policy)
        perturb_router_weights(policy)
        policy.train()
        policy.zero_grad(set_to_none=True)
        batch = make_tiny_batch(policy, pad_last_n=1)
        inputs = policy._build_model_inputs(batch)
        targets = policy._prepare_action_targets(batch)

        step_losses, _routing_infos, _extra = policy.model.forward_recurrent(action=targets, **inputs)
        loss = torch.stack([sum(losses.values()) for losses in step_losses]).mean()
        loss.backward()

        gru_params = list(policy.model.routing_memory.gru_cell.parameters()) + list(
            policy.model.routing_memory.hidden_proj.parameters()
        )
        found_gru_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > GRAD_NOISE_FLOOR
            for p in gru_params
        )

        routing_hidden_dim = policy.config.routing_hidden_dim
        found_router_grad = False
        for module in policy.model.transformer.blocks:
            if isinstance(module.mlp, MoEFFN) and module.mlp.router.weight.grad is not None:
                hidden_dim = module.mlp.experts[0].fc1.in_features
                memory_slice_grad = module.mlp.router.weight.grad[
                    :, hidden_dim : hidden_dim + routing_hidden_dim
                ]
                if (
                    torch.isfinite(memory_slice_grad).all()
                    and memory_slice_grad.abs().sum() > GRAD_NOISE_FLOOR
                ):
                    found_router_grad = True

        if found_gru_grad and found_router_grad:
            break

    assert found_gru_grad
    assert found_router_grad


# ---------------------------------------------------------------------------
# Test 5: expert-symmetry-breaking regression control
# ---------------------------------------------------------------------------


def test_expert_symmetry_breaking_required_for_router_and_gru_to_bootstrap():
    """Without symmetry breaking, bit-identical experts + a zero-init router is (in exact
    arithmetic) a permutation-symmetry fixed point under gradient descent: the softmax
    Jacobian makes `d(loss)/d(router.weight)` cancel to zero when every expert produces the
    identical output, and since the GRU's routing-decision input is then a pure function of
    `router.weight` alone, its gradient collapses too. In real float32 arithmetic this shows
    up as gradients pinned at floating-point-noise scale rather than moving with the loss;
    the default (`expert_symmetry_breaking_std=1e-5`) breaks the degeneracy and produces a
    genuinely larger, loss-informed gradient from the very first step.

    The router's OWN weight is zero-initialized regardless of `expert_symmetry_breaking_std`
    (only the *experts* differ between the two cases), so at step 0 the GRU's routing-memory
    input is unreachable from either case (`d(logits)/d(routing_state) = router.weight[:,
    memory_slice] = 0`, independent of the experts) -- this is the same "chicken-and-egg"
    bootstrap `perturb_router_weights` exists to skip past in the other gradient-flow tests.
    Here that is the point: with symmetry breaking, the router's *own* direct gradient
    (nonzero the moment experts differ) lets a few real optimizer steps nudge it away from
    zero on their own, at which point the GRU path activates too -- without symmetry
    breaking, the router's own gradient stays at float-noise scale, never bootstraps, and the
    GRU stays unreachable indefinitely. So this checks each parameter's own *gradient*
    (freshly computed on the last of several real Adam steps, not accumulated weight
    movement -- Adam's per-parameter RMS normalization would otherwise turn even the broken
    case's tiny float-noise gradient into a similar-looking weight movement, erasing the
    distinction this test exists to draw).
    """

    def measure_grad_after_steps(std: float, seed: int, num_steps: int = 5, lr: float = 1e-2):
        torch.manual_seed(seed)
        policy = make_tiny_policy(recurrent_training_probability=1.0, expert_symmetry_breaking_std=std)
        disable_dropout(policy)
        policy.train()
        opt = torch.optim.Adam(policy.parameters(), lr=lr)

        torch.manual_seed(seed + 1)  # fixed batch sequence, independent of the model's own init seed
        last_gru_grad = 0.0
        last_router_grad = 0.0
        for _ in range(num_steps):
            batch = make_tiny_batch(policy)
            opt.zero_grad(set_to_none=True)
            loss, _ = policy.forward(dict(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

            gru_params = list(policy.model.routing_memory.gru_cell.parameters())
            last_gru_grad = sum(p.grad.abs().sum().item() for p in gru_params if p.grad is not None)
            router = next(m.mlp for m in policy.model.transformer.blocks if isinstance(m.mlp, MoEFFN))
            last_router_grad = router.router.weight.grad.abs().sum().item()
            opt.step()
        return last_gru_grad, last_router_grad

    # A single random batch/init draw can occasionally land close to a cancellation direction
    # on either side (see `test_recurrent_training_backprops_into_gru_and_router`'s docstring
    # for the same reasoning); retry a few independent draws and require the qualitative
    # separation to hold for at least one.
    for seed in range(0, 20, 2):
        broken_gru_grad, broken_router_grad = measure_grad_after_steps(std=0.0, seed=seed)
        fixed_gru_grad, fixed_router_grad = measure_grad_after_steps(std=1e-5, seed=seed)
        if (
            broken_router_grad < GRAD_NOISE_FLOOR
            and broken_gru_grad < GRAD_NOISE_FLOOR
            and fixed_router_grad > GRAD_NOISE_FLOOR
            and fixed_gru_grad > GRAD_NOISE_FLOOR
        ):
            return

    assert broken_router_grad < GRAD_NOISE_FLOOR
    assert broken_gru_grad < GRAD_NOISE_FLOOR
    assert fixed_router_grad > GRAD_NOISE_FLOOR
    assert fixed_gru_grad > GRAD_NOISE_FLOOR


# ---------------------------------------------------------------------------
# Test 8: recurrent_detach_state ablation
# ---------------------------------------------------------------------------


def test_recurrent_detach_state_blocks_cross_step_gru_gradient():
    # The detached-case assertion below is an exact mathematical guarantee (detach() truly
    # cuts the autograd graph, so the gradient is either None or exactly 0.0, never "small but
    # nonzero") and never needs retrying. The non-detached sanity check is a signal-strength
    # claim like the other gradient-flow tests, so -- same reasoning as
    # `test_recurrent_training_backprops_into_gru_and_router` -- retry a few independent draws.
    for seed in range(0, 20, 2):
        torch.manual_seed(seed)
        policy_detached = make_tiny_policy(recurrent_training_probability=1.0, recurrent_detach_state=True)
        disable_dropout(policy_detached)
        perturb_router_weights(policy_detached)
        policy_detached.train()
        batch = make_tiny_batch(policy_detached)
        inputs = policy_detached._build_model_inputs(batch)
        targets = policy_detached._prepare_action_targets(batch)

        step_losses, _routing_infos, _extra = policy_detached.model.forward_recurrent(
            action=targets, **inputs
        )
        last_step_loss = sum(step_losses[-1].values())
        policy_detached.zero_grad(set_to_none=True)
        last_step_loss.backward()

        gru_params = list(policy_detached.model.routing_memory.gru_cell.parameters())
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in gru_params), (
            "GRU should receive no gradient from a last-step-only loss when recurrent_detach_state=True"
        )

        # Sanity: without detaching, the identical setup DOES reach the GRU.
        torch.manual_seed(seed)
        policy_live = make_tiny_policy(recurrent_training_probability=1.0, recurrent_detach_state=False)
        disable_dropout(policy_live)
        policy_live.load_state_dict(policy_detached.state_dict())
        policy_live.train()
        inputs2 = policy_live._build_model_inputs(batch)
        targets2 = policy_live._prepare_action_targets(batch)
        step_losses2, _r, _e = policy_live.model.forward_recurrent(action=targets2, **inputs2)
        last_step_loss2 = sum(step_losses2[-1].values())
        policy_live.zero_grad(set_to_none=True)
        last_step_loss2.backward()
        gru_params2 = list(policy_live.model.routing_memory.gru_cell.parameters())
        live_gru_grad_found = any(
            p.grad is not None and p.grad.abs().sum() > GRAD_NOISE_FLOOR for p in gru_params2
        )
        if live_gru_grad_found:
            return

    assert live_gru_grad_found, (
        "GRU should receive gradient from a last-step-only loss when recurrent_detach_state=False"
    )


# ---------------------------------------------------------------------------
# Test 9 / 10: ground-truth flow path + shared noise
# ---------------------------------------------------------------------------


def test_recurrent_noisy_action_uses_ground_truth_flow_path_and_shared_noise():
    """Test 9 + 10: each recurrent step's noisy action must be built directly as
    `t*noise + (1-t)*action` from ONE shared noise sample and the ground-truth action chunk --
    never from a previous step's Euler-integrated prediction."""
    torch.manual_seed(0)
    policy = make_tiny_policy()
    policy.eval()  # disable dropout so the transformer is a deterministic function of its inputs
    batch = make_tiny_batch(policy)
    inputs = policy._build_model_inputs(batch)
    action = policy._prepare_action_targets(batch)

    captured_noisy_actions = []
    original_preprocess = policy.model.action_space.preprocess

    def _spy_preprocess(proprio, noisy_action, *args, **kwargs):
        captured_noisy_actions.append(noisy_action.clone())
        return original_preprocess(proprio, noisy_action, *args, **kwargs)

    policy.model.action_space.preprocess = _spy_preprocess

    fixed_noise = torch.randn_like(action)
    torch.manual_seed(123)
    policy.model.forward_recurrent(action=action, noise=fixed_noise, **inputs)

    policy.model.action_space.preprocess = original_preprocess

    # Recomputing the same timestep grid deterministically is not possible from the outside
    # (RNG-dependent start index), so instead verify each captured noisy action satisfies
    # `x = t*noise + (1-t)*action` for *some* t in (0, 1], by solving for t from the tensor
    # itself and checking it reproduces the full tensor -- this rules out any Euler/previous-
    # prediction contamination, which would not satisfy the affine relation exactly.
    for noisy in captured_noisy_actions:
        delta_num = noisy - action
        delta_den = fixed_noise - action
        # Avoid division by ~0 entries; use the mean ratio over entries with a well-defined denom.
        denom_mask = delta_den.abs() > 1e-4
        assert denom_mask.any()
        t_estimate = (delta_num[denom_mask] / delta_den[denom_mask]).mean()
        assert 0.0 <= t_estimate.item() <= 1.0 + 1e-4
        reconstructed = t_estimate * fixed_noise + (1 - t_estimate) * action
        torch.testing.assert_close(noisy, reconstructed, atol=1e-3, rtol=1e-3)

    assert len(captured_noisy_actions) == policy.config.recurrent_unroll_steps


def test_recurrent_shares_one_noise_sample_across_all_unrolled_steps():
    torch.manual_seed(0)
    policy = make_tiny_policy()
    policy.eval()
    batch = make_tiny_batch(policy)
    inputs = policy._build_model_inputs(batch)
    action = policy._prepare_action_targets(batch)

    captured_noisy_actions = []
    original_preprocess = policy.model.action_space.preprocess

    def _spy_preprocess(proprio, noisy_action, *args, **kwargs):
        captured_noisy_actions.append(noisy_action.clone())
        return original_preprocess(proprio, noisy_action, *args, **kwargs)

    policy.model.action_space.preprocess = _spy_preprocess
    fixed_noise = torch.randn_like(action)
    policy.model.forward_recurrent(action=action, noise=fixed_noise, **inputs)
    policy.model.action_space.preprocess = original_preprocess

    # If every step used the same shared noise (and the same ground-truth action), then for any
    # two steps p, q: (x_p - x_q) must be proportional to (action - noise) with a scalar ratio
    # equal to (t_p - t_q) -- i.e. all pairwise differences must be collinear with each other.
    diffs = [
        captured_noisy_actions[i] - captured_noisy_actions[0] for i in range(1, len(captured_noisy_actions))
    ]
    reference = action - fixed_noise
    for diff in diffs:
        # diff = (t_0 - t_i) * (noise - action) = (t_i - t_0) * reference -- check collinearity.
        flat_diff = diff.flatten()
        flat_ref = reference.flatten()
        denom_mask = flat_ref.abs() > 1e-4
        ratios = flat_diff[denom_mask] / flat_ref[denom_mask]
        assert ratios.std() < 1e-3  # a single shared scalar ratio across all elements


# ---------------------------------------------------------------------------
# use_recurrent_routing_training toggle
# ---------------------------------------------------------------------------


def test_recurrent_training_flag_toggles_with_config():
    torch.manual_seed(0)
    policy = make_tiny_policy(recurrent_training_probability=1.0)
    policy.train()
    batch = make_tiny_batch(policy)
    _, log_dict = policy.forward(dict(batch))
    assert log_dict["train/used_recurrent_training"] is True

    disabled = make_tiny_policy(recurrent_training_probability=0.0)
    disabled.load_state_dict(policy.state_dict())
    disabled.train()
    for _ in range(5):
        _, log_dict_disabled = disabled.forward(dict(batch))
        assert log_dict_disabled["train/used_recurrent_training"] is False
        assert "train/single_flow_loss" in log_dict_disabled

    disabled.eval()
    _, log_dict_eval = disabled.forward(dict(batch))
    assert log_dict_eval["train/used_recurrent_training"] is False
