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
"""MolmoAct2's AdamW, which clips each parameter group and compensates bf16 rounding, and its LR schedule."""

import copy
import math

import pytest
import torch

from lerobot.optim import (
    MolmoAct2AdamWConfig,
    MolmoAct2CosineWithWarmupSchedulerConfig,
    load_optimizer_state,
    save_optimizer_state,
)
from lerobot.policies.molmoact2.optimizer_molmoact2 import MolmoAct2AdamW
from tests.utils import require_cuda


def _lr_multipliers(config: MolmoAct2CosineWithWarmupSchedulerConfig, num_training_steps: int):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1.0)
    return config.build(optimizer, num_training_steps=num_training_steps).lr_lambdas[0]


def test_scheduler_keeps_an_explicit_decay_endpoint_when_training_stops_early():
    config = MolmoAct2CosineWithWarmupSchedulerConfig(
        num_warmup_steps=200, num_decay_steps=30_000, alpha_f=0.1
    )
    multiplier = _lr_multipliers(config, num_training_steps=3_000)

    # A short diagnostic run follows the start of the full schedule instead of being compressed to the floor.
    expected = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (3_000 - 200) / (30_000 - 200)))
    assert multiplier(2_999) == pytest.approx(expected)
    assert multiplier(29_999) == pytest.approx(0.1)


def test_scheduler_resume_matches_uninterrupted_updates():
    config = MolmoAct2CosineWithWarmupSchedulerConfig(num_warmup_steps=7, num_decay_steps=41, alpha_f=0.1)

    def make_state(value):
        parameter = torch.nn.Parameter(value.clone())
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.9, 0.95), eps=1e-6)
        return parameter, optimizer, config.build(optimizer, num_training_steps=41)

    def run(parameter, optimizer, scheduler, start, stop):
        for step in range(start, stop):
            parameter.grad = torch.tensor([0.25 + step / 128, -0.5 + (step % 5) / 32])
            optimizer.step()
            scheduler.step()

    initial = torch.tensor([0.75, -0.125])
    full = make_state(initial)
    run(*full, 0, 31)
    split = make_state(initial)
    run(*split, 0, 13)
    resumed = make_state(split[0].detach())
    resumed[1].load_state_dict(copy.deepcopy(split[1].state_dict()))
    resumed[2].load_state_dict(copy.deepcopy(split[2].state_dict()))
    run(*resumed, 13, 31)

    assert torch.equal(full[0], resumed[0])
    assert full[2].state_dict() == resumed[2].state_dict()


def _effective_parameter(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter) -> torch.Tensor:
    """The BF16 parameter plus its Kahan residual, i.e. the value an FP32 optimizer would hold."""
    compensation = optimizer.state.get(parameter, {}).get("compensation")
    value = parameter.detach().float()
    return value if compensation is None else value + compensation.float()


def _assert_same_adam_state(
    optimizer, parameter, reference, reference_parameter, names=("step", "exp_avg", "exp_avg_sq")
):
    for name in names:
        assert torch.equal(optimizer.state[parameter][name], reference.state[reference_parameter][name])


def test_adamw_clips_each_group_independently():
    llm, action_expert = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    optimizer = MolmoAct2AdamWConfig(lr=0.0, group_grad_clip_norm=1.0).build(
        [{"params": [llm]}, {"params": [action_expert]}]
    )
    assert isinstance(optimizer, MolmoAct2AdamW)
    assert [group["foreach"] for group in optimizer.param_groups] == [None, None]
    assert [group["fused"] for group in optimizer.param_groups] == [None, None]
    llm.grad, action_expert.grad = torch.tensor([3.0, 4.0]), torch.tensor([0.0, 5.0])

    optimizer.step()

    assert torch.linalg.vector_norm(llm.grad).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(action_expert.grad).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="group_grad_clip_norm"):
        MolmoAct2AdamW([llm], group_grad_clip_norm=0.0)


def test_adamw_skips_the_whole_step_when_any_group_is_not_finite():
    finite, nonfinite = (
        torch.nn.Parameter(torch.tensor([1.0, -0.5])),
        torch.nn.Parameter(torch.tensor([0.75, -1.25])),
    )
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=1.0).build(
        [{"params": [finite]}, {"params": [nonfinite]}]
    )
    finite.grad, nonfinite.grad = torch.tensor([3.0, 4.0]), torch.tensor([float("nan"), 1.0])
    optimizer.step()
    # No state is created by a skipped first step.
    assert finite not in optimizer.state and nonfinite not in optimizer.state
    assert finite.grad is None and nonfinite.grad is None

    for param in (finite, nonfinite):
        param.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    params_before = [finite.detach().clone(), nonfinite.detach().clone()]
    state_before = {param: copy.deepcopy(optimizer.state[param]) for param in (finite, nonfinite)}
    finite.grad, nonfinite.grad = torch.tensor([0.5, -0.25]), torch.tensor([float("inf"), 0.0])
    optimizer.step()

    # Neither the parameters nor the Adam moments and step counts advance.
    for param, before in zip((finite, nonfinite), params_before, strict=True):
        assert torch.equal(param, before) and param.grad is None
        assert all(torch.equal(optimizer.state[param][k], v) for k, v in state_before[param].items())


@pytest.mark.parametrize(("amsgrad", "maximize"), [(False, False), (True, True)])
def test_adamw_matches_native_adamw_and_compensates_bfloat16(amsgrad, maximize):
    """FP32 parameters take native AdamW steps; BF16 ones keep native BF16 moments plus a BF16 residual."""
    kwargs = {"lr": 7e-4, "betas": (0.7, 0.91), "eps": 1e-6, "weight_decay": 0.03, "amsgrad": amsgrad}
    kwargs.update(maximize=maximize, foreach=False)
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32)
    initial = [
        torch.linspace(-0.75, 0.75, steps=size, dtype=dtype)
        for size, dtype in zip((17, 9, 3), dtypes, strict=True)
    ]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    native_params = [torch.nn.Parameter(value.clone()) for value in initial]
    oracle_params = [torch.nn.Parameter(value.float()) for value in initial]
    optimizer = MolmoAct2AdamW(params, **kwargs, group_grad_clip_norm=1e6)
    native = torch.optim.AdamW(native_params, **kwargs)
    oracle = torch.optim.AdamW(oracle_params, **kwargs)

    for step in range(7):
        for index, (param, native_param, oracle_param) in enumerate(
            zip(params, native_params, oracle_params, strict=True)
        ):
            grad = torch.linspace(-0.5 + step / 32, 0.25 + step / 64, steps=param.numel(), dtype=param.dtype)
            # The middle parameter skips every other step.
            grad = None if index == 1 and step % 2 else grad
            param.grad = None if grad is None else grad.clone()
            native_param.grad = None if grad is None else grad.clone()
            oracle_param.grad = None if grad is None else grad.float()
        optimizer.step()
        native.step()
        oracle.step()

    names = ("step", "exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ())
    for param, native_param, oracle_param in zip(params, native_params, oracle_params, strict=True):
        _assert_same_adam_state(optimizer, param, native, native_param, names)
        assert optimizer.state[param]["exp_avg"].dtype == param.dtype
        if param.dtype == torch.float32:
            assert torch.equal(param, native_param) and "compensation" not in optimizer.state[param]
            continue
        # The residual keeps sub-ULP updates: closer to the FP32 oracle than native BF16 AdamW.
        assert optimizer.state[param]["compensation"].dtype == torch.bfloat16
        compensated = torch.linalg.vector_norm(_effective_parameter(optimizer, param) - oracle_param.detach())
        assert compensated < torch.linalg.vector_norm(native_param.detach().float() - oracle_param.detach())


def test_adamw_resume_is_bitwise_and_keeps_the_groups(tmp_path):
    preset = MolmoAct2AdamWConfig(lr=7e-4, betas=(0.7, 0.91), weight_decay=0.03, group_grad_clip_norm=100.0)
    initial = [torch.tensor([1.0, -0.5], dtype=torch.bfloat16), torch.tensor([0.75, -1.25])]

    def build(values):
        params = [torch.nn.Parameter(value.detach().clone()) for value in values]
        groups = [
            {"params": [params[0]], "group_name": "llm"},
            {"params": [params[1]], "group_name": "expert"},
        ]
        return params, preset.build(groups)

    params, optimizer = build(initial)
    grads = [torch.tensor([0.25 + i / 32, -0.5 + i / 64]) for i in range(12)]
    for grad in grads[:5]:
        for param in params:
            param.grad = grad.to(param.dtype)
        optimizer.step()
    save_optimizer_state(optimizer, tmp_path)
    resumed_params, resumed = build(params)
    load_optimizer_state(resumed, tmp_path)
    for grad in grads[5:]:
        for param in (*params, *resumed_params):
            param.grad = grad.to(param.dtype)
        optimizer.step()
        resumed.step()

    assert [group["group_name"] for group in resumed.param_groups] == ["llm", "expert"]
    for param, resumed_param in zip(params, resumed_params, strict=True):
        assert torch.equal(param, resumed_param)
        assert optimizer.state[param].keys() == resumed.state[resumed_param].keys()
        _assert_same_adam_state(resumed, resumed_param, optimizer, param, names=tuple(optimizer.state[param]))


def test_adamw_loads_native_adamw_state():
    legacy_param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    legacy = torch.optim.AdamW([legacy_param], lr=1e-3, betas=(0.9, 0.95), eps=1e-6)
    legacy_param.grad = torch.ones_like(legacy_param)
    legacy.step()
    param = torch.nn.Parameter(legacy_param.detach().clone())
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=100.0).build([param])
    optimizer.load_state_dict(legacy.state_dict())

    param.grad = torch.ones_like(param)
    optimizer.step()

    assert optimizer.state[param]["step"].item() == 2
    assert optimizer.state[param]["compensation"].dtype == torch.bfloat16


def test_adamw_leaves_parameters_without_grad_untouched():
    updated = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    untouched = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16))
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, weight_decay=0.1, group_grad_clip_norm=100.0).build(
        [updated, untouched]
    )
    updated.grad = torch.ones_like(updated)

    optimizer.step()

    # No state and no weight decay for a parameter without a gradient.
    assert updated in optimizer.state and untouched not in optimizer.state
    assert torch.equal(untouched, torch.tensor([2.0], dtype=torch.bfloat16))


def test_adamw_bfloat16_tracks_fp32_over_a_long_run():
    kwargs = {"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-6, "weight_decay": 0.01}
    initial = torch.tensor([0.75, -0.5, 0.125], dtype=torch.bfloat16)
    param, native_param = (torch.nn.Parameter(initial.clone()) for _ in range(2))
    oracle_param = torch.nn.Parameter(initial.float())
    optimizer = MolmoAct2AdamWConfig(**kwargs, group_grad_clip_norm=100.0).build([param])
    native = torch.optim.AdamW([native_param], **kwargs)
    oracle = torch.optim.AdamW([oracle_param], **kwargs)

    for step in range(1, 1_001):
        grad = torch.tensor(
            [0.125 + (step % 17 - 8) / 256, -0.25 + (step % 13 - 6) / 512, 0.0625 - (step % 11 - 5) / 1024],
            dtype=torch.bfloat16,
        )
        param.grad, native_param.grad, oracle_param.grad = grad.clone(), grad.clone(), grad.float()
        optimizer.step()
        native.step()
        oracle.step()

    compensated = torch.linalg.vector_norm(_effective_parameter(optimizer, param) - oracle_param.detach())
    assert compensated < 0.1 * torch.linalg.vector_norm(native_param.detach().float() - oracle_param.detach())
    _assert_same_adam_state(optimizer, param, native, native_param)


@require_cuda
def test_adamw_uses_native_cuda_foreach(monkeypatch):
    bfloat16_param = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=torch.bfloat16))
    float32_param = torch.nn.Parameter(torch.ones(16, device="cuda"))
    optimizer = MolmoAct2AdamWConfig(lr=1e-3, group_grad_clip_norm=100.0).build(
        [bfloat16_param, float32_param]
    )
    observed_foreach = []
    native_step = torch.optim.AdamW.step

    def record_native_step(native_optimizer, *args, **kwargs):
        observed_foreach.extend(group["foreach"] for group in native_optimizer.param_groups)
        return native_step(native_optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_native_step)
    bfloat16_param.grad = torch.ones_like(bfloat16_param)
    float32_param.grad = torch.ones_like(float32_param)
    optimizer.step()

    assert observed_foreach == [None]
    assert optimizer.state[bfloat16_param]["compensation"].dtype == torch.bfloat16
    assert "compensation" not in optimizer.state[float32_param]
    assert torch.isfinite(bfloat16_param).all() and torch.isfinite(float32_param).all()
