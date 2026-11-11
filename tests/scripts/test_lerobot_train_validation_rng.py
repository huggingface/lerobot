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

import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.scripts import lerobot_train as train_module  # noqa: E402
from lerobot.scripts.lerobot_train import _evaluate_held_out, update_policy  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig  # noqa: E402

pytestmark = pytest.mark.dataset


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _assert_rng(state):
    actual = _rng_state()
    assert actual[0] == state[0]
    assert actual[1][0] == state[1][0]
    np.testing.assert_array_equal(actual[1][1], state[1][1])
    assert actual[1][2:] == state[1][2:]
    assert torch.equal(actual[2], state[2])


def _draw():
    return random.random(), np.random.rand(), torch.rand(()).item()


@pytest.fixture(autouse=True)
def restore_rng():
    original = _rng_state()
    random.seed(123)
    np.random.seed(123)
    torch.random.default_generator.manual_seed(123)
    try:
        yield
    finally:
        _restore_rng(original)


class StubAccelerator:
    sync_gradients = True
    optimizer_step_was_skipped = False
    scaler = None

    def __init__(self):
        self.reductions = []

    def autocast(self):
        return torch.autocast("cpu", dtype=torch.bfloat16)

    def accumulate(self, policy):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        return policy

    def reduce(self, value, reduction):
        self.reductions.append((value.item(), reduction))
        return value


class TinyPolicy(torch.nn.Module):
    def __init__(self, stochastic=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.stochastic = stochastic
        self.calls = []

    def forward(self, batch):
        self.calls.append((self.training, torch.is_grad_enabled(), torch.is_autocast_enabled("cpu")))
        target = batch["value"] + sum(_draw()) if self.stochastic else batch["value"]
        return ((self.weight - target) ** 2).mean(), None


class TinyDataset(Dataset):
    meta = SimpleNamespace(
        camera_keys=[],
        stats={},
        has_language_columns=False,
        episodes={"dataset_from_index": [0], "dataset_to_index": [5]},
    )
    num_frames = 5
    num_episodes = 1
    episodes = None
    absolute_to_relative_idx = None

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        return {"value": torch.tensor(float(index))}


def _preprocess(batch):
    return {"value": batch["value"] + sum(_draw())}


def _evaluate(policy, loader, accelerator, preprocessor=_preprocess):
    return _evaluate_held_out(policy, loader, [], {}, preprocessor, accelerator, torch.device("cpu"))


def _tracker():
    return MetricsTracker(
        1,
        5,
        1,
        {name: AverageMeter(name, ":.3f") for name in ("loss", "lr", "update_s")},
    )


@pytest.mark.parametrize("private_loader_generator", [False, True])
def test_validation_preserves_next_draw_and_optimizer_update(monkeypatch, private_loader_generator):
    # An explicit generator remains loader-owned; only global RNGs are restored.
    initial = _rng_state()

    def run(validate):
        _restore_rng(initial)
        policy = TinyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        accelerator = StubAccelerator()
        generator = torch.Generator().manual_seed(456) if private_loader_generator else None
        generator_before = generator.get_state() if generator is not None else None
        loader = DataLoader(TinyDataset(), batch_size=2, num_workers=0, generator=generator)
        loss = _evaluate(policy, loader, accelerator) if validate else None
        draws = _draw()
        update_policy(_tracker(), policy, {"value": torch.ones(2)}, optimizer, 0, accelerator)
        if validate:
            assert policy.calls[:-1] == [(False, False, True)] * 3
            assert len(accelerator.reductions) == 1
            if generator is not None:
                assert not torch.equal(generator.get_state(), generator_before)
        return draws, policy.weight.detach().clone(), loss

    omitted = run(False)
    preserved = run(True)
    assert preserved[0] == omitted[0]
    assert torch.equal(preserved[1], omitted[1])
    monkeypatch.setattr(train_module, "preserve_rng_state", lambda device: nullcontext())
    unprotected = run(True)
    assert all(a != b for a, b in zip(unprotected[0], omitted[0], strict=True))
    assert not torch.equal(unprotected[1], omitted[1])
    assert unprotected[2] == preserved[2]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("stage", ["iterator", "iteration", "preprocessor", "forward", "reduction"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_validation_failure_restores_rng_mode_and_contexts(monkeypatch, training, stage, error_type):
    policy = TinyPolicy().train(training)
    accelerator = StubAccelerator()
    error = error_type("validation failed")

    def fail(*args, **kwargs):
        _draw()
        raise error

    class FailingLoader:
        def __iter__(self):
            if stage == "iterator":
                fail()
            return self

        def __next__(self):
            fail()

    loader = DataLoader(TinyDataset(), batch_size=2)
    preprocessor = _preprocess
    if stage in ("iterator", "iteration"):
        loader = FailingLoader()
    elif stage == "preprocessor":
        preprocessor = fail
    elif stage == "forward":
        monkeypatch.setattr(policy, "forward", fail)
    else:
        monkeypatch.setattr(accelerator, "reduce", fail)
    # Existing gradients must survive evaluation, including failures.
    policy.weight.grad = torch.tensor(7.0)
    initial = _rng_state()
    grad_enabled = torch.is_grad_enabled()
    autocast_enabled = torch.is_autocast_enabled("cpu")
    with pytest.raises(error_type) as exc_info:
        _evaluate(policy, loader, accelerator, preprocessor)
    assert exc_info.value is error
    _assert_rng(initial)
    assert policy.training is training
    assert policy.weight.grad.item() == 7
    assert torch.is_grad_enabled() is grad_enabled
    assert torch.is_autocast_enabled("cpu") is autocast_enabled


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_validation_preserves_metrics_hooks_and_mode(monkeypatch, training, empty):
    policy = TinyPolicy(stochastic=False).train(training)
    accelerator = StubAccelerator()
    hook_calls = []
    policy.register_forward_pre_hook(lambda *args: hook_calls.append("pre"))
    policy.register_forward_hook(lambda *args: hook_calls.append("post"))
    loader = DataLoader([] if empty else TinyDataset(), batch_size=2)
    # Include mode changes and reduction in the preservation boundary too.
    original_train = policy.train

    def change_mode(mode=True):
        _draw()
        return original_train(mode)

    monkeypatch.setattr(policy, "train", change_mode)
    original_reduce = accelerator.reduce

    def reduce(value, reduction):
        _draw()
        return original_reduce(value, reduction) + 1

    monkeypatch.setattr(accelerator, "reduce", reduce)
    initial = _rng_state()
    loss = _evaluate(policy, loader, accelerator, preprocessor=lambda batch: batch)
    # Batch means are 0.25, 4.25, 12.25, deliberately NOT the sample-weighted mean.
    expected = 0.0 if empty else (0.25 + 4.25 + 12.25) / 3
    assert accelerator.reductions == [(pytest.approx(expected), "mean")]
    assert loss == pytest.approx(expected + 1)
    assert hook_calls == ([] if empty else ["pre", "post"] * 3)
    assert policy.calls == ([] if empty else [(False, False, True)] * 3)
    assert policy.training is training
    assert policy.weight.grad is None
    _assert_rng(initial)


@pytest.mark.parametrize("main_process", [True, False])
@pytest.mark.parametrize(
    "eval_steps, has_loader, expected_steps",
    [
        (2, True, [2, 4]),
        (None, True, []),
        (10, True, []),
        (2, False, []),
    ],
)
def test_train_validation_wiring(
    monkeypatch,
    tmp_path,
    caplog,
    restore_torch_backend_flags,
    main_process,
    eval_steps,
    has_loader,
    expected_steps,
):
    # Keep this skip local: the evaluator regressions run in the dataset tier without Accelerate.
    accelerate = pytest.importorskip("accelerate", reason="train integration requires accelerate-dep")
    accelerator = accelerate.Accelerator(cpu=True, mixed_precision="no")
    policy = TinyPolicy()
    policy.config = DummyCheckpointConfig(device="cpu", push_to_hub=False)
    dataset = TinyDataset()
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/in-memory", eval_split=0.2),
        policy=policy.config,
        output_dir=tmp_path / "train",
        num_workers=0,
        batch_size=2,
        steps=4,
        env_eval_freq=0,
        log_freq=0,
        save_checkpoint=False,
    )
    cfg.trackio.enable = False
    if eval_steps is not None:
        cfg.eval_steps = eval_steps
    else:
        assert cfg.eval_steps == 0
    monkeypatch.setattr("sys.argv", ["lerobot-train"])
    # Keep pytest's log capture; production init_logging replaces all root handlers.
    monkeypatch.setattr(train_module, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(train_module, "is_main_process", lambda: main_process)

    def make_accelerator(config):
        config.parallelism.resolve(1)
        return accelerator

    monkeypatch.setattr(train_module, "make_accelerator", make_accelerator)
    monkeypatch.setattr(
        train_module, "make_train_eval_datasets", lambda cfg: (dataset, dataset if has_loader else None)
    )
    monkeypatch.setattr(train_module, "make_policy", lambda **kwargs: policy)
    monkeypatch.setattr(train_module, "make_pre_post_processors", lambda **kwargs: (_preprocess, _preprocess))
    monkeypatch.setattr(
        train_module,
        "make_optimizer_and_scheduler",
        lambda cfg, model: (torch.optim.SGD(model.parameters(), lr=0.01), None),
    )
    evaluation_steps = []
    reductions = []
    original_reduce = accelerator.reduce

    def reduce(value, reduction):
        reductions.append(reduction)
        return original_reduce(value, reduction=reduction)

    monkeypatch.setattr(accelerator, "reduce", reduce)

    def evaluate(*args, **kwargs):
        evaluation_steps.append(sum(training for training, _, _ in policy.calls))
        initial = _rng_state()
        result = _evaluate_held_out(*args, **kwargs)
        _assert_rng(initial)
        return result

    monkeypatch.setattr(train_module, "_evaluate_held_out", evaluate)
    initial_weight = policy.weight.detach().clone()
    with caplog.at_level("INFO"):
        train_module.train(cfg)
    assert evaluation_steps == expected_steps
    assert reductions == ["mean"] * len(expected_steps)
    assert sum(training for training, _, _ in policy.calls) == cfg.steps
    assert not torch.equal(policy.weight, initial_weight)
    assert policy.training
    eval_logs = [record.message for record in caplog.records if "eval_loss=" in record.message]
    assert len(eval_logs) == (len(expected_steps) if main_process else 0)
    for message, step in zip(eval_logs, expected_steps, strict=main_process):
        assert message.startswith(f"step {step}: eval_loss=")
