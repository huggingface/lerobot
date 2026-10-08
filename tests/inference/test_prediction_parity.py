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

"""Local RTC and the remote runner share post-preprocessing action semantics."""

import pytest
import torch

pytest.importorskip("datasets", reason="local rollout imports the dataset integration")

from lerobot.inference import ExecutionMode, predict_chunk
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import transition_to_prediction
from tests.inference.policy_helpers import (
    ConformingPolicy,
    local_engine_for,
    observation,
    runner_for,
    tiny_config,
)


class ObservedPolicy(ConformingPolicy):
    name = "observed_prediction"

    def __init__(self, config):
        super().__init__(config)
        self.context = None
        self.output = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        self.context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
        return self.output.clone() if isinstance(self.output, torch.Tensor) else self.output


def local_prediction(monkeypatch, policy, *, relative=False, previous=None, canonical_previous=None):
    """Run one real local prediction, capturing the chunk before its queue merge."""
    engine = local_engine_for(policy, relative=relative, queue_threshold=8)
    engine.resume()
    if previous is not None and len(previous):
        engine.action_queue.merge(previous, canonical_previous, real_delay=0)
    engine._runtime.turnarounds.append(1 / 30)
    engine.notify_observation(
        {"a.pos": 20.0, "b.pos": 20.0, "c.pos": 20.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    result = []

    def capture(request, chunk, *, task_version):
        result.append(chunk)
        engine._shutdown_event.set()
        return True

    monkeypatch.setattr(engine._runtime, "accept", capture)
    engine._rtc_loop()
    return engine, result


@pytest.mark.parametrize(
    "mode,relative,prefix_steps",
    [
        (ExecutionMode.CHUNK, False, 0),
        (ExecutionMode.RTC_GUIDED, True, 2),
        (ExecutionMode.RTC_TRAINED, True, 6),
    ],
)
def test_local_and_remote_prefix_context_and_execution_slice_match(monkeypatch, mode, relative, prefix_steps):
    config = tiny_config()
    config.rtc_training_max_delay = 3
    config.rtc_config = RTCConfig(
        enabled=mode is not ExecutionMode.CHUNK,
        mode="trained" if mode is ExecutionMode.RTC_TRAINED else "guided",
        execution_horizon=4,
    )
    previous = torch.arange(prefix_steps * 3, dtype=torch.float32).reshape(prefix_steps, 3)
    canonical_previous = previous * 2 + torch.tensor([1.0, 2.0, 3.0]) + (10 if relative else 0)
    local_policy = ObservedPolicy(config)
    engine, chunks = local_prediction(
        monkeypatch,
        local_policy,
        relative=relative,
        previous=previous,
        canonical_previous=canonical_previous,
    )
    assert not engine.failed, engine.failure_traceback
    assert len(chunks) == 1
    remote_policy = ObservedPolicy(config)
    remote = runner_for(remote_policy, relative=relative, modes=(mode,)).predict(
        observation(20),
        mode=mode,
        inference_delay=1 if prefix_steps else 0,
        model_continuation=previous,
        canonical_continuation=canonical_previous,
    )
    assert local_policy.context == remote_policy.context == (mode is not ExecutionMode.RTC_GUIDED, False)
    torch.testing.assert_close(chunks[0].canonical_actions, remote.canonical_actions)
    assert chunks[0].execution_steps == remote.execution_steps == (3 if mode is ExecutionMode.CHUNK else 8)
    if mode is ExecutionMode.CHUNK:
        assert local_policy.last_kwargs == remote_policy.last_kwargs == {}
        assert remote.model_actions is None
        return
    torch.testing.assert_close(chunks[0].model_actions, remote.model_actions)
    assert local_policy.last_kwargs["inference_delay"] == remote_policy.last_kwargs["inference_delay"]
    local_prefix = local_policy.last_kwargs["prev_chunk_left_over"]
    remote_prefix = remote_policy.last_kwargs["prev_chunk_left_over"]
    expected = previous[:4] - (5 if relative else 0)
    if len(expected) < 4:
        expected = torch.cat([expected, expected[-1:].expand(4 - len(expected), -1)])
    torch.testing.assert_close(local_prefix, expected)
    torch.testing.assert_close(remote_prefix, expected)


@pytest.mark.parametrize("output", [torch.zeros(1, 7, 3), torch.full((1, 8, 3), float("nan"))])
def test_runner_rejects_invalid_policy_outputs(output):
    policy = ObservedPolicy(tiny_config())
    policy.output = output
    with pytest.raises(ValueError):
        runner_for(policy).predict(observation())


def test_shared_prediction_preserves_model_coordinates_before_inplace_postprocessing():
    policy = ObservedPolicy(tiny_config())
    prediction = predict_chunk(
        policy,
        lambda actions: actions.add_(10),
        {},
        spec=policy.chunk_inference_spec(),
        mode=ExecutionMode.CHUNK,
        canonical_action_dim=3,
        model_action_dim=None,
        device="cpu",
    )
    expected = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    torch.testing.assert_close(prediction.model_actions, expected)
    torch.testing.assert_close(prediction.canonical_actions, expected + 10)
    assert not prediction.model_actions.requires_grad
    assert not prediction.canonical_actions.requires_grad


def test_shared_prediction_keeps_the_policy_prediction_beside_the_actions():
    policy = ObservedPolicy(tiny_config())
    # A `PolicyOutput`: the chunk and a prediction batched like it.
    policy.output = {"action": policy.output, "prediction": {"language": {"subtask": ["pick"]}}}
    prediction = predict_chunk(
        policy,
        lambda actions: actions,
        {},
        spec=policy.chunk_inference_spec(),
        mode=ExecutionMode.CHUNK,
        canonical_action_dim=3,
        model_action_dim=None,
        device="cpu",
    )
    torch.testing.assert_close(prediction.model_actions, torch.arange(9, dtype=torch.float32).reshape(3, 3))
    assert prediction.prediction == {"language": {"subtask": ["pick"]}}


def _rtc_engine_after_one_chunk(monkeypatch, *, accept=True):
    """A local RTC engine that ran one chunk of a policy predicting a subtask with it."""
    config = tiny_config()
    config.rtc_config = RTCConfig(enabled=False)
    policy = ObservedPolicy(config)
    policy.output = {"action": policy.output, "prediction": {"language": {"subtask": ["reach"]}}}
    engine = local_engine_for(policy, queue_threshold=8)
    engine.resume()
    engine._runtime.turnarounds.append(1 / 30)
    engine.notify_observation(
        {"a.pos": 20.0, "b.pos": 20.0, "c.pos": 20.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    real_accept = engine._runtime.accept

    def accept_once(request, chunk, *, task_version):
        engine._shutdown_event.set()
        return real_accept(request, chunk, task_version=task_version) if accept else False

    monkeypatch.setattr(engine._runtime, "accept", accept_once)
    engine._rtc_loop()
    return engine


def test_rtc_engine_hands_a_chunk_prediction_out_once_with_the_next_action(monkeypatch):
    engine = _rtc_engine_after_one_chunk(monkeypatch)

    first = engine.get_action(None)
    assert transition_to_prediction(first) == {"language": {"subtask": ["reach"]}}
    assert isinstance(engine.get_action(None), torch.Tensor)  # the rest of the chunk is bare


def test_rtc_engine_drops_the_prediction_of_a_discarded_chunk(monkeypatch):
    # A chunk the runtime does not accept is never served: its prediction must not be shown either.
    engine = _rtc_engine_after_one_chunk(monkeypatch, accept=False)
    assert engine._chunk_prediction is None


def test_rtc_engine_reset_clears_a_pending_prediction(monkeypatch):
    engine = _rtc_engine_after_one_chunk(monkeypatch)
    engine.reset()
    assert engine._chunk_prediction is None
