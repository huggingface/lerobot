"""Local RTC and the remote runner share post-preprocessing action semantics."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="local rollout imports the dataset integration")

from lerobot.inference.contracts import ExecutionMode
from lerobot.inference.prediction import predict_chunk
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE
from tests.inference.test_policy_runner import (
    ConformingPolicy,
    observation,
    processors,
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
    names = ("a.pos", "b.pos", "c.pos")
    env_names = ("env_a", "env_b", "env_c")
    engine = RTCInferenceEngine(
        policy,
        *processors(policy.config, relative=relative),
        robot_wrapper=SimpleNamespace(robot_type="test", action_features=dict.fromkeys(names, float)),
        rtc_config=policy.config.rtc_config,
        dataset_features={
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": names},
            OBS_ENV_STATE: {"dtype": "float32", "shape": (3,), "names": env_names},
        },
        task="pick up the cube",
        fps=30,
        device="cpu",
        rtc_queue_threshold=8,
    )
    engine.resume()
    if previous is not None and len(previous):
        engine.action_queue.merge(previous, canonical_previous, real_delay=0)
    engine._runtime.turnarounds.append(1 / 30)
    engine.notify_observation({**dict.fromkeys(names, 20.0), **dict.fromkeys(env_names, 0.0)})
    result = []

    def capture(request, chunk, *, task_version):
        result.append(chunk)
        engine._shutdown_event.set()
        return True

    monkeypatch.setattr(engine._runtime, "accept", capture)
    engine._rtc_loop()
    return engine, result


@pytest.mark.parametrize("mode", list(ExecutionMode))
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("prefix_steps", [0, 2, 4, 6])
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
    if not prefix_steps:
        assert local_prefix is remote_prefix is None
        return
    expected = previous[:4] - (5 if relative else 0)
    if len(expected) < 4:
        expected = torch.cat([expected, expected[-1:].expand(4 - len(expected), -1)])
    torch.testing.assert_close(local_prefix, expected)
    torch.testing.assert_close(remote_prefix, expected)


@pytest.mark.parametrize(
    "output,error",
    [
        (None, "declared batch and horizon"),
        (torch.zeros(1, 7, 3), "declared batch and horizon"),
        (torch.zeros(1, 8, 3, dtype=torch.int64), "finite floating"),
        (torch.full((1, 8, 3), float("nan")), "finite floating"),
    ],
)
def test_local_and_remote_reject_invalid_policy_outputs(monkeypatch, output, error):
    config = tiny_config()
    config.rtc_config = RTCConfig(enabled=False)
    local_policy = ObservedPolicy(config)
    local_policy.output = output
    engine, chunks = local_prediction(monkeypatch, local_policy)
    assert not chunks
    assert engine.failed
    assert error in engine.failure_traceback
    remote_policy = ObservedPolicy(config)
    remote_policy.output = output
    with pytest.raises(ValueError, match=error):
        runner_for(remote_policy).predict(observation())


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
