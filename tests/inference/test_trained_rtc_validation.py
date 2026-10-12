"""Local and remote trained RTC apply the same conservative prefix bounds."""

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.inference import ExecutionMode, RTCInferenceConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.context import _validate_trained_rtc_rollout_config
from tests.inference.test_policy_runner import ConformingPolicy, observation, runner_for, tiny_config


@pytest.mark.parametrize(
    "delay,horizon,allowed",
    [(3, 2, False), (3, 3, True), (3, 5, True), (3, 6, False), (4, 4, True), (5, 5, False)],
)
def test_trained_prefix_bounds_match_across_local_and_remote_admission(delay, horizon, allowed):
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_training_max_delay = delay
    policy.config.rtc_config = RTCConfig(mode="trained", execution_horizon=horizon)
    local_config = RTCInferenceConfig(rtc=policy.config.rtc_config, queue_threshold=delay)
    if not allowed:
        with pytest.raises(ValueError, match="execution_horizon") as local_error:
            _validate_trained_rtc_rollout_config(policy.config, local_config)
        with pytest.raises(ValueError, match="execution_horizon") as remote_error:
            runner_for(policy, modes=(ExecutionMode.RTC_TRAINED,))
        assert str(local_error.value) == str(remote_error.value)
        return

    _validate_trained_rtc_rollout_config(policy.config, local_config)
    runner = runner_for(policy, modes=(ExecutionMode.RTC_TRAINED,))
    prefix = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    chunk = runner.predict(
        observation(),
        mode=ExecutionMode.RTC_TRAINED,
        inference_delay=delay,
        model_continuation=prefix,
    )
    assert runner.capabilities.rtc_horizon == horizon
    torch.testing.assert_close(policy.last_kwargs["prev_chunk_left_over"], prefix[:horizon])
    assert policy.last_kwargs["inference_delay"] == delay
    assert chunk.execution_steps == 8, "prefix capacity does not shorten the returned prediction"


def test_local_queue_capacity_remains_separate_from_server_prefix_configuration():
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_training_max_delay = 3
    policy.config.rtc_config = RTCConfig(mode="trained", execution_horizon=3)
    with pytest.raises(ValueError, match="queue_threshold"):
        _validate_trained_rtc_rollout_config(
            policy.config, RTCInferenceConfig(rtc=policy.config.rtc_config, queue_threshold=2)
        )
    # The server has no local playback queue; the remote client owns its refill timing.
    runner = runner_for(policy, modes=(ExecutionMode.RTC_TRAINED,))
    assert runner.capabilities.rtc_horizon == 3


@pytest.mark.parametrize("mode", [ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED])
def test_trained_upper_bound_does_not_restrict_other_execution_modes(mode):
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_training_max_delay = 3
    policy.config.rtc_config = RTCConfig(enabled=mode != ExecutionMode.CHUNK, execution_horizon=8)
    local_config = RTCInferenceConfig(rtc=policy.config.rtc_config)
    _validate_trained_rtc_rollout_config(policy.config, local_config)
    assert runner_for(policy, modes=(mode,)).capabilities.modes == (mode,)
