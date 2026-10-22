"""Tests for RTC + relative actions integration.

Validates that Real-Time Chunking (RTC) works correctly when the policy uses
relative actions. The key invariant: RTC guidance operates in model space
(normalized relative actions), while the robot receives absolute actions after postprocessing.

Flow under test:
  Preprocessor: raw obs → relative step caches state → normalizer
  Model: generates normalized relative actions (guided by RTC using leftover relative actions)
  Postprocessor: unnormalize → absolute step (relative + cached state) → robot actions
"""

import importlib.util
import sys
from pathlib import Path

import torch

from lerobot.configs.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    RTCAttentionSchedule,
)
from lerobot.processor import TransitionKey, batch_to_transition, create_transition
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _import_rtc_module(module_name: str, filename: str):
    """Import an RTC module directly from its file path, bypassing lerobot.policies.__init__."""
    rtc_dir = Path(__file__).resolve().parents[3] / "src" / "lerobot" / "policies" / "rtc"
    spec = importlib.util.spec_from_file_location(module_name, rtc_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_rtc_cfg_mod = _import_rtc_module("lerobot.policies.rtc.configuration_rtc", "configuration_rtc.py")
RTCConfig = _rtc_cfg_mod.RTCConfig

_action_queue_mod = _import_rtc_module("lerobot.policies.rtc.action_queue", "action_queue.py")
ActionQueue = _action_queue_mod.ActionQueue

_rtc_debug_mod = _import_rtc_module("lerobot.policies.rtc.debug_tracker", "debug_tracker.py")
_rtc_mod = _import_rtc_module("lerobot.policies.rtc.modeling_rtc", "modeling_rtc.py")
RTCProcessor = _rtc_mod.RTCProcessor

_rtc_relative_mod = _import_rtc_module("lerobot.policies.rtc.relative", "relative.py")
reanchor_relative_rtc_prefix = _rtc_relative_mod.reanchor_relative_rtc_prefix

ACTION_DIM = 6
CHUNK_SIZE = 50
EXECUTION_HORIZON = 10


def _make_rtc_config(enabled=True):
    return RTCConfig(
        enabled=enabled,
        execution_horizon=EXECUTION_HORIZON,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )


def _make_relative_pipeline(action_dim=ACTION_DIM, norm_mode=NormalizationMode.MEAN_STD):
    """Build paired relative/absolute processor steps and normalizer/unnormalizer."""
    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    norm_map = {FeatureType.ACTION: norm_mode}

    stats = {
        ACTION: {
            "mean": torch.zeros(action_dim).numpy(),
            "std": torch.ones(action_dim).numpy(),
            "q01": (-2 * torch.ones(action_dim)).numpy(),
            "q99": (2 * torch.ones(action_dim)).numpy(),
            "min": (-3 * torch.ones(action_dim)).numpy(),
            "max": (3 * torch.ones(action_dim)).numpy(),
        }
    }

    relative_step = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    return relative_step, normalizer, unnormalizer, absolute_step


class TestActionQueueRelativeActions:
    """Verify ActionQueue stores model-space (relative) actions for RTC and absolute for robot."""

    def test_left_over_returns_relative_actions(self):
        """get_left_over() should return the original (relative-space) actions."""
        cfg = _make_rtc_config()
        queue = ActionQueue(cfg)

        relative_actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        absolute_actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        queue.merge(relative_actions, absolute_actions, real_delay=0)

        for _ in range(5):
            queue.get()

        leftover = queue.get_left_over()
        torch.testing.assert_close(leftover, relative_actions[5:])

    def test_robot_receives_absolute_actions(self):
        """The robot (via get()) should receive postprocessed absolute actions."""
        cfg = _make_rtc_config()
        queue = ActionQueue(cfg)

        relative_actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        absolute_actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        queue.merge(relative_actions, absolute_actions, real_delay=0)

        first_action = queue.get()
        torch.testing.assert_close(first_action, absolute_actions[0])


class TestRTCDenoiseWithRelativeLeftovers:
    """Verify RTC denoise_step correctly handles relative-space prev_chunk_left_over."""

    def test_first_chunk_no_guidance(self):
        """First chunk (no leftovers) should return v_t without guidance."""
        rtc = RTCProcessor(_make_rtc_config())
        x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        def mock_denoise(x):
            return torch.ones_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=0,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
        )
        torch.testing.assert_close(result, torch.ones_like(x_t))

    def test_relative_leftovers_shape_preserved(self):
        """RTC output should have the same shape as input regardless of leftover shape."""
        rtc = RTCProcessor(_make_rtc_config())
        x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        shorter_leftover = torch.randn(1, 20, ACTION_DIM)

        def mock_denoise(x):
            return torch.zeros_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=shorter_leftover,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
        )
        assert result.shape == x_t.shape

    def test_guidance_steers_toward_previous_relative_actions(self):
        """RTC guidance should push x1_t toward prev_chunk_left_over in relative space."""
        rtc = RTCProcessor(_make_rtc_config())
        x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        prev_relatives = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        def mock_denoise(x):
            return torch.zeros_like(x)

        result_without_guidance = rtc.denoise_step(
            x_t=x_t.clone(),
            prev_chunk_left_over=None,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
        )

        result_with_guidance = rtc.denoise_step(
            x_t=x_t.clone(),
            prev_chunk_left_over=prev_relatives,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
        )

        assert not torch.allclose(result_with_guidance, result_without_guidance, atol=1e-6)


class TestFullPipelineRelativeRTC:
    """End-to-end test of the RTC + relative actions pipeline matching lerobot-rollout flow."""

    def test_preprocessor_caches_state_for_postprocessor(self):
        """Preprocessor's relative step should cache state so postprocessor can convert back."""
        relative_step, normalizer, unnormalizer, absolute_step = _make_relative_pipeline()

        state = torch.randn(1, ACTION_DIM)
        actions = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        batch = {ACTION: actions, OBS_STATE: state}
        transition = batch_to_transition(batch)

        relative_step(transition)
        assert relative_step._last_state is not None
        torch.testing.assert_close(relative_step._last_state, state)

    def test_preprocessor_caches_state_without_actions(self):
        """During inference, preprocessor receives only observations (no actions).
        Relative step should still cache state for the postprocessor."""
        relative_step, _, _, _ = _make_relative_pipeline()

        state = torch.randn(1, ACTION_DIM)
        batch = {OBS_STATE: state}
        transition = batch_to_transition(batch)

        relative_step(transition)
        assert relative_step._last_state is not None
        torch.testing.assert_close(relative_step._last_state, state)

    def test_roundtrip_with_identity_normalization(self):
        """Actions → relative → normalize → [model] → unnormalize → absolute should recover originals.

        Using mean=0, std=1 normalization (identity).
        """
        relative_step, normalizer, unnormalizer, absolute_step = _make_relative_pipeline()

        state = torch.randn(1, ACTION_DIM)
        actions = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        batch = {ACTION: actions.clone(), OBS_STATE: state}
        transition = batch_to_transition(batch)

        t1 = relative_step(transition)
        t2 = normalizer(t1)

        model_output = t2[TransitionKey.ACTION].clone()

        model_transition = {TransitionKey.ACTION: model_output}
        t3 = unnormalizer(model_transition)
        t4 = absolute_step(t3)

        recovered = t4[TransitionKey.ACTION]
        torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)

    def test_eval_loop_simulation(self):
        """Simulate the lerobot-rollout loop with relative actions.

        Iteration 1: No leftovers → model generates relative actions → store for RTC
        Iteration 2: Use leftovers as RTC guidance → model generates new relative actions
        Both iterations: postprocessor converts relative actions to absolute for robot
        """
        relative_step, normalizer, unnormalizer, absolute_step = _make_relative_pipeline()
        rtc = RTCProcessor(_make_rtc_config())
        queue = ActionQueue(_make_rtc_config())

        def mock_model(prev_chunk_left_over, inference_delay, state):
            """Simulate model generating relative actions with RTC."""
            x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

            def denoise(x):
                return -0.1 * x

            guided_v = rtc.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=0.5,
                original_denoise_step_partial=denoise,
            )
            return x_t - 0.5 * guided_v

        # --- Iteration 1: first chunk, no leftovers ---
        state_1 = torch.randn(1, ACTION_DIM)
        obs_batch_1 = {OBS_STATE: state_1}
        relative_step(batch_to_transition(obs_batch_1))

        model_relatives_1 = mock_model(prev_chunk_left_over=None, inference_delay=0, state=state_1)
        original_actions_1 = model_relatives_1.squeeze(0)

        model_transition_1 = {TransitionKey.ACTION: model_relatives_1}
        postprocessed_1 = absolute_step(unnormalizer(model_transition_1))[TransitionKey.ACTION].squeeze(0)

        queue.merge(original_actions_1, postprocessed_1, real_delay=0)

        # Consume some actions (simulate robot executing)
        for _ in range(5):
            action = queue.get()
            assert action is not None

        # --- Iteration 2: use leftovers for RTC ---
        prev_actions = queue.get_left_over()
        assert prev_actions is not None
        assert prev_actions.shape[0] == CHUNK_SIZE - 5

        state_2 = state_1 + 0.01 * torch.randn(1, ACTION_DIM)
        obs_batch_2 = {OBS_STATE: state_2}
        relative_step(batch_to_transition(obs_batch_2))

        model_relatives_2 = mock_model(
            prev_chunk_left_over=prev_actions.unsqueeze(0), inference_delay=3, state=state_2
        )
        original_actions_2 = model_relatives_2.squeeze(0)

        model_transition_2 = {TransitionKey.ACTION: model_relatives_2}
        postprocessed_2 = absolute_step(unnormalizer(model_transition_2))[TransitionKey.ACTION].squeeze(0)

        queue.merge(original_actions_2, postprocessed_2, real_delay=3)

        # Postprocessed actions should be in absolute space
        action = queue.get()
        assert action is not None
        assert action.shape == (ACTION_DIM,)

        # Verify leftovers are in relative space (original_queue stores relative actions)
        leftover_relatives = queue.get_left_over()
        assert leftover_relatives is not None
        assert leftover_relatives.shape[1] == ACTION_DIM

    def test_postprocessor_uses_correct_state_per_iteration(self):
        """Each iteration's postprocessor should use the state from that iteration's preprocessor,
        not a stale state from a previous iteration."""
        relative_step, _, unnormalizer, absolute_step = _make_relative_pipeline()

        state_1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        state_2 = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
        relatives = torch.zeros(1, 5, ACTION_DIM)

        # Iteration 1: cache state_1
        relative_step(batch_to_transition({OBS_STATE: state_1}))
        result_1 = absolute_step(unnormalizer({TransitionKey.ACTION: relatives.clone()}))[
            TransitionKey.ACTION
        ]
        # relative=0 + state_1 should give state_1
        for t in range(5):
            torch.testing.assert_close(result_1[0, t], state_1[0], atol=1e-5, rtol=1e-5)

        # Iteration 2: cache state_2
        relative_step(batch_to_transition({OBS_STATE: state_2}))
        result_2 = absolute_step(unnormalizer({TransitionKey.ACTION: relatives.clone()}))[
            TransitionKey.ACTION
        ]
        for t in range(5):
            torch.testing.assert_close(result_2[0, t], state_2[0], atol=1e-5, rtol=1e-5)


class TestStateRebasingApproximation:
    """Verify that the approximation from not rebasing leftover relative actions is small
    when state changes between inference calls are small (real-time control regime)."""

    def test_small_state_change_produces_small_error(self):
        """With small state changes (typical in real-time control),
        using stale relative actions for RTC guidance introduces negligible error."""
        state_old = torch.randn(1, ACTION_DIM)
        state_new = state_old + 0.01 * torch.randn(1, ACTION_DIM)

        actions_absolute = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        mask = [True] * ACTION_DIM

        relatives_old = to_relative_actions(actions_absolute, state_old, mask)
        relatives_new = to_relative_actions(actions_absolute, state_new, mask)

        error = (relatives_old - relatives_new).abs().mean()
        state_change = (state_old - state_new).abs().mean()

        # Error should be proportional to state change
        assert error < 0.1, (
            f"Relative-action error {error:.4f} should be small for small state change {state_change:.4f}"
        )

    def test_large_state_change_produces_proportional_error(self):
        """With large state changes, stale relative actions diverge more (but RTC guidance decays)."""
        state_old = torch.randn(1, ACTION_DIM)
        state_new = state_old + 10.0 * torch.randn(1, ACTION_DIM)

        actions_absolute = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        mask = [True] * ACTION_DIM

        relatives_old = to_relative_actions(actions_absolute, state_old, mask)
        relatives_new = to_relative_actions(actions_absolute, state_new, mask)

        error = (relatives_old - relatives_new).abs().mean()
        state_change = (state_old - state_new).abs().mean()

        # Error should be roughly equal to state change
        torch.testing.assert_close(
            error.clone().detach(), state_change.clone().detach(), atol=1e-5, rtol=1e-5
        )

    def test_excluded_joints_not_affected_by_state_change(self):
        """Joints excluded from relative conversion should not contribute rebasing error."""
        state_old = torch.randn(1, ACTION_DIM)
        state_new = state_old.clone()
        state_new[0, -1] = state_old[0, -1] + 100.0

        actions = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        mask = [True] * (ACTION_DIM - 1) + [False]

        relatives_old = to_relative_actions(actions, state_old, mask)
        relatives_new = to_relative_actions(actions, state_new, mask)

        # Last dim (excluded) should have zero error
        error_excluded = (relatives_old[..., -1] - relatives_new[..., -1]).abs().max()
        assert error_excluded < 1e-6, f"Excluded joint should have zero error, got {error_excluded}"


class TestRTCReanchoringWithStateNormalizer:
    """RTC re-anchoring under non-identity OBS_STATE normalization."""

    @staticmethod
    def _build_normalizer_with_state_stats():
        """Build a relative-action preprocessor with non-trivial OBS_STATE stats."""
        features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,)),
        }
        norm_map = {
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
        }
        stats = {
            ACTION: {
                "mean": torch.zeros(ACTION_DIM).numpy(),
                "std": (0.5 * torch.ones(ACTION_DIM)).numpy(),
            },
            OBS_STATE: {
                "mean": (5.0 * torch.ones(ACTION_DIM)).numpy(),
                "std": (2.0 * torch.ones(ACTION_DIM)).numpy(),
            },
        }
        relative_step = RelativeActionsProcessorStep(enabled=True)
        normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        return relative_step, normalizer

    def test_reanchor_with_raw_state_matches_normalize_of_absolute_minus_state(self):
        """Reanchoring with the raw cached state yields ``normalize(prev_actions_absolute - raw_state)``."""
        relative_step, normalizer = self._build_normalizer_with_state_stats()

        raw_state = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
        relative_step(batch_to_transition({OBS_STATE: raw_state.clone()}))

        prev_actions_absolute = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * 5)

        result = reanchor_relative_rtc_prefix(
            prev_actions_absolute=prev_actions_absolute,
            current_state=relative_step.get_cached_state(),
            relative_step=relative_step,
            normalizer_step=normalizer,
            policy_device="cpu",
        )

        expected_relative = to_relative_actions(prev_actions_absolute, raw_state, [True] * ACTION_DIM)
        expected = normalizer(create_transition(action=expected_relative))[TransitionKey.ACTION]
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_reanchor_with_normalized_state_produces_wrong_result(self):
        """Reanchoring with raw vs. normalized state produces meaningfully different outputs."""
        relative_step, normalizer = self._build_normalizer_with_state_stats()

        raw_state = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
        relative_step(batch_to_transition({OBS_STATE: raw_state.clone()}))

        normalized_obs = normalizer(batch_to_transition({OBS_STATE: raw_state.clone()}))
        normalized_state = normalized_obs[TransitionKey.OBSERVATION][OBS_STATE]
        assert not torch.allclose(normalized_state, raw_state)

        prev_actions_absolute = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * 5)

        result_raw = reanchor_relative_rtc_prefix(
            prev_actions_absolute=prev_actions_absolute,
            current_state=raw_state,
            relative_step=relative_step,
            normalizer_step=normalizer,
            policy_device="cpu",
        )
        result_normalized = reanchor_relative_rtc_prefix(
            prev_actions_absolute=prev_actions_absolute,
            current_state=normalized_state,
            relative_step=relative_step,
            normalizer_step=normalizer,
            policy_device="cpu",
        )

        max_abs_diff = (result_raw - result_normalized).abs().max()
        assert max_abs_diff > 0.5, (
            f"Raw and normalized state produced near-identical outputs (max diff {max_abs_diff:.4f}); "
            "OBS_STATE stats are too close to identity to be sensitive."
        )

    def test_engine_pipeline_cached_state_is_raw_after_full_preprocess(self):
        """``get_cached_state()`` returns raw OBS_STATE after the full preprocessor pipeline runs."""
        relative_step, normalizer = self._build_normalizer_with_state_stats()

        raw_state = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])

        transition = batch_to_transition({OBS_STATE: raw_state.clone()})
        transition = relative_step(transition)
        preprocessed = normalizer(transition)

        cached = relative_step.get_cached_state()
        torch.testing.assert_close(cached, raw_state, atol=1e-6, rtol=1e-6)

        post_normalize_state = preprocessed[TransitionKey.OBSERVATION][OBS_STATE]
        assert not torch.allclose(cached, post_normalize_state, atol=1e-3)


def _detect_relative_actions(preprocessor) -> bool:
    """Mirror of the helper in lerobot-rollout for testing without importing it."""
    return any(isinstance(step, RelativeActionsProcessorStep) and step.enabled for step in preprocessor.steps)


class TestDetectRelativeActions:
    """Test the _detect_relative_actions helper logic used by lerobot-rollout."""

    def test_detects_enabled_relative_step(self):
        class FakePipeline:
            steps = [RelativeActionsProcessorStep(enabled=True)]

        assert _detect_relative_actions(FakePipeline()) is True

    def test_ignores_disabled_relative_step(self):
        class FakePipeline:
            steps = [RelativeActionsProcessorStep(enabled=False)]

        assert _detect_relative_actions(FakePipeline()) is False

    def test_returns_false_when_no_relative_step(self):
        class FakePipeline:
            steps = []

        assert _detect_relative_actions(FakePipeline()) is False


class TestNonRelativePolicy:
    """Verify the same pipeline works when relative actions are disabled (standard absolute policy)."""

    def test_disabled_relative_step_is_noop(self):
        relative_step = RelativeActionsProcessorStep(enabled=False)
        absolute_step = AbsoluteActionsProcessorStep(enabled=False, relative_step=relative_step)

        state = torch.randn(1, ACTION_DIM)
        actions = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        transition = batch_to_transition({ACTION: actions.clone(), OBS_STATE: state})
        t1 = relative_step(transition)
        torch.testing.assert_close(t1[TransitionKey.ACTION], actions)

        t2 = absolute_step({TransitionKey.ACTION: actions.clone()})
        torch.testing.assert_close(t2[TransitionKey.ACTION], actions)

    def test_eval_loop_without_relative_actions(self):
        """Full eval loop simulation with relative actions disabled: original and processed actions are identical."""
        features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
        stats = {
            ACTION: {
                "mean": torch.zeros(ACTION_DIM).numpy(),
                "std": torch.ones(ACTION_DIM).numpy(),
            }
        }

        relative_step = RelativeActionsProcessorStep(enabled=False)
        unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        absolute_step = AbsoluteActionsProcessorStep(enabled=False, relative_step=relative_step)

        rtc = RTCProcessor(_make_rtc_config())
        queue = ActionQueue(_make_rtc_config())

        state = torch.randn(1, ACTION_DIM)
        relative_step(batch_to_transition({OBS_STATE: state}))

        model_output = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        post = absolute_step(unnormalizer({TransitionKey.ACTION: model_output.clone()}))[
            TransitionKey.ACTION
        ].squeeze(0)
        original = model_output.squeeze(0)

        # With identity norm and no relative-action transform, original and postprocessed should match
        torch.testing.assert_close(original, post, atol=1e-5, rtol=1e-5)

        queue.merge(original, post, real_delay=0)

        for _ in range(5):
            queue.get()

        prev_actions = queue.get_left_over()
        assert prev_actions is not None

        # RTC guidance works the same way (absolute space)
        x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev_actions.unsqueeze(0),
            inference_delay=3,
            time=0.5,
            original_denoise_step_partial=lambda x: torch.zeros_like(x),
        )
        assert result.shape == x_t.shape

    def test_detect_relative_returns_false_when_disabled(self):
        class FakePipeline:
            steps = [RelativeActionsProcessorStep(enabled=False)]

        assert not _detect_relative_actions(FakePipeline())

    def test_detect_relative_returns_false_when_absent(self):
        class FakePipeline:
            steps = []

        assert not _detect_relative_actions(FakePipeline())


class TestMultiChunkConsistency:
    """Test multiple RTC iterations with relative actions maintain consistency."""

    def test_three_iteration_pipeline(self):
        """Simulate three consecutive RTC iterations and verify queue state consistency."""
        relative_step, normalizer, unnormalizer, absolute_step = _make_relative_pipeline()
        queue = ActionQueue(_make_rtc_config())

        states = [torch.randn(1, ACTION_DIM) + i * 0.01 for i in range(3)]

        for i in range(3):
            queue.get_left_over()

            relative_step(batch_to_transition({OBS_STATE: states[i]}))

            model_output = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
            post_transition = absolute_step(unnormalizer({TransitionKey.ACTION: model_output.clone()}))
            postprocessed = post_transition[TransitionKey.ACTION].squeeze(0)
            original = model_output.squeeze(0)

            delay = min(i * 2, CHUNK_SIZE - 1)
            queue.merge(original, postprocessed, real_delay=delay)

            for _ in range(5):
                action = queue.get()
                assert action is not None
                assert action.shape == (ACTION_DIM,)

        # After 3 iterations, queue should still be in valid state
        assert queue.qsize() > 0
        leftover = queue.get_left_over()
        assert leftover is not None
        assert leftover.ndim == 2
        assert leftover.shape[1] == ACTION_DIM

    def test_leftover_and_processed_differ_when_relative_enabled(self):
        """With relative actions enabled, original leftovers (relative) must differ from processed (absolute)."""
        relative_step, _, unnormalizer, absolute_step = _make_relative_pipeline()
        queue = ActionQueue(_make_rtc_config())

        state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        relative_step(batch_to_transition({OBS_STATE: state}))

        model_relatives = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        post = absolute_step(unnormalizer({TransitionKey.ACTION: model_relatives.clone()}))[
            TransitionKey.ACTION
        ].squeeze(0)
        original = model_relatives.squeeze(0)

        queue.merge(original, post, real_delay=0)

        relative_leftover = queue.get_left_over()

        # Leftovers (relative) must differ from the postprocessed absolute actions
        assert not torch.allclose(relative_leftover, post, atol=1e-3)
        state_expanded = state.squeeze(0).unsqueeze(0).expand_as(relative_leftover)
        torch.testing.assert_close(post, relative_leftover + state_expanded, atol=1e-5, rtol=1e-5)

    def test_rtc_guidance_uses_relative_space(self):
        """Verify that RTC denoise_step receives relative-space leftovers, not absolute."""
        relative_step, _, unnormalizer, absolute_step = _make_relative_pipeline()
        rtc = RTCProcessor(_make_rtc_config())
        queue = ActionQueue(_make_rtc_config())

        state = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
        relative_step(batch_to_transition({OBS_STATE: state}))

        model_relatives = 0.1 * torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        post = absolute_step(unnormalizer({TransitionKey.ACTION: model_relatives.clone()}))[
            TransitionKey.ACTION
        ].squeeze(0)
        original = model_relatives.squeeze(0)

        queue.merge(original, post, real_delay=0)

        for _ in range(5):
            queue.get()

        prev_left_over = queue.get_left_over()

        # prev_left_over should be small relative offsets (around 0.1 * randn), not large absolute values
        assert prev_left_over.abs().mean() < 5.0, (
            f"Leftover should be small relative offsets, got mean abs {prev_left_over.abs().mean():.2f}"
        )

        x_t = torch.randn(1, CHUNK_SIZE, ACTION_DIM)

        def denoise(x):
            return torch.zeros_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev_left_over.unsqueeze(0),
            inference_delay=3,
            time=0.5,
            original_denoise_step_partial=denoise,
        )

        assert result.shape == x_t.shape
