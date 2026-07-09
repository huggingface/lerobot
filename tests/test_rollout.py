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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# Hoisted rather than re-imported inside each test: the module-scoped `importorskip`
# above already guards everything below it.
from lerobot.utils.cycle_timer import CycleTimer  # noqa: E402

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


def test_rollout_config_passes_policy_pretrained_revision(monkeypatch):
    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    captured = {}

    def fake_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        captured["pretrained_name_or_path"] = pretrained_name_or_path
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", pretrained_revision=kwargs["revision"])

    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=yaml-sha"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-rollout", "--policy.path=user/policy", "--policy.pretrained_revision=cli-sha"],
    )
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_from_pretrained))

    cfg = RolloutConfig(robot=MockRobotConfig())

    assert captured["pretrained_name_or_path"] == "user/policy"
    assert captured["revision"] == "cli-sha"
    assert captured["cli_overrides"] == [
        "--pretrained_revision=yaml-sha",
        "--pretrained_revision=cli-sha",
    ]
    assert cfg.policy.pretrained_path == "user/policy"
    assert cfg.policy.pretrained_revision == "cli-sha"


@pytest.mark.parametrize("multiplier", [0, -1])
def test_rollout_config_rejects_a_multiplier_below_one(multiplier):
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    # Validated in __post_init__ ahead of everything else, so the run fails at
    # parse time rather than after the hardware is already connected.
    with pytest.raises(ValueError, match="interpolation_multiplier must be >= 1"):
        RolloutConfig(robot=MockRobotConfig(), interpolation_multiplier=multiplier)


def test_load_pretrained_policy_passes_revision(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=False,
        pretrained_path="user/policy",
        pretrained_revision="policy-sha",
    )
    policy_class = MagicMock()
    loaded_policy = MagicMock()
    policy_class.from_pretrained.return_value = loaded_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is loaded_policy
    policy_class.from_pretrained.assert_called_once_with(
        "user/policy",
        config=policy_config,
        revision="policy-sha",
    )


def test_load_pretrained_peft_policy_keeps_adapter_and_base_revisions_separate(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=True,
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
    )
    policy_class = MagicMock()
    base_policy = MagicMock()
    policy_class.from_pretrained.return_value = base_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    peft_config = SimpleNamespace(
        base_model_name_or_path="user/base-policy",
        revision="base-sha",
    )
    peft_config_from_pretrained = MagicMock(return_value=peft_config)
    adapted_policy = MagicMock()
    peft_model_from_pretrained = MagicMock(return_value=adapted_policy)
    require_package = MagicMock()
    monkeypatch.setattr(rollout_context, "require_package", require_package)
    monkeypatch.setattr(
        rollout_context,
        "PeftConfig",
        SimpleNamespace(from_pretrained=peft_config_from_pretrained),
        raising=False,
    )
    monkeypatch.setattr(
        rollout_context,
        "PeftModel",
        SimpleNamespace(from_pretrained=peft_model_from_pretrained),
        raising=False,
    )

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is adapted_policy
    require_package.assert_called_once_with("peft", extra="peft")
    peft_config_from_pretrained.assert_called_once_with("user/adapter", revision="adapter-sha")
    policy_class.from_pretrained.assert_called_once_with(
        pretrained_name_or_path="user/base-policy",
        config=policy_config,
        revision="base-sha",
    )
    peft_model_from_pretrained.assert_called_once_with(
        base_policy,
        "user/adapter",
        config=peft_config,
        revision="adapter-sha",
    )


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_align_state_feature_order_matches_checkpoint_and_preserves_cameras(caplog):
    from lerobot.rollout.context import _align_state_feature_order
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    features = {
        "wrist_camera": (480, 640, 3),
        "joint_b.pos": float,
        "joint_a.pos": float,
    }

    aligned = _align_state_feature_order(features, ["joint_a.pos", "joint_b.pos"])

    assert list(aligned) == ["joint_a.pos", "joint_b.pos", "wrist_camera"]
    assert aligned["wrist_camera"] == (480, 640, 3)
    assert "reordering state" in caplog.text

    dataset_features = hw_to_dataset_features(aligned, "observation")
    frame = build_dataset_frame(
        dataset_features,
        {"joint_a.pos": 1.0, "joint_b.pos": 2.0, "wrist_camera": object()},
        "observation",
    )
    assert frame["observation.state"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("policy_action_names", "feature_names"),
    [
        (None, ["joint_b.pos", "joint_a.pos", "wrist_camera"]),
        (["joint_b.pos", "joint_a.pos"], ["joint_b.pos", "joint_a.pos", "wrist_camera"]),
        (["joint_a.pos"], ["joint_b.pos", "joint_a.pos", "wrist_camera"]),
        (["joint_a.pos", "gripper.pos"], ["joint_b.pos", "joint_a.pos", "wrist_camera"]),
    ],
)
def test_align_state_feature_order_is_noop_without_an_exact_name_match(policy_action_names, feature_names):
    from lerobot.rollout.context import _align_state_feature_order

    features = {
        "joint_b.pos": float,
        "joint_a.pos": float,
        "wrist_camera": (480, 640, 3),
    }

    aligned = _align_state_feature_order(features, policy_action_names)

    assert aligned is features
    assert list(aligned) == feature_names


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}


# ---------------------------------------------------------------------------
# Cadence pacing through the strategies
#
# The timer's own contract lives in tests/utils/test_cycle_timer.py; these check
# how the strategies drive it.  `clock` comes from tests/fixtures/cadence.py.
# ---------------------------------------------------------------------------

_TIMER_LOGGER = "lerobot.utils.cycle_timer"


def _timer_warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _info_messages(caplog):
    """INFO messages from the timer only (i.e. the cadence reports)."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.name == _TIMER_LOGGER]


def test_handle_warmup_paces_then_flushes_and_exempts_the_reprimed_group(caplog, clock):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Warm-up ticks are paced through the timer, and the flush resets the
    # interpolator — which then re-primes over two consecutive inference ticks,
    # exactly like loop start-up.  The group spanning them legitimately exceeds
    # budget, so `timer.restart()` must re-arm the start-up exemption; without it
    # every torch.compile run would warn once right after warm-up.
    fps, multiplier = 20.0, 2  # 25 ms slots, 50 ms cycle budget
    engine = MagicMock(ready=False)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._interpolator = ActionInterpolator(multiplier=multiplier)
    timer = CycleTimer(fps, multiplier)

    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        # Two slow warm-up ticks close one over-budget group (80 ms > 50 ms).
        for _ in range(2):
            timer.tick(new_cycle=True)
            clock.advance(0.04)
            assert strategy._handle_warmup(True, timer) is True

        engine.ready = True
        timer.tick(new_cycle=True)
        assert strategy._handle_warmup(True, timer) is False

        # Stale warm-up state is discarded and the engine is put back to work.
        engine.reset.assert_called_once()
        engine.resume.assert_called_once()
        assert strategy._warmup_flushed
        assert strategy._interpolator.needs_new_action()

        # The two re-priming inference ticks that follow are also over budget.
        clock.advance(0.04)
        timer.wait()
        timer.tick(new_cycle=True)
        clock.advance(0.04)
        timer.wait()

    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]

    # A subsequent over-budget group is still reported — the exemption is armed
    # once per restart, not switched off.
    with caplog.at_level(logging.WARNING, logger=_TIMER_LOGGER):
        for _ in range(2):
            timer.tick(new_cycle=True)
            clock.advance(0.04)
            timer.wait()
    assert len(_timer_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# Recording cadence with interpolation
# ---------------------------------------------------------------------------


_LOOP_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["m.pos"]},
    "action": {"dtype": "float32", "shape": (1,), "names": ["m.pos"]},
}


def _make_loop_ctx(fps: float, multiplier: int, num_ticks: int, on_tick=None):
    """Build a mocked RolloutContext-alike for driving a strategy loop.

    The robot's ``get_observation`` sets the shutdown event on the
    *num_ticks*-th call, so the loop runs exactly *num_ticks* full ticks, and
    invokes *on_tick* with the 1-based tick index so tests can inject events
    (keypresses, phase changes) at a chosen tick.

    The engine yields a strictly increasing action per policy call, so an
    interpolated command (a midpoint, ``x.5``) is distinguishable from the
    policy's own end-point action (a whole number) in the recorded frames.
    """
    shutdown_event = threading.Event()
    cfg = SimpleNamespace(
        fps=fps,
        interpolation_multiplier=multiplier,
        duration=0.0,
        dataset=SimpleNamespace(single_task="task"),
        task="task",
        play_sounds=False,
        display_data=False,
        display_mode="rerun",
        display_compressed_images=False,
        display_ip=None,
        display_port=None,
        use_torch_compile=False,
    )
    robot = MagicMock()
    calls = {"n": 0}

    def _get_observation():
        calls["n"] += 1
        if on_tick is not None:
            on_tick(calls["n"])
        if calls["n"] >= num_ticks:
            shutdown_event.set()
        return {"m.pos": float(calls["n"])}

    robot.get_observation.side_effect = _get_observation
    engine = MagicMock()
    engine.dispatched_task = "task"
    actions = {"n": 0}

    def _get_action(_obs_frame):
        actions["n"] += 1
        return torch.tensor([float(actions["n"])])

    engine.get_action.side_effect = _get_action
    dataset = MagicMock()
    dataset.num_episodes = 0
    # A real (non-existent) path so VideoEncodingManager's image-dir cleanup no-ops.
    dataset.root = Path("/nonexistent-lerobot-rollout-test")
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event, cadence_report=None),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=MagicMock(), initial_position=None),
        processors=SimpleNamespace(
            robot_observation_processor=MagicMock(side_effect=lambda obs: obs),
            robot_action_processor=lambda pair: pair[0],
            teleop_action_processor=lambda pair: pair[0],
        ),
        policy=SimpleNamespace(inference=engine),
        data=SimpleNamespace(dataset=dataset, dataset_features=_LOOP_FEATURES, ordered_action_keys=["m.pos"]),
    )
    return ctx, dataset


def _recorded_actions(dataset) -> list[float]:
    """Action values of the frames handed to ``dataset.add_frame``."""
    return [call.args[0]["action"][0] for call in dataset.add_frame.call_args_list]


def _make_sentry(ctx, multiplier: int):
    from lerobot.rollout import SentryStrategyConfig
    from lerobot.rollout.strategies import SentryStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    strategy = SentryStrategy(SentryStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=multiplier)
    strategy._episode_duration_s = 1e9
    return strategy


def test_sentry_records_once_per_interpolation_cycle():
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)

    _make_sentry(ctx, 2).run(ctx)

    # 8 ticks at fps × 2 → 4 recorded frames at the base fps cadence, each
    # carrying the policy's own action rather than an interpolated midpoint.
    assert dataset.add_frame.call_count == 4
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]


def test_sentry_runs_inference_once_per_cycle_and_reuses_processed_observation():
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)

    _make_sentry(ctx, 2).run(ctx)

    # The interpolator's startup buffer holds one action, so a fresh action is
    # pulled on ticks 1, 2, 4, 6 and 8 — five policy ticks across eight ticks,
    # never once per tick.
    engine = ctx.policy.inference
    assert engine.get_action.call_count == 5
    assert engine.notify_observation.call_count == 5
    # The observation processor is the expensive step gated to policy ticks;
    # interpolated ticks reuse the cached result.
    assert ctx.processors.robot_observation_processor.call_count == 5
    # ...while the raw observation is still refreshed every tick for the
    # action processors.
    assert ctx.hardware.robot_wrapper.get_observation.call_count == 8


def test_sentry_records_every_tick_at_default_multiplier():
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=1, num_ticks=4)

    _make_sentry(ctx, 1).run(ctx)

    # multiplier=1 (the default): every tick is a full cycle and records.
    assert dataset.add_frame.call_count == 4
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]


def test_highlight_buffers_once_per_interpolation_cycle():
    from lerobot.rollout import HighlightStrategyConfig
    from lerobot.rollout.ring_buffer import RolloutRingBuffer
    from lerobot.rollout.strategies import HighlightStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    strategy = HighlightStrategy(HighlightStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._ring = RolloutRingBuffer(max_seconds=10.0, max_memory_mb=64, fps=200.0)

    strategy.run(ctx)

    # Without a save request every recorded frame lands in the ring buffer,
    # once per interpolation cycle.
    assert len(strategy._ring) == 4
    assert dataset.add_frame.call_count == 0


def test_highlight_save_toggle_starts_and_ends_live_recording(caplog):
    from lerobot.rollout import HighlightStrategyConfig
    from lerobot.rollout.ring_buffer import RolloutRingBuffer
    from lerobot.rollout.strategies import HighlightStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    strategy = HighlightStrategy(HighlightStrategyConfig())

    # First keypress lands on tick 2, so the record tick that follows it
    # (tick 3) flushes the ring and starts live recording; the second lands on
    # tick 6, so tick 7 closes the episode.
    def on_tick(n):
        if n in (2, 6):
            strategy._save_requested.set()

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8, on_tick=on_tick)
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._ring = RolloutRingBuffer(max_seconds=10.0, max_memory_mb=64, fps=200.0)

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        strategy.run(ctx)

    # Record ticks are 1, 3, 5, 7.  Tick 1 buffers into the ring; tick 3 drains
    # that buffered frame into the dataset — the whole point of the strategy —
    # and starts live recording, so its own frame follows; tick 5 records live;
    # tick 7 closes the episode with its frame added exactly once.
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]
    assert dataset.save_episode.call_count == 1
    assert not strategy._recording_live.is_set()
    assert len(strategy._ring) == 0
    # Closing the episode also reports its cadence, and the run block follows.
    messages = _info_messages(caplog)
    assert len(messages) == 3
    assert messages[0].startswith("Cadence (episode 0):")
    assert messages[1].startswith("Cadence (final episode):")
    assert messages[2].startswith("Cadence summary — whole run, 2 episodes")


def test_episodic_records_once_per_interpolation_cycle():
    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.strategies import EpisodicStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    strategy = EpisodicStrategy(EpisodicStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)

    strategy._policy_loop(
        ctx=ctx,
        robot=ctx.hardware.robot_wrapper,
        events={"exit_early": False},
        features=_LOOP_FEATURES,
        timer=CycleTimer(200.0, 2),
        control_time_s=10.0,
        dataset=dataset,
        single_task="task",
    )

    assert dataset.add_frame.call_count == 4
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]


def test_dagger_continuous_records_once_per_interpolation_cycle():
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True, num_episodes=1))
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._episode_duration_s = 1e9

    strategy._run_continuous(ctx)

    assert dataset.add_frame.call_count == 4
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]
    for call in dataset.add_frame.call_args_list:
        assert call.args[0]["intervention"].item() is False


@pytest.mark.parametrize("correction_ticks", [1, 2, 3])
def test_dagger_records_policy_actions_after_a_correction_of_any_length(correction_ticks):
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Returning to AUTONOMOUS resets the interpolator, restarting its cycle.
    # The record phase has to restart with it: an odd-length correction
    # otherwise flips the parity, and every later autonomous frame stores the
    # interpolated midpoint (x.5) instead of the policy's own action.
    strategy = DAggerStrategy(
        DAggerStrategyConfig(record_autonomous=True, num_episodes=1, smooth_handover=False)
    )
    first_correcting = 8
    last_correcting = first_correcting + correction_ticks - 1
    schedule = {
        6: "pause_resume",  # -> PAUSED
        7: "correction",  # -> CORRECTING
        last_correcting: "correction",  # -> PAUSED
        last_correcting + 1: "pause_resume",  # -> AUTONOMOUS (interpolator reset)
    }

    def on_tick(n):
        if (event := schedule.get(n)) is not None:
            strategy._events.request_transition(event)

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=last_correcting + 10, on_tick=on_tick)
    ctx.hardware.teleop.get_action.return_value = {"m.pos": 0.0}
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._episode_duration_s = 1e9

    strategy._run_continuous(ctx)

    autonomous = [
        call.args[0]["action"][0]
        for call in dataset.add_frame.call_args_list
        if not call.args[0]["intervention"].item()
    ]
    assert autonomous, "expected autonomous frames before and after the correction"
    assert all(float(a).is_integer() for a in autonomous), autonomous


def test_dagger_correction_frames_keep_the_cycle_cadence_and_are_tagged():
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    strategy = DAggerStrategy(
        DAggerStrategyConfig(record_autonomous=True, num_episodes=1, smooth_handover=False)
    )

    # Transitions are consumed at the top of the tick after they are requested,
    # so the loop is AUTONOMOUS for ticks 1-2, PAUSED on tick 3, then CORRECTING.
    def on_tick(n):
        if n == 2:
            strategy._events.request_transition("pause_resume")
        elif n == 3:
            strategy._events.request_transition("correction")

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8, on_tick=on_tick)
    ctx.hardware.teleop.get_action.return_value = {"m.pos": 42.0}
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._episode_duration_s = 1e9

    strategy._run_continuous(ctx)

    # Human-driven frames are recorded on the same one-per-cycle cadence as
    # autonomous ones, carry the teleop action, and are tagged as interventions.
    tags = [call.args[0]["intervention"].item() for call in dataset.add_frame.call_args_list]
    assert tags == [False, True, True, True]
    assert _recorded_actions(dataset) == [1.0, 42.0, 42.0, 42.0]


# ---------------------------------------------------------------------------
# Cadence summaries through the real strategy loops
# ---------------------------------------------------------------------------


def test_base_strategy_reports_the_run_summary_when_the_loop_ends(caplog):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, _ = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        strategy.run(ctx)

    (summary,) = _info_messages(caplog)
    # Base records nothing, so there are no episode boundaries: one block covering
    # the whole run, and it names the loop-body steps the strategy wrapped.
    assert summary.startswith("Cadence summary — whole run ·")
    for step in ("observe", "process_obs", "infer", "send", "telemetry"):
        assert step in summary, step
    assert "record" not in summary


def test_base_strategy_reports_the_run_summary_even_when_the_loop_raises(caplog):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # The summary lives in a `finally` precisely so a dying camera, a duration
    # limit and a KeyboardInterrupt all still report what the loop achieved.
    ctx, _ = _make_loop_ctx(fps=200.0, multiplier=1, num_ticks=8)
    ctx.hardware.robot_wrapper.get_observation.side_effect = [
        {"m.pos": 1.0},
        RuntimeError("camera died"),
    ]
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=1)

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER), pytest.raises(RuntimeError, match="camera"):
        strategy.run(ctx)

    (summary,) = _info_messages(caplog)
    assert summary.startswith("Cadence summary —")


def test_starved_engine_is_counted_through_the_real_dispatch_path(caplog):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # An async backend with an empty queue yields no action, so the tick commands
    # nothing and records nothing.  The dataset cannot show that gap — frame
    # timestamps are synthesised from the frame index — so the count in the
    # summary is the only signal a user gets.
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=1, num_ticks=4)
    ctx.policy.inference.get_action.side_effect = lambda _obs_frame: None
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=1)

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        strategy.run(ctx)

    (summary,) = _info_messages(caplog)
    assert "ticks with no action to send (inference engine starved): 4" in summary
    assert ctx.hardware.robot_wrapper.send_action.call_count == 0
    assert dataset.add_frame.call_count == 0


def test_episodic_run_reports_a_summary_per_episode_and_for_the_run(caplog):
    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.strategies import EpisodicStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Episodic owns one timer across the whole session (episodes used to get a
    # fresh one each), so this also covers the `restart()` that keeps a
    # re-primed interpolator from being reported as a slow episode.
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    ctx.runtime.cfg.dataset = SimpleNamespace(
        single_task="task",
        episode_time_s=10.0,
        reset_time_s=0.0,
        num_episodes=1,
        push_to_hub=False,
        tags=None,
        private=False,
    )
    strategy = EpisodicStrategy(EpisodicStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}

    with caplog.at_level(logging.INFO, logger=_TIMER_LOGGER):
        strategy.run(ctx)

    messages = _info_messages(caplog)
    assert len(messages) == 2
    assert messages[0].startswith("Cadence (episode 0):")
    assert messages[1].startswith("Cadence summary — whole run, 1 episode ·")
    # Recording still lands once per interpolation cycle over the 8 ticks.
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]
    assert not _timer_warnings(caplog)


# ---------------------------------------------------------------------------
# Sync engine: relative-action anchoring (drift-free chunk execution)
# ---------------------------------------------------------------------------

_REL_ACTION_NAMES = ["j0.pos", "j1.pos", "j2.pos", "gripper.pos"]
_REL_ACTION_DIM = len(_REL_ACTION_NAMES)


def _relative_pre_post(exclude_joints=None):
    """Build fake pre/post processors wrapping real relative/absolute steps.

    The preprocessor runs the ``RelativeActionsProcessorStep`` (caching/holding the
    anchor state) and passes the observation through; the postprocessor runs the
    paired ``AbsoluteActionsProcessorStep`` (relative + cached state) and returns the
    absolute action tensor.  Shapes mirror what the sync engine feeds them.
    """
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
        TransitionKey,
        create_transition,
    )
    from lerobot.utils.constants import OBS_STATE

    relative_step = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=exclude_joints or [], action_names=list(_REL_ACTION_NAMES)
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    class _Pre:
        steps = [relative_step]

        def __call__(self, observation):
            # observation carries a batched OBS_STATE tensor; run the relative step so
            # it caches (or holds) the anchor, then pass the batch through unchanged.
            transition = create_transition(observation={OBS_STATE: observation[OBS_STATE]})
            relative_step(transition)
            return observation

        def reset(self):
            pass

    class _Post:
        def __call__(self, action):
            transition = create_transition(action=action)
            return absolute_step(transition)[TransitionKey.ACTION]

        def reset(self):
            pass

    return _Pre(), _Post(), relative_step


def _fake_relative_policy(chunk_rel, n_action_steps, with_queue=True):
    """Fake chunk policy: refills an ``_action_queue`` with ``chunk_rel`` when empty."""
    from collections import deque

    policy = MagicMock()
    policy.config.use_amp = False
    policy.config.action_feature_names = list(_REL_ACTION_NAMES)
    state = {"predict_calls": 0}

    if with_queue:
        policy._action_queue = deque(maxlen=n_action_steps)
    else:
        # Ensure the attribute is truly absent so getattr(...) falls back.
        del policy._action_queue

    def select_action(_observation):
        if with_queue:
            if len(policy._action_queue) == 0:
                state["predict_calls"] += 1
                policy._action_queue.extend(chunk_rel[i].unsqueeze(0) for i in range(n_action_steps))
            return policy._action_queue.popleft()
        # No queue: recompute every tick (like temporal ensembling).
        state["predict_calls"] += 1
        return chunk_rel[0].unsqueeze(0)

    policy.select_action.side_effect = select_action
    policy.reset.side_effect = lambda: policy._action_queue.clear() if with_queue else None
    policy._predict_state = state
    return policy


def _build_sync_engine(policy, pre, post):
    from lerobot.rollout import SyncInferenceEngine

    return SyncInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        dataset_features={"action": {"names": list(_REL_ACTION_NAMES)}},
        ordered_action_keys=list(_REL_ACTION_NAMES),
        task="test",
        device="cpu",
        robot_type="mock",
    )


def _obs_frame(state_values):
    import numpy as np

    return {"observation.state": np.asarray(state_values, dtype=np.float32)}


def test_sync_relative_holds_anchor_across_chunk():
    """Every action popped within a chunk must anchor to the tick-0 state (no drift)."""
    n = 4
    # A distinct relative offset per chunk step so a wrong anchor would be visible.
    chunk_rel = torch.stack([torch.full((_REL_ACTION_DIM,), 0.1 * (i + 1)) for i in range(n)])
    pre, post, relative_step = _relative_pre_post()
    policy = _fake_relative_policy(chunk_rel, n_action_steps=n)
    engine = _build_sync_engine(policy, pre, post)

    assert engine._relative_step is relative_step  # introspection wired the step

    s0 = [1.0, 2.0, 3.0, 4.0]
    outputs = []
    for tick in range(n):
        # Feed a *different* state each tick; a drifting anchor would use it.
        state = [v + tick for v in s0]
        outputs.append(engine.get_action(_obs_frame(state)))

    # Exactly one chunk was predicted across the n ticks.
    assert policy._predict_state["predict_calls"] == 1
    for tick in range(n):
        expected = torch.tensor(s0) + chunk_rel[tick]
        torch.testing.assert_close(outputs[tick], expected)

    # Next tick empties the queue -> recompute -> anchor refreshes to the new state.
    s_next = [10.0, 20.0, 30.0, 40.0]
    out = engine.get_action(_obs_frame(s_next))
    assert policy._predict_state["predict_calls"] == 2
    torch.testing.assert_close(out, torch.tensor(s_next) + chunk_rel[0])
    assert relative_step._hold_state is False  # released after every call


def test_sync_relative_fallback_without_action_queue():
    """A policy without ``_action_queue`` refreshes the anchor every tick."""
    n = 3
    chunk_rel = torch.stack([torch.full((_REL_ACTION_DIM,), 0.5) for _ in range(n)])
    pre, post, _ = _relative_pre_post()
    policy = _fake_relative_policy(chunk_rel, n_action_steps=n, with_queue=False)
    engine = _build_sync_engine(policy, pre, post)

    s0 = [1.0, 1.0, 1.0, 1.0]
    for tick in range(3):
        state = [v + tick for v in s0]
        out = engine.get_action(_obs_frame(state))
        # Anchor tracks the current state every tick.
        torch.testing.assert_close(out, torch.tensor(state) + chunk_rel[0])


def test_sync_engine_no_relative_step_is_none():
    """Without an enabled relative step, the engine takes the plain select_action path."""
    policy = MagicMock()
    policy.config.use_amp = False
    engine = _build_sync_engine(policy, MagicMock(steps=[]), MagicMock())
    assert engine._relative_step is None
