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
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# Hoisted rather than re-imported inside each test: the module-scoped `importorskip`
# above already guards it, and the timer tests below reference it ~20 times.
from lerobot.rollout.strategies import CycleTimer  # noqa: E402

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
# CycleTimer
# ---------------------------------------------------------------------------

_CORE_LOGGER = "lerobot.rollout.strategies.core"


def _timer_warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


class _FakeClock:
    """Virtual clock standing in for ``time.perf_counter`` and ``precise_sleep``.

    ``CycleTimer``'s contract is pure arithmetic over deadlines, so exercising it
    against the wall clock only adds scheduler noise: every margin has to be wide
    enough for a loaded CI machine, which makes the assertions loose and the suite
    slow.  Driving it here instead makes the pacing exact — ``advance`` stands in
    for loop-body work, and the timer's own sleeps move the same clock forward.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.overshoot = 0.0
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Simulate *seconds* of work inside the loop body."""
        self.now += seconds

    def precise_sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        # `overshoot` models a sleep that returns late — the OS descheduling the
        # process, or a coarse timer granularity.
        self.now += seconds + self.overshoot

    def __getattr__(self, name):
        # Anything else CycleTimer's module reaches for on `time` still works.
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    """Patch the strategies-core clock, scoped to that module's namespace only."""
    from lerobot.rollout.strategies import core

    fake = _FakeClock()
    monkeypatch.setattr(core, "time", fake)
    monkeypatch.setattr(core, "precise_sleep", fake.precise_sleep)
    return fake


def _drive(timer, clock, work=0.0, ticks=1, new_cycle=True):
    """Run *ticks* iterations of a loop body costing *work* seconds each.

    Stands in for a strategy's loop: ``tick``, burn *work* on the virtual clock,
    ``wait``.  *new_cycle* is what the interpolator would report — ``True`` models a
    starved or frozen one (every tick asks for a fresh action), and a callable
    ``tick_index -> bool`` covers the cadences in between.
    """
    for i in range(ticks):
        timer.tick(new_cycle=new_cycle(i) if callable(new_cycle) else new_cycle)
        clock.advance(work)
        timer.wait()


def _debug_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]


def _info_messages(caplog):
    """INFO messages from the strategies-core logger only (i.e. the cadence reports)."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.name == _CORE_LOGGER]


def test_cycle_timer_validates_arguments():
    with pytest.raises(ValueError, match="fps"):
        CycleTimer(0.0)
    with pytest.raises(ValueError, match="multiplier"):
        CycleTimer(30.0, 0)


def test_cycle_timer_paces_ticks_to_base_fps(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        _drive(timer, clock, ticks=4, new_cycle=lambda i: i % 2 == 0)  # two full cycles
    assert clock.now == pytest.approx(2 * (1 / 10.0))
    assert not _timer_warnings(caplog)


def test_cycle_timer_spaces_interpolated_commands_evenly(clock):
    # Interpolation exists to smooth motion, so every tick must be spaced by
    # 1/(fps × multiplier) — not batched at the start of each cycle.
    timer = CycleTimer(10.0, 2)  # 50 ms slots
    stamps = []
    for tick in range(4):
        timer.tick(new_cycle=tick % 2 == 0)
        stamps.append(clock.now)
        timer.wait()
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert gaps == pytest.approx([0.05, 0.05, 0.05])


def test_cycle_timer_slow_policy_tick_borrows_from_interpolated_ticks(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        timer.tick(new_cycle=True)
        clock.advance(0.06)  # policy tick overruns its 50 ms slot
        timer.wait()
        _drive(timer, clock, new_cycle=False)  # instant interpolated tick absorbs it
    # The 60 ms + 0 ms cycle fits the 100 ms budget: no user-facing warning,
    # only a DEBUG note about the tick that missed its slot.
    assert not _timer_warnings(caplog)
    assert any("slot" in m for m in _debug_messages(caplog))
    # The cycle still ends on its deadline, so the policy cadence is held.
    assert clock.now == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("fps", "multiplier", "work", "ticks", "expected"),
    [
        # Every tick is its own cycle at multiplier 1, so each slow tick past the
        # exempt start-up one warns: 80 ms of work against a 50 ms budget.
        (20.0, 1, 0.08, 3, 2),
        # At multiplier 2 a single 120 ms tick already blows the 100 ms cycle
        # budget, but it is judged per group of two: 4 ticks, 2 groups, 1 exempt.
        (10.0, 2, 0.12, 4, 1),
    ],
    ids=["multiplier-1", "multiplier-2"],
)
def test_cycle_timer_warns_once_per_over_budget_cycle(caplog, clock, fps, multiplier, work, ticks, expected):
    timer = CycleTimer(fps, multiplier)
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        _drive(timer, clock, work=work, ticks=ticks)
    warnings = _timer_warnings(caplog)
    assert len(warnings) == expected
    assert f"target FPS ({fps:g}" in warnings[0].getMessage()
    assert "Dataset frames" in warnings[0].getMessage()


def test_cycle_timer_does_not_report_the_startup_group(caplog, clock):
    # The interpolator primes its buffer with a single action, so inference runs
    # on two consecutive ticks at start-up and the first group legitimately runs
    # over budget.  Reporting it would warn on every healthy launch.
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.12, ticks=2)
    assert not _timer_warnings(caplog)


def test_cycle_timer_control_only_phrasing(caplog, clock):
    timer = CycleTimer(20.0, 1, records_data=False)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.08, ticks=2)
    (warning,) = _timer_warnings(caplog)
    assert "Dataset frames" not in warning.getMessage()
    assert "Robot control might be unstable" in warning.getMessage()


def test_cycle_timer_debug_message_omits_recording_when_not_recording(caplog, clock):
    timer = CycleTimer(10.0, 2, records_data=False)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        # Overruns the 50 ms slot, fits the 100 ms budget.
        _drive(timer, clock, work=0.06)
    (debug,) = _debug_messages(caplog)
    assert "recording" not in debug


def test_cycle_timer_reports_a_slot_overrun_on_the_tick_that_closes_a_group(caplog, clock):
    # Regression: the overrun note used to sit in an `elif` behind the group-close
    # branch, so every multiplier-th tick's overrun went unreported even when the
    # group as a whole was healthy.  Here the *closing* tick is the slow one.
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _drive(timer, clock, ticks=2, new_cycle=lambda i: i == 0)  # start-up group, exempt
        timer.tick(new_cycle=True)
        timer.wait()  # instant opening tick
        timer.tick(new_cycle=False)
        clock.advance(0.06)  # closing tick overruns its 50 ms slot...
        timer.wait()
    assert not _timer_warnings(caplog)  # ...while 60 ms still fits the 100 ms budget
    assert [m for m in _debug_messages(caplog) if "overran its" in m]


def test_cycle_timer_warns_when_every_tick_reports_a_new_cycle(caplog, clock):
    # Regression: when the interpolator is starved or frozen (async backend
    # yielding no action, DAgger paused/correcting), every tick reports
    # new_cycle=True.  Tying the slow-loop warning to cycle completion made it
    # structurally unreachable in exactly that regime.  Groups of `multiplier`
    # ticks are measured regardless, so a genuinely slow loop still warns.
    timer = CycleTimer(10.0, 2)  # 100 ms budget per 2 ticks
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.07, ticks=6)  # 140 ms per 2-tick group, 3 groups
    warnings = _timer_warnings(caplog)
    assert len(warnings) == 2  # one per closed group, not one per tick
    assert "target FPS (10" in warnings[0].getMessage()


@pytest.mark.parametrize("multiplier", [2, 3])
def test_cycle_timer_silent_on_healthy_loop_driven_by_the_real_interpolator(caplog, clock, multiplier):
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Regression: the reporting window must not depend on WHERE it starts
    # relative to the policy cycle.  The real interpolator primes with a
    # single-action buffer, which permanently offsets groups from cycles, so
    # hand-aligned `new_cycle` flags (what a naive test supplies) hide the bug.
    # Drive the flags from the real interpolator instead, with an inference tick
    # deliberately longer than one 1/(fps × N) slot — the regime interpolation
    # exists for — while total work per cycle stays well inside the 1/fps budget.
    fps = 20.0
    policy_work = 1.0 / (fps * multiplier) * 1.5  # overruns its slot
    interp_work = 0.002
    assert policy_work + (multiplier - 1) * interp_work < 1.0 / fps  # healthy loop

    interpolator = ActionInterpolator(multiplier=multiplier)
    timer = CycleTimer(fps, multiplier)
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        for i in range(4 * multiplier):
            needs_action = interpolator.needs_new_action()
            timer.tick(new_cycle=needs_action)
            if needs_action:
                interpolator.add(torch.tensor([float(i)]))
                clock.advance(policy_work)
            else:
                clock.advance(interp_work)
            interpolator.get()
            timer.wait()
    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]


def test_cycle_timer_reports_time_lost_inside_the_pacing_sleep(caplog, clock):
    # Loop-body work fits the budget comfortably, but every pacing sleep returns
    # 10 ms late, so the achieved cadence really is below target.  That shortfall
    # is not the caller's doing and would fire constantly on a loaded machine, so
    # it is reported at DEBUG rather than as the slow-loop warning.
    clock.overshoot = 0.01
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.01, ticks=6)

    assert not _timer_warnings(caplog)
    debugs = _debug_messages(caplog)
    assert any("went missing outside the loop body" in m for m in debugs), debugs


def test_cycle_timer_tolerates_a_sliver_of_sleep_overshoot(caplog, clock):
    # Sleeps that return a few hundred microseconds late are the ordinary state of
    # a paced loop, not a cadence problem.  With no tolerance at all this note
    # fired on nearly every group of a healthy run, which made both the message
    # and the count in the summary worthless.
    clock.overshoot = 0.0004  # 0.8 ms per 100 ms cycle: inside the 1% tolerance
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.01, ticks=6)

    assert not caplog.records


def test_cycle_timer_stays_silent_about_sleeps_on_a_punctual_clock(caplog, clock):
    # Same loop, sleeps that return on time: nothing to report at all.
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.01, ticks=6)

    assert not caplog.records


def test_cycle_timer_starved_but_fast_loop_stays_silent(caplog, clock):
    # The same all-new-cycle regime must not warn when the loop is keeping up.
    timer = CycleTimer(10.0, 2)
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        _drive(timer, clock, ticks=4)
    assert not _timer_warnings(caplog)


def test_cycle_timer_new_cycle_reanchors_pacing(clock):
    # The interpolator's startup buffer holds a single action, so the second
    # tick requests a fresh action one tick early.  Re-anchoring must restart
    # the pacing slots from that tick rather than keep the stale anchor.
    timer = CycleTimer(10.0, 2)  # 50 ms slots
    timer.tick(new_cycle=True)
    timer.wait()  # sleeps to the first slot deadline
    reanchor = clock.now
    timer.tick(new_cycle=True)
    timer.wait()
    # Paced from the re-anchor, so a full slot — not the 0 ms a stale anchor
    # (already past its second deadline) would have produced.
    assert clock.now - reanchor == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Cadence statistics and summaries
# ---------------------------------------------------------------------------


def test_cycle_timer_run_summary_reports_effective_cadence_and_sections(caplog, clock):
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    for i in range(6):
        timer.tick(new_cycle=i % 2 == 0)
        with timer.section("observe"):
            clock.advance(0.01)
        with timer.section("infer"):
            clock.advance(0.02 if i % 2 == 0 else 0.0)  # inference on policy ticks only
        timer.wait()

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    # The sample size is in the heading, so it survives even when the rate below
    # cannot be computed.
    assert summary.startswith(
        "Cadence summary — whole run · target 10 Hz × 2 (50.0 ms tick slot, 100.0 ms cycle budget): "
        "6 ticks, 2 cycles judged"
    )
    # A correctly paced loop holds its target exactly on a virtual clock.
    # 5 gaps of 50 ms — the span sums gaps, so it is one gap short of 6 ticks' worth.
    assert "effective cadence: 10.00 Hz policy / 20.00 Hz commands over 0.2 s measured" in summary
    # Three groups of two, the first exempt as start-up; 40 ms of work each.
    assert "cycles over the 100.0 ms work budget: 0/2 (0.0%)" in summary
    assert "work mean 40.0 ms, worst 40.0 ms" in summary
    # Sections split the measured work; both ran on every tick, 30 ms each cycle.
    assert "observe" in summary and "infer" in summary
    assert summary.count("6 calls") == 2
    # Headroom is the pacing sleep, which is the whole budget minus the work: the
    # policy ticks sleep 20 ms of their 50 ms slot, the interpolated ticks 40 ms.
    assert "pacing headroom: 30.0 ms slept per tick on average (max 40.0 ms)" in summary


def test_cycle_timer_sections_report_calls_mean_and_worst(caplog, clock):
    timer = CycleTimer(10.0, 1)
    for work in (0.01, 0.03, 0.02):
        timer.tick(new_cycle=True)
        with timer.section("observe"):
            clock.advance(work)
        timer.wait()

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "observe  mean  20.00 ms · worst  30.00 ms · 100.0% of work · 3 calls" in summary


def test_cycle_timer_episode_summaries_fold_into_the_run_total(caplog, clock):
    timer = CycleTimer(10.0, 1)  # every tick is a cycle, 100 ms budget
    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.01, ticks=4)
        timer.log_episode_summary("episode 1")
        _drive(timer, clock, work=0.01, ticks=4)
        timer.log_episode_summary("episode 2")
        _drive(timer, clock, work=0.01, ticks=2)
        timer.log_run_summary()  # closes the window still open

    messages = _info_messages(caplog)
    assert len(messages) == 4
    assert [m.split(":")[0] for m in messages[:3]] == [
        "Cadence (episode 1)",
        "Cadence (episode 2)",
        "Cadence (final episode)",
    ]
    assert messages[0].startswith("Cadence (episode 1): 10.00 Hz policy vs 10 Hz target · 4 ticks")
    # The run total covers every tick of all three windows, and only the run block
    # carries the full breakdown.
    assert messages[3].startswith("Cadence summary — whole run, 3 episodes")
    assert "10 ticks" in messages[3]
    assert "pacing headroom: 90.0 ms slept per tick on average (max 90.0 ms)" in messages[3]


def test_cycle_timer_run_summary_always_states_its_sample_size(caplog, clock):
    # Every other number in the block is a rate or an average over the ticks, so the
    # tick count has to be there to judge any of them.  It used to ride on the
    # effective-cadence line, which is skipped when no rate could be measured — as
    # here, where each tick is its own episode so every elapsed gap is dropped.
    timer = CycleTimer(200.0, 1)
    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        for i in range(4):
            timer.tick(new_cycle=True)
            clock.advance(0.0001)
            timer.log_episode_summary(f"episode {i}")
            timer.wait()
        timer.log_run_summary()

    run = _info_messages(caplog)[-1]
    assert "4 ticks, 3 cycles judged" in run
    assert "effective cadence" not in run  # nothing to measure a rate from


def test_cycle_timer_empty_window_reports_nothing(clock, caplog):
    # Boundaries can fall before any tick completes (a rotation on the very first
    # iteration), and a strategy should not have to guard against that.
    timer = CycleTimer(10.0, 1)
    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_episode_summary("episode 1")
        timer.log_run_summary()
    assert not _info_messages(caplog)


def test_cycle_timer_restart_inside_a_loop_body_drops_the_stall_it_follows(caplog, clock):
    # Regression: `restart()` is only ever called from *inside* a loop body — after
    # the blocking work and before `wait()` (a ring-buffer flush, DAgger's handover
    # ramp).  Elapsed time is a backward-looking gap, so cutting the link at that
    # moment discarded the healthy gap already behind the tick and billed the stall
    # to the next one, understating the achieved cadence instead of excluding it.
    timer = CycleTimer(10.0, 1)  # 100 ms budget
    _drive(timer, clock, work=0.005, ticks=4)
    timer.tick(new_cycle=True)
    clock.advance(0.005)  # honest work...
    clock.advance(0.5)  # ...then the blocking one-off, still inside the body
    timer.restart()
    timer.wait()
    _drive(timer, clock, work=0.005, ticks=4)

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "9 ticks" in summary
    assert "effective cadence: 10.00 Hz policy over 0.7 s measured" in summary
    assert not _timer_warnings(caplog)  # the stalled group is exempted too


def test_cycle_timer_episode_boundary_inside_a_loop_body_belongs_to_the_closing_episode(caplog, clock):
    # Regression: strategies request a boundary after the blocking `save_episode()`
    # and before `wait()`, so the tick in progress is the closing episode's last one.
    # Folding the window there booked that tick — the entire save — against the
    # episode about to start, whose headline cadence then collapsed while the work
    # column beside it still read healthy.
    timer = CycleTimer(10.0, 1)
    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        _drive(timer, clock, work=0.005, ticks=4)
        timer.tick(new_cycle=True)  # the rotation tick
        clock.advance(0.005)
        clock.advance(0.5)  # blocking save_episode
        timer.log_episode_summary("episode 1")
        timer.restart()
        timer.wait()
        _drive(timer, clock, work=0.005, ticks=4)
        timer.log_run_summary()

    episode_1, episode_2, run = _info_messages(caplog)
    # The rotation tick counts toward the episode it finalised, and its stalled group
    # is exempt, so the work column stays honest at 5 ms rather than 505 ms.
    assert episode_1 == (
        "Cadence (episode 1): 10.00 Hz policy vs 10 Hz target · 5 ticks, 0.4 s measured · "
        "0/3 cycles over the 100.0 ms budget (work mean 5.0 ms, worst 5.0 ms)"
    )
    # ...and the next episode is not slandered by a stall it never paid for.
    assert episode_2.startswith(
        "Cadence (final episode): 10.00 Hz policy vs 10 Hz target · 4 ticks, 0.3 s measured"
    )
    assert "9 ticks" in run and "effective cadence: 10.00 Hz policy" in run


def test_cycle_timer_summary_counts_ticks_that_overran_their_slot(caplog, clock):
    # The per-tick slot overrun is only visible in the summary (its log line is
    # DEBUG), so the counter needs its own assertion — without one, disabling the
    # increment entirely leaves the suite green.
    timer = CycleTimer(10.0, 2)  # 50 ms slots, 100 ms cycle budget
    _drive(timer, clock, ticks=2, new_cycle=lambda i: i == 0)  # start-up group
    for _ in range(2):
        timer.tick(new_cycle=True)
        clock.advance(0.06)  # overruns the slot, still inside the cycle budget
        timer.wait()
        _drive(timer, clock, new_cycle=False)  # instant interpolated tick

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "ticks over their 50.0 ms slot: 2/6 (costs interpolation smoothness only)" in summary
    assert not _timer_warnings(caplog)


def test_cycle_timer_statistics_survive_restart_but_drop_the_blocking_gap(caplog, clock):
    # `restart()` exists for one-off blocking work inside the timed body (DAgger's
    # handover ramps, a ring-buffer flush).  The statistics still have to describe
    # the whole run, while the stall itself must not be billed to the cadence.
    timer = CycleTimer(10.0, 1)
    _drive(timer, clock, work=0.01, ticks=3)
    clock.advance(2.0)  # a two-second stall inside the loop body
    timer.restart()
    _drive(timer, clock, work=0.01, ticks=3)

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "6 ticks" in summary  # every tick from before the restart is still counted
    assert "10.00 Hz policy" in summary  # ...but the stall is not counted against it


def test_cycle_timer_starved_ticks_are_counted_and_reported(caplog, clock):
    timer = CycleTimer(10.0, 1)
    _drive(timer, clock, work=0.01, ticks=2)
    timer.note_starved_tick()
    timer.note_starved_tick()

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        timer.log_run_summary()

    (summary,) = _info_messages(caplog)
    assert "ticks with no action to send (inference engine starved): 2" in summary


def test_handle_warmup_is_a_noop_without_torch_compile():
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy

    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = MagicMock(ready=False)
    timer = CycleTimer(20.0, 2)

    assert strategy._handle_warmup(False, timer) is False
    strategy._engine.reset.assert_not_called()


def test_handle_warmup_paces_then_flushes_and_exempts_the_reprimed_group(caplog, clock):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy, CycleTimer
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

    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
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
    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
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
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event),
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


def test_base_strategy_drives_robot_without_recording():
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)

    strategy.run(ctx)

    # Commands go out every tick (fps × 2) while inference stays at fps, and
    # nothing is recorded.
    assert ctx.hardware.robot_wrapper.send_action.call_count == 8
    assert ctx.policy.inference.get_action.call_count == 5
    assert dataset.add_frame.call_count == 0


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

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
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


def test_dagger_resume_does_not_warn_about_the_reprimed_interpolator(caplog, clock):
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Returning to AUTONOMOUS resets the interpolator, which re-primes over two
    # ticks exactly like loop start-up.  On a healthy loop that must not be
    # reported as a slow cycle.
    fps, multiplier = 20.0, 2
    policy_work = 1.0 / (fps * multiplier) * 1.5  # overruns its slot, fits the cycle
    strategy = DAggerStrategy(
        DAggerStrategyConfig(record_autonomous=True, num_episodes=1, smooth_handover=False)
    )

    def on_tick(n):
        if n == 6:
            strategy._events.request_transition("pause_resume")  # -> PAUSED
        elif n == 8:
            strategy._events.request_transition("correction")  # -> CORRECTING
        elif n == 12:
            strategy._events.request_transition("correction")  # -> PAUSED
        elif n == 14:
            strategy._events.request_transition("pause_resume")  # -> AUTONOMOUS

    ctx, dataset = _make_loop_ctx(fps=fps, multiplier=multiplier, num_ticks=22, on_tick=on_tick)
    ctx.hardware.teleop.get_action.return_value = {"m.pos": 42.0}
    ctx.policy.inference.get_action.side_effect = lambda _f: (clock.advance(policy_work), torch.zeros(1))[1]
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=multiplier)
    strategy._episode_duration_s = 1e9

    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        strategy._run_continuous(ctx)

    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]


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


def test_dagger_handover_ramp_is_not_reported_as_a_slow_loop(caplog, clock, monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_mod
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import CycleTimer, DAggerPhase, DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # The smooth-handover ramps block for a good fraction of a second *inside*
    # the timed loop body.  That is a one-off operator event, not the steady-state
    # cadence, so the group carrying it must be dropped rather than reported —
    # otherwise every takeover warns that the loop is slow.
    monkeypatch.setattr(dagger_mod, "teleop_smooth_move_to", lambda *_: clock.advance(0.5))
    monkeypatch.setattr(dagger_mod, "teleop_supports_feedback", lambda _: True)

    fps, multiplier = 20.0, 2  # 50 ms cycle budget
    strategy = DAggerStrategy(
        DAggerStrategyConfig(record_autonomous=True, num_episodes=1, smooth_handover=True)
    )
    ctx, _ = _make_loop_ctx(fps=fps, multiplier=multiplier, num_ticks=1)
    interpolator = ActionInterpolator(multiplier=multiplier)
    timer = CycleTimer(fps, multiplier)

    def healthy_ticks(count):
        for _ in range(count):
            timer.tick(new_cycle=True)
            clock.advance(0.001)
            timer.wait()

    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        healthy_ticks(4)  # spend the start-up exemption
        timer.tick(new_cycle=True)
        strategy._apply_transition(
            DAggerPhase.AUTONOMOUS,
            DAggerPhase.PAUSED,
            ctx.policy.inference,
            interpolator,
            ctx,
            {"m.pos": 1.0},
            timer,
        )
        timer.wait()
        healthy_ticks(4)

    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]


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

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
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

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER), pytest.raises(RuntimeError, match="camera"):
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

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        strategy.run(ctx)

    (summary,) = _info_messages(caplog)
    assert "ticks with no action to send (inference engine starved): 4" in summary
    assert ctx.hardware.robot_wrapper.send_action.call_count == 0
    assert dataset.add_frame.call_count == 0


def test_sentry_reports_a_cadence_summary_per_episode_and_for_the_run(caplog):
    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=1, num_ticks=4)
    strategy = _make_sentry(ctx, 1)
    strategy._episode_duration_s = 0.0  # rotate on every tick

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        strategy.run(ctx)

    messages = _info_messages(caplog)
    # Rotating on literally every tick is degenerate — it exists here only to drive
    # the reporting plumbing.  Each boundary is requested mid-tick and lands on the
    # next one, so the four rotations produce four one-tick digests (the last flushed
    # by the run summary) and no separate trailing window.
    assert len(messages) == 5
    assert [m.split(":")[0] for m in messages[:4]] == ["Cadence (episode 0)"] * 4
    assert messages[4].startswith("Cadence summary — whole run, 4 episodes")
    assert dataset.save_episode.call_count >= 4


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

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        strategy.run(ctx)

    messages = _info_messages(caplog)
    assert len(messages) == 2
    assert messages[0].startswith("Cadence (episode 0):")
    assert messages[1].startswith("Cadence summary — whole run, 1 episode ·")
    # Recording still lands once per interpolation cycle over the 8 ticks.
    assert _recorded_actions(dataset) == [1.0, 2.0, 3.0, 4.0]
    assert not _timer_warnings(caplog)


def test_episodic_exempts_each_episode_reprimed_interpolator(caplog, clock):
    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.strategies import EpisodicStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    # Every episode resets the interpolator, which then re-primes over two
    # consecutive inference ticks — exactly like loop start-up, so that group
    # legitimately runs over budget.  Episodes used to get a fresh timer each (and
    # the exemption for free); now one timer spans the session and `run()` has to
    # re-arm it, or the second episode onward warns on every healthy launch.
    events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}

    def on_tick(n):
        if n == 4:
            events["exit_early"] = True  # end episode 1; tick 5 sees it and breaks
        if n in (5, 6):
            clock.advance(0.5)  # episode 2's two re-priming inference ticks

    ctx, _ = _make_loop_ctx(fps=200.0, multiplier=2, num_ticks=8, on_tick=on_tick)
    ctx.runtime.cfg.dataset = SimpleNamespace(
        single_task="task",
        episode_time_s=10.0,
        reset_time_s=0.0,
        num_episodes=2,
        push_to_hub=False,
        tags=None,
        private=False,
    )
    # No teleop and no homing: with two episodes the reset phase between them runs,
    # and a mock teleop would drag a real two-second smooth-handover ramp into a test
    # that is about the timer.
    ctx.hardware.teleop = None
    strategy = EpisodicStrategy(EpisodicStrategyConfig(reset_to_initial_position=False))
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._events = events

    with caplog.at_level(logging.WARNING, logger=_CORE_LOGGER):
        strategy.run(ctx)

    assert not _timer_warnings(caplog), [r.getMessage() for r in _timer_warnings(caplog)]


def test_dagger_continuous_reports_a_cadence_summary_per_episode_and_for_the_run(caplog):
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx, dataset = _make_loop_ctx(fps=200.0, multiplier=1, num_ticks=4)
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True, num_episodes=1))
    strategy._engine = ctx.policy.inference
    strategy._interpolator = ActionInterpolator(multiplier=1)
    strategy._episode_duration_s = 0.0  # rotate on every tick

    with caplog.at_level(logging.INFO, logger=_CORE_LOGGER):
        strategy._run_continuous(ctx)

    messages = _info_messages(caplog)
    # As in the sentry case, rotating every tick is only a way to drive the
    # reporting: four boundaries, each landing on the following tick.
    assert [m.split(":")[0] for m in messages[:-1]] == ["Cadence (episode 0)"] * 4
    assert messages[-1].startswith("Cadence summary — whole run, 4 episodes")
