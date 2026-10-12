"""Recording and policy conditioning start with the same resolved instruction."""

from dataclasses import asdict
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("datasets")
pytest.importorskip("msgpack")

from lerobot.inference import RemoteInferenceConfig
from lerobot.rollout import context
from tests.remote_inference.test_engine import ControlledClient


@pytest.fixture
def setup(monkeypatch, tmp_path):
    client = ControlledClient()
    client.admit = Mock()
    client.descriptor.update(capabilities=asdict(client.capabilities), instance_id=client.instance_id)
    robot = SimpleNamespace(
        name="test_position_robot",
        supports_position_hold=True,
        action_features={"a.pos": float, "b.pos": float},
        observation_features={"a.pos": float, "b.pos": float},
        connect=Mock(),
        disconnect=Mock(),
        is_connected=True,
        get_observation=lambda: {"a.pos": 0.0, "b.pos": 0.0},
    )
    monkeypatch.setattr(context.RemoteClient, "connect", lambda _: client)
    make_robot = Mock(return_value=robot)
    monkeypatch.setattr("lerobot.rollout.context.make_robot_from_config", make_robot)
    monkeypatch.setattr(
        context,
        "_build_rollout_dataset",
        lambda cfg, *_: SimpleNamespace(root=tmp_path) if cfg.dataset else None,
    )
    cfg = SimpleNamespace(
        inference=RemoteInferenceConfig(deployment="test", semantics="radians"),
        robot=SimpleNamespace(type="test_position_robot"),
        teleop=None,
        policy=None,
        device=None,
        use_torch_compile=False,
        task="top-level instruction",
        dataset=SimpleNamespace(single_task="recorded instruction", video=False),
        rename_map={},
        fps=10,
    )
    return cfg, client, robot, make_robot


@pytest.mark.parametrize("recording", [True, False])
def test_remote_initial_conditioning_and_labels_follow_local_task_precedence(setup, recording):
    cfg, client, robot, _ = setup
    if not recording:
        cfg.dataset = None
    expected = cfg.dataset.single_task if recording else cfg.task
    ctx = context.build_remote_rollout_context(cfg, Event())
    try:
        engine = ctx.policy.inference
        assert engine.task == engine.dispatched_task == expected
        engine.notify_observation({"a.pos": 0.0, "b.pos": 0.0})
        assert engine._observation.task == expected
        client.admit.assert_called_once()
    finally:
        ctx.policy.inference.stop()
        robot.disconnect()


@pytest.mark.parametrize("recorded_task, top_level_task", [("too long", "ok"), ("ok", "too long")])
def test_initial_length_guard_validates_the_instruction_actually_used(setup, recorded_task, top_level_task):
    cfg, client, robot, make_robot = setup
    client.descriptor["limits"]["max_input_chars"] = 3
    cfg.dataset.single_task = recorded_task
    cfg.task = top_level_task
    if len(recorded_task) > 3:
        with pytest.raises(ValueError, match="Initial instruction"):
            context.build_remote_rollout_context(cfg, Event())
        make_robot.assert_not_called()
        assert client.closed.is_set()
    else:
        ctx = context.build_remote_rollout_context(cfg, Event())
        try:
            assert ctx.policy.inference.task == recorded_task
        finally:
            ctx.policy.inference.stop()
            robot.disconnect()
