# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
import threading
from pathlib import Path
from unittest.mock import Mock

import draccus
import numpy as np
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.language_task import task_from_recipe
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig  # noqa: F401
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.rollout.planner import PlannerConfig, VisionLanguagePlanner
from tests.test_interactive_rollout import _FakeEngine


@pytest.mark.parametrize("credential", [None, "", " \t\n"])
def test_missing_planner_credential_fails_before_policy_or_hardware(monkeypatch, credential):
    import lerobot.rollout.context as rollout_context
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    key_name = "LEROBOT_TEST_PLANNER_KEY"
    if credential is None:
        monkeypatch.delenv(key_name, raising=False)
    else:
        monkeypatch.setenv(key_name, credential)
    # A default-endpoint key must not satisfy a separately configured credential.
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-default")
    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        policy=ACTConfig(device="cpu"),
        device="cpu",
        interactive=True,
        planner=PlannerConfig(enabled=True, api_key_env=key_name),
    )
    load_policy, make_robot, post = Mock(), Mock(), Mock()
    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", load_policy)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", make_robot)
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    with pytest.raises(ValueError, match=f"Set {key_name}"):
        rollout_context.build_rollout_context(cfg, threading.Event())
    load_policy.assert_not_called()
    make_robot.assert_not_called()
    post.assert_not_called()


def test_external_planning_reuses_task_switch_and_holds_on_failure():
    engine = _FakeEngine()
    engine.supports_text_queries = False
    planner = Mock(return_value="reach for the tape")
    engine.set_language_planner(planner)
    assert engine.supports_planning and not engine.supports_text_queries
    assert engine.planner_halted
    engine.start_autosteer("put tape in bin", 0)
    assert engine.pump_query({"base": "image"})
    assert engine.task == "reach for the tape"
    assert engine._take_task()[1]
    assert not engine.planner_halted
    engine.pump_query({"base": "later image"})
    assert engine._take_task()[1]  # identical command still flushes pre-API queued actions
    planner.side_effect = TimeoutError("no answer")
    engine.pump_query({"base": "next image"})
    assert engine.planner_halted
    assert engine.autosteer_goal is None
    # A real sync engine must not select an action while the planner is halted.
    assert SyncInferenceEngine.get_action(engine, {"observation.state": object()}) is None
    engine.set_task("open the left gripper")
    assert not engine.planner_halted


def test_manual_language_override_discards_in_flight_plan():
    engine = _FakeEngine()

    def planner(*args):
        engine.stop_autosteer()
        engine.set_task("operator instruction")
        return "stale plan"

    engine.set_language_planner(planner)
    engine.start_autosteer("goal", 0)
    engine.pump_query({"base": "image"})
    assert engine.task == "operator instruction"
    assert not engine.planner_halted


def test_planner_audit_times_returns_holds_and_failures_without_claiming_execution(monkeypatch, tmp_path):
    decision = {
        "command": "reach for tape",
        "camera": None,
        "points": [],
        "point_mode": None,
        "style": "subtask",
        "assessment": "tape visible",
        "status": "continue",
    }
    post = Mock(
        side_effect=lambda *a, **kw: Mock(
            json=lambda: {
                "status": "completed",
                "id": "test-response",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    log = tmp_path / "planner.jsonl"
    planner = VisionLanguagePlanner(PlannerConfig(camera_keys=["base"], log_path=str(log)))
    obs = {"base": np.zeros((48, 64, 3), dtype=np.uint8)}
    planner(obs, "goal", 1)
    decision["status"] = "uncertain"
    with pytest.raises(ValueError, match="Planner stopped"):
        planner(obs, "goal", 1)
    post.side_effect = TimeoutError("test-only-secret must not enter the journal")
    with pytest.raises(TimeoutError):
        planner(obs, "goal", 1)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    terminal = [e for e in events if "elapsed_s" in e]
    assert [e["event"] for e in terminal] == ["planner_returned", "planner_hold", "planner_error"]
    assert all(e["elapsed_s"] >= 0 and e["started_at"] <= e["ended_at"] for e in terminal)
    assert len({e["request_id"] for e in terminal}) == 3
    assert terminal[0]["execution_verified"] is False
    assert "test-only-secret" not in log.read_text()


def test_planner_sends_named_images_and_bounded_history_without_action_tools(monkeypatch):
    calls = []
    decision = {
        "command": "reach for tape",
        "camera": None,
        "points": [],
        "point_mode": None,
        "style": "subtask",
        "assessment": "tape visible",
        "status": "continue",
    }

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Mock(
            json=lambda: {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    planner = VisionLanguagePlanner(PlannerConfig(camera_keys=["base"], history_turns=1))
    obs = {"base": np.zeros((48, 64, 3), dtype=np.uint8)}
    for _ in range(3):
        assert planner(obs, "goal", 1) == "reach for tape"
    assert [len(c["input"]) for c in calls] == [1, 3, 3]
    assert "tools" not in calls[0]
    assert calls[0]["store"] is False
    assert "64x48" in calls[0]["input"][0]["content"][1]["text"]
    planner(obs, "goal", 2)
    assert len(calls[-1]["input"]) == 1
    decision["status"] = "complete"
    with pytest.raises(ValueError, match="Planner stopped"):
        planner(obs, "goal", 2)
    planner.config.styles.append("point")
    planner.config.grounding_camera_keys.append("base")
    decision.update(status="continue", style="point", camera="base", points=[[32, 24]], point_mode="targets")
    assert planner(obs, "goal", 2) == "In base view (64x48 pixels), reach for tape: [32, 24]."
    decision["points"] = [[64, 24]]
    with pytest.raises(ValueError, match="outside"):
        planner(obs, "goal", 2)
    planner.config.styles.append("combination")
    decision.update(style="combination", points=[[32, 24], [33, 25]], point_mode="path")
    with pytest.raises(ValueError, match="trace steering"):
        planner(obs, "goal", 2)


@pytest.mark.parametrize("coordinate_format", ["original_pixels", "native_points_v1"])
def test_four_gpu_training_config_uses_main_parser(tmp_path, coordinate_format):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, argv = module["prepare_run"](tmp_path, 4, 1, True, coordinate_format=coordinate_format)
    parsed = draccus.decode(TrainPipelineConfig, config)
    assert parsed.policy.type == "wall_x"
    assert parsed.policy.steering_coordinate_format == coordinate_format
    assert parsed.policy.base_model_revision == "44e827683819957d8c574e8b746a1a97e77f518a"
    assert parsed.policy.recipe["messages"][0]["stream"] == "low_level"
    assert parsed.steps == parsed.eval_steps == parsed.save_freq == 10
    assert parsed.max_eval_samples == 20
    assert "--nproc-per-node=4" in argv
    with pytest.raises(ValueError):
        module["prepare_run"](tmp_path, 5, 1, True)


def test_smoke_reload_requires_saved_checkpoint_and_preserves_topology(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, argv = module["prepare_run"](tmp_path, 4, 1, True)
    with pytest.raises(FileNotFoundError, match="did not save"):
        module["reload_command"](argv, tmp_path, config["steps"])
    checkpoint = tmp_path / "training/checkpoints/last/pretrained_model/train_config.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(config))
    reload_argv = module["reload_command"](argv, tmp_path, config["steps"])
    assert "--nproc-per-node=4" in reload_argv
    assert [arg for arg in reload_argv if arg.startswith("--config_path=")] == [f"--config_path={checkpoint}"]
    assert "--resume=true" in reload_argv
    assert "--steps=11" in reload_argv
    assert "--eval_steps=11" in reload_argv
    assert "--save_checkpoint=false" in reload_argv


def test_rebot_task_branch_corrects_source_task_in_both_training_conditions(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, _ = module["prepare_run"](tmp_path, 1, 1, True)
    recipe = TrainingRecipe.from_dict(config["dataset"]["task_recipe"])
    sample = {
        "task": "Pick up all blocks on the table and place them into the green bin.",
        "timestamp": 0,
        "index": 0,
    }
    expected = "Pick up objects from the table and place them into the bin."
    assert task_from_recipe(sample, recipe.blend["high_level_task"])["task"] == expected
    # Empty coverage will fail in the dataset loader; it suffices to inspect launch conditioning here.
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {k: config["dataset"][k] for k in ("repo_id", "revision")},
                "segments": [],
            }
        )
    )
    rich, _ = module["prepare_run"](tmp_path, 1, 1, True, path)
    assert rich["dataset"]["steering_skip_uncovered"] is False
    partial, _ = module["prepare_run"](tmp_path, 1, 1, True, path, skip_uncovered=True)
    assert partial["dataset"]["steering_skip_uncovered"] is True
    with pytest.raises(ValueError, match="requires --steering-manifest"):
        module["prepare_run"](tmp_path, 1, 1, True, skip_uncovered=True)
    assert set(rich["dataset"]["steering_required_styles"]) == {
        "subtask",
        "motion",
        "point",
        "trace",
        "combination",
    }
    assert (
        task_from_recipe(sample, TrainingRecipe.from_dict(rich["dataset"]["task_recipe"]))["task"] == expected
    )


def test_grounding_camera_configuration_requires_explicit_observed_views():
    with pytest.raises(ValueError, match="explicit trained"):
        PlannerConfig(styles=["point"])
    with pytest.raises(ValueError, match="included"):
        PlannerConfig(camera_keys=["base"], grounding_camera_keys=["left_wrist"])


def test_planner_observes_all_views_but_limits_coordinate_commands(monkeypatch):
    config = PlannerConfig(
        camera_keys=["base", "left_wrist"], grounding_camera_keys=["base"], styles=["point", "combination"]
    )
    planner = VisionLanguagePlanner(config)
    observation = {key: np.zeros((48, 64, 3), dtype=np.uint8) for key in config.camera_keys}
    decision = {
        "command": "move the object at the first point to the second point",
        "style": "point",
        "camera": "base",
        "points": [[10, 20], [30, 40]],
        "point_mode": "targets",
        "assessment": "both targets visible",
        "status": "continue",
    }
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return Mock(
            json=lambda: {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    assert "[10, 20], [30, 40]" in planner(observation, "goal", 1)
    payload = requests[0]
    assert sum(c["type"] == "input_image" for c in payload["input"][0]["content"]) == 2
    assert payload["text"]["format"]["schema"]["properties"]["camera"]["enum"] == ["base", None]
    decision["camera"] = "left_wrist"
    with pytest.raises(ValueError, match="without trained coordinate"):
        planner(observation, "goal", 1)
    decision.update(camera="base", style="combination", point_mode="path")
    with pytest.raises(ValueError, match="trace steering"):
        planner(observation, "goal", 1)
    assert len(planner._history) == 2  # Rejected proposals never become issued-command history.
