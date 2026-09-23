# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
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


def test_planner_sends_named_images_and_bounded_history_without_action_tools(monkeypatch):
    calls = []
    decision = {
        "command": "reach for tape",
        "camera": None,
        "points": [],
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
    decision.update(status="continue", style="point", camera="base", points=[[32, 24]])
    assert planner(obs, "goal", 2) == "In base view (64x48 pixels), reach for tape: [32, 24]."
    decision["points"] = [[64, 24]]
    with pytest.raises(ValueError, match="outside"):
        planner(obs, "goal", 2)
    planner.config.styles.append("combination")
    decision.update(style="combination", points=[[32, 24], [33, 25]])
    with pytest.raises(ValueError, match="trace steering"):
        planner(obs, "goal", 2)


def test_four_gpu_training_config_uses_main_parser(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, argv = module["prepare_run"](tmp_path, 4, 1, True)
    parsed = draccus.decode(TrainPipelineConfig, config)
    assert parsed.policy.type == "wall_x"
    assert parsed.policy.base_model_revision == "44e827683819957d8c574e8b746a1a97e77f518a"
    assert parsed.policy.recipe["messages"][0]["stream"] == "low_level"
    assert parsed.steps == parsed.eval_steps == parsed.save_freq == 10
    assert "--nproc-per-node=4" in argv
    with pytest.raises(ValueError):
        module["prepare_run"](tmp_path, 5, 1, True)


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
    assert (
        task_from_recipe(sample, TrainingRecipe.from_dict(rich["dataset"]["task_recipe"]))["task"] == expected
    )
