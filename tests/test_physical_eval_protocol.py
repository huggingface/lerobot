# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def protocol(tmp_path):
    api = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/physical_eval.py"))
    scenarios = [
        {
            "id": str(i),
            "task": "fixture task",
            "reset_instructions": "fixture reset",
            "success_rubric": "fixture rubric",
            "tags": ["paraphrase", "distractor", "novel_object", "multi_step"],
        }
        for i in range(17)
    ]
    plan = api["plan_attempts"](scenarios, 42)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return api, path, tmp_path / "events.jsonl", plan


def run_metadata(tmp_path, attempt):
    return {
        "checkpoint_sha256": "a" * 64,
        "rollout_config_sha256": "b" * 64,
        "reset_operator": "test-operator",
        "initial_state_evidence": f"test-only/{attempt}/initial.png",
        "dataset_root": str(tmp_path / attempt / "dataset"),
        "planner_log": str(tmp_path / attempt / "planner.jsonl"),
    }


def test_schedule_is_balanced_and_preserves_matched_resets(protocol):
    api, _, _, plan = protocol
    assert len(plan["attempts"]) == 50
    assert Counter(a["condition"] for a in plan["attempts"]) == {"task_only": 17, "semantic": 16, "full": 17}
    for i in range(16):
        group = plan["attempts"][i * 3 : i * 3 + 3]
        assert {a["scenario_id"] for a in group} == {str(i)}
        assert {a["condition"] for a in group} == {"task_only", "semantic", "full"}
    assert plan == api["plan_attempts"](plan["scenarios"], 42)


def test_unstarted_schedule_is_not_reported_as_physical_attempts(protocol):
    api, path, journal, _ = protocol
    report = api["summarize"](path, journal)
    assert report["planned_attempts"] == 50 and report["started_attempts"] == 0
    assert all(c["success_rate_bounds"] is None for c in report["conditions"].values())


def test_failure_unknown_and_model_only_success_are_never_dropped(protocol, tmp_path):
    api, path, journal, plan = protocol
    append = api["append_event"]
    for index, attempt in enumerate(plan["attempts"][:3]):
        aid = attempt["id"]
        append(path, journal, aid, "begin", run_metadata(tmp_path, aid))
        append(
            path,
            journal,
            aid,
            "model",
            {"outcome": "success", "reviewer": "test-model", "evidence": "fixture"},
        )
        if index < 2:
            append(path, journal, aid, "end", {"reason": "runtime_error" if index == 0 else "completed"})
            append(
                path,
                journal,
                aid,
                "operator",
                {
                    "outcome": "failure" if index == 0 else "success",
                    "reviewer": "test-human",
                    "evidence": "fixture",
                },
            )
    report = api["summarize"](path, journal)
    assert report["started_attempts"] == 3 and report["unstarted_attempts"] == 47
    assert [a["operator_outcome"] for a in report["attempts"]] == ["failure", "success", "unknown"]
    assert all(a["model_outcome"] == "success" for a in report["attempts"])
    unknown = report["conditions"][plan["attempts"][2]["condition"]]
    assert unknown["success_rate_bounds"] == [0, 1] and unknown["ended"] == 0
    assert sum(p["pairs_with_unknown"] for p in report["matched_comparisons"].values()) == 2
    for comparison in report["matched_comparisons"].values():
        low, high = comparison["success_difference_bounds"]
        assert -1 <= low <= high <= 1
        assert high - low == comparison["pairs_with_unknown"]
    with pytest.raises(ValueError, match="outstanding"):
        append(path, journal, plan["attempts"][3]["id"], "begin", run_metadata(tmp_path, "next"))


def test_journal_rejects_replaced_attempts_and_changed_plan(protocol, tmp_path):
    api, path, journal, plan = protocol
    aid = plan["attempts"][0]["id"]
    api["append_event"](path, journal, aid, "begin", run_metadata(tmp_path, aid))
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="already started"):
        api["append_event"](path, journal, aid, "begin", run_metadata(tmp_path, aid))
    assert journal.read_bytes() == before
    plan["scenarios"][0]["task"] = "changed after seeing results"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Plan changed"):
        api["summarize"](path, journal)


def test_invalid_metrics_do_not_modify_journal(protocol, tmp_path):
    api, path, journal, plan = protocol
    aid = plan["attempts"][0]["id"]
    api["append_event"](path, journal, aid, "begin", run_metadata(tmp_path, aid))
    before = journal.read_bytes()
    data = {
        "outcome": "success",
        "reviewer": "test-human",
        "evidence": "fixture",
        "metrics": {"arm_accuracy": {"correct": True, "incorrect": 0, "unknown": 0}},
    }
    with pytest.raises(ValueError, match="nonnegative integers"):
        api["append_event"](path, journal, aid, "operator", data)
    assert journal.read_bytes() == before


def test_checkpoint_changes_and_shared_logs_are_rejected(protocol, tmp_path):
    api, path, journal, plan = protocol
    first = plan["attempts"][0]
    data = run_metadata(tmp_path, first["id"])
    api["append_event"](path, journal, first["id"], "begin", data)
    api["append_event"](path, journal, first["id"], "end", {"reason": "interrupted"})
    with pytest.raises(ValueError, match="separate recording"):
        api["append_event"](path, journal, plan["attempts"][1]["id"], "begin", data)
    second = next(a for a in plan["attempts"][3:] if a["condition"] == first["condition"])
    with pytest.raises(ValueError, match="planned attempt order"):
        api["append_event"](path, journal, second["id"], "begin", run_metadata(tmp_path, second["id"]))
    for attempt in plan["attempts"][1:]:
        if attempt == second:
            break
        api["append_event"](path, journal, attempt["id"], "begin", run_metadata(tmp_path, attempt["id"]))
        api["append_event"](path, journal, attempt["id"], "end", {"reason": "completed"})
    data = run_metadata(tmp_path, second["id"])
    data["checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Checkpoint changed"):
        api["append_event"](path, journal, second["id"], "begin", data)


def test_operator_metric_corrections_are_appended_and_aggregated(protocol, tmp_path):
    api, path, journal, plan = protocol
    aid = plan["attempts"][0]["id"]
    api["append_event"](path, journal, aid, "begin", run_metadata(tmp_path, aid))
    for correct in [1, 2]:
        api["append_event"](
            path,
            journal,
            aid,
            "operator",
            {
                "outcome": "unknown",
                "reviewer": "test-human",
                "evidence": "fixture",
                "metrics": {"arm_accuracy": {"correct": correct, "incorrect": 1, "unknown": 1}},
            },
        )
    assert len(journal.read_text().splitlines()) == 3
    metric = api["summarize"](path, journal)["conditions"][plan["attempts"][0]["condition"]][
        "operator_metrics"
    ]["arm_accuracy"]
    assert metric["correct"] == 2 and metric["unknown"] == 1
    assert metric["known_accuracy"] == pytest.approx(2 / 3)


def test_planner_timing_counts_errors_holds_and_unfinished_calls(protocol, tmp_path):
    api, *_ = protocol
    log = tmp_path / "planner.jsonl"
    events = []
    for i, event in enumerate(["planner_returned", "planner_hold", "planner_error", None]):
        events.append({"event": "planner_request", "request_id": str(i)})
        if event:
            events.append({"event": event, "request_id": str(i), "elapsed_s": i + 1})
    log.write_text("\n".join(json.dumps(e) for e in events))
    report = api["planner_summary"](log)
    assert report["mean_call_seconds"] == 2
    assert report["request_count"] == 4 and report["unfinished_requests"] == 1
    assert report["terminal_events"]["planner_error"] == 1
    assert report["execution_verified"] is False
