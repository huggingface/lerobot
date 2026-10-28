# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import io
import json
import subprocess
import sys
import tomllib
import zipfile
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts.ci import model_profiling as ci


def test_matrix_extras_exist():
    extras = tomllib.loads((ci.ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert ci.policy_matrix()
    assert all(not row["extra"] or row["extra"] in extras for row in ci.policy_matrix())


def test_matrix_discovers_new_policies_and_ignores_other_registries(tmp_path):
    path = tmp_path / "src/lerobot/policies/new/configuration_new.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '@PreTrainedConfig.register_subclass("new")\nclass NewConfig: pass\n'
        '@OptimizerConfig.register_subclass("optimizer")\nclass Optimizer: pass\n'
    )
    assert ci.policy_matrix(tmp_path) == [{"policy": "new", "extra": "new"}]


def test_summary_retains_missing_and_failed_policies(tmp_path):
    (tmp_path / "act").mkdir()
    (tmp_path / "act/result.json").write_text(
        json.dumps(
            {
                "policy": "act",
                "status": "failed",
                "inference": {"status": "success", "median_ms": 12.5},
                "training": {"status": "failed", "error": "out of memory"},
            }
        )
    )
    report = ci.summary(tmp_path, ["act", "diffusion"])
    assert "12.50" in report
    assert "out of memory" in report
    rows = json.loads((tmp_path / "results.json").read_text())
    assert [row["policy"] for row in rows] == ["act", "diffusion"]
    assert rows[1]["status"] == "failed"


def test_unknown_policy_filter_fails(monkeypatch):
    monkeypatch.setenv("POLICIES", "act,typo")
    monkeypatch.setattr(sys, "argv", ["model_profiling.py", "matrix"])
    with pytest.raises(SystemExit, match="2"):
        ci.main()


@pytest.fixture
def measured_policy():
    return {
        "policy": "act",
        "status": "success",
        "device_name": "NVIDIA L4",
        "device_total_memory_bytes": 23659151360,
        "batch_size": 1,
        "parameter_dtype": "torch.float32",
        "mixed_precision": "no",
        "warmup": 3,
        "iterations": 10,
        "dataset": "lerobot/pusht",
        "dataset_revision": None,
        "dataset_episodes": [0],
        "checkpoint": None,
        "checkpoint_revision": None,
        "config_fingerprint": "recorded-configuration",
        "gradient_checkpointing": False,
        "inference": {"status": "success", "median_ms": 10.0},
        "training": {"status": "success", "median_ms": 20.0},
    }


def history_point(row, days=28, attempt=1):
    return {
        "run": {
            "id": 123,
            "key": f"123-{attempt}",
            "run_attempt": attempt,
            "status": "completed",
            "conclusion": "failure",
            # Creation time is stale on retries; use the attempt's start time.
            "created_at": "2026-01-01T01:00:00Z",
            "run_started_at": (datetime(2026, 10, 26, 1, tzinfo=UTC) - timedelta(days=days)).isoformat(),
            "html_url": "https://github.com/huggingface/lerobot/actions/runs/123",
        },
        "result": row,
    }


def test_four_week_comparison_selects_nearest_attempt_per_operation(measured_policy):
    old = deepcopy(measured_policy)
    old["inference"]["median_ms"] = 12.5
    old["training"]["median_ms"] = 16.0
    failed_training = deepcopy(old)
    failed_training["training"] = {"status": "failed", "error": "out of memory"}
    points = [history_point(old, 30), history_point(old, 28), history_point(failed_training, 28, 2)]
    now = datetime(2026, 10, 26, 1, tzinfo=UTC)
    inference = ci.comparison(measured_policy, "inference", points, now)
    training = ci.comparison(measured_policy, "training", points, now)
    assert inference["change_pct"] == pytest.approx(-20.0)
    assert inference["baseline"]["run"] == "123-2"
    assert training["change_pct"] == pytest.approx(25.0)
    assert training["baseline"]["run"] == "123-1"
    assert training["baseline"]["started_at"].startswith("2026-09-28")


@pytest.mark.parametrize("days", [0, 24, 32, 56])
def test_four_week_comparison_does_not_substitute_recent_or_stale_runs(measured_policy, days):
    result = ci.comparison(
        measured_policy,
        "inference",
        [history_point(measured_policy, days)],
        datetime(2026, 10, 26, 1, tzinfo=UTC),
    )
    assert result["change_pct"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("device_name", "NVIDIA L40S"),
        ("device_total_memory_bytes", 48000000000),
        ("batch_size", 2),
        ("parameter_dtype", "torch.bfloat16"),
        ("mixed_precision", "bf16"),
        ("warmup", 1),
        ("iterations", 50),
        ("dataset", "another/dataset"),
        ("dataset_revision", "new-revision"),
        ("dataset_episodes", [1]),
        ("checkpoint", "another/checkpoint"),
        ("checkpoint_revision", "new-checkpoint"),
        ("config_fingerprint", "different-optimizer"),
        ("gradient_checkpointing", True),
    ],
)
def test_four_week_comparison_rejects_different_setups(measured_policy, field, value):
    old = {**measured_policy, field: value}
    result = ci.comparison(
        measured_policy, "training", [history_point(old)], datetime(2026, 10, 26, 1, tzinfo=UTC)
    )
    assert result["baseline"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, None])
def test_invalid_latency_is_never_compared(measured_policy, value):
    measured_policy["inference"]["median_ms"] = value
    assert ci.measured(measured_policy, "inference") is None


def test_summary_slack_and_check_keep_failures_and_dashboard_link(tmp_path, measured_policy, monkeypatch):
    monkeypatch.setenv("PROFILE_RUN_URL", "https://github.com/huggingface/lerobot/actions/runs/456")
    policies = [entry["policy"] for entry in ci.policy_matrix()]
    for policy in policies:
        folder = tmp_path / policy
        folder.mkdir()
        row = {**measured_policy, "policy": policy}
        if policy == "eo1":
            row["status"] = "failed"
            row["training"] = {"status": "failed", "error": "CUDA out of memory"}
        (folder / "result.json").write_text(json.dumps(row))
    old = deepcopy(measured_policy)
    old["inference"]["median_ms"] = 12.5
    old["training"]["median_ms"] = 16.0
    report = ci.summary(
        tmp_path, points=[history_point(old)], started_at=datetime(2026, 10, 26, 1, tzinfo=UTC)
    )
    assert "| act | success | 10.00 | -20.0% | 20.00 | +25.0% |" in report
    assert "| eo1 | failed | 10.00 | — | GPU limit | — |" in report
    payload = json.loads((tmp_path / "slack.json").read_text())
    table = payload["blocks"][2]["text"]["text"]
    assert len(table) < 3000  # Slack section limit; includes all registered policies.
    assert all(policy in table for policy in policies)
    assert "GPU limit" in table
    assert payload["blocks"][4]["elements"][0]["url"] == ci.SPACE_URL
    check = json.loads((tmp_path / "status.json").read_text())
    assert check["target_url"] == ci.SPACE_URL
    assert check["context"] == "Profiling report"
    assert ci.SPACE_URL in report
    compared = json.loads((tmp_path / "comparison.json").read_text())
    assert compared["rows"][0]["inference"]["baseline"]["median_ms"] == 12.5


def test_history_outage_still_writes_current_report(monkeypatch, tmp_path):
    monkeypatch.setenv("POLICIES", "act")
    monkeypatch.setattr(
        sys, "argv", ["model_profiling.py", "summary", "--compare-history", "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(ci, "current_run", lambda: {"run_started_at": "2026-10-26T01:00:00Z"})

    def unavailable(*args, **kwargs):
        raise OSError("Space unavailable")

    monkeypatch.setattr(ci, "urlopen", unavailable)
    assert ci.main() == 0
    assert "History unavailable" in (tmp_path / "README.md").read_text()
    assert json.loads((tmp_path / "comparison.json").read_text())["rows"][0]["inference"]["baseline"] is None


def test_publish_latest_retains_sources_and_compares_measurement_date(monkeypatch, tmp_path, measured_policy):
    old = deepcopy(measured_policy)
    old["inference"]["median_ms"] = 12.5
    row = {**measured_policy, "source_run": history_point(measured_policy, 0)["run"]}
    responses = {
        ci.LATEST_URL: {"rows": [row]},
        ci.HISTORY_URL: {"points": [history_point(old)]},
    }
    monkeypatch.setattr(ci, "urlopen", lambda url, **kwargs: io.BytesIO(json.dumps(responses[url]).encode()))
    monkeypatch.setattr(ci, "current_run", lambda: {"run_started_at": "2026-11-23T01:00:00Z"})
    monkeypatch.setenv("POLICIES", "act")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_profiling.py",
            "summary",
            "--compare-history",
            "--publish-latest=true",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert ci.main() == 0
    assert json.loads((tmp_path / "results.json").read_text()) == [row]
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["rows"][0]["inference"]["change_pct"] == pytest.approx(-20.0)
    assert "multiple source runs" in (tmp_path / "README.md").read_text()


def test_unrecorded_configuration_cannot_be_a_baseline(measured_policy):
    old = deepcopy(measured_policy)
    del old["config_fingerprint"]
    assert (
        ci.comparison(
            measured_policy, "inference", [history_point(old)], datetime(2026, 10, 26, 1, tzinfo=UTC)
        )["baseline"]
        is None
    )


def test_different_recorded_cpu_thread_counts_are_not_compared(measured_policy):
    old = deepcopy(measured_policy)
    old["cpu_threads"] = 16
    now = datetime(2026, 10, 26, 1, tzinfo=UTC)
    assert ci.comparison(measured_policy, "inference", [history_point(old)], now)["baseline"] is None
    measured_policy["cpu_threads"] = 16
    assert ci.comparison(measured_policy, "inference", [history_point(old)], now)["baseline"] is not None


def test_different_sample_indices_are_not_compared(measured_policy):
    old = deepcopy(measured_policy)
    old["sample_indices"] = [1]
    now = datetime(2026, 10, 26, 1, tzinfo=UTC)
    assert ci.comparison(measured_policy, "training", [history_point(old)], now)["baseline"] is None
    measured_policy["sample_indices"] = [1]
    assert ci.comparison(measured_policy, "training", [history_point(old)], now)["baseline"] is not None


def test_archive_preserves_run_attempt_and_only_public_report_files(monkeypatch, tmp_path):
    run = {
        "id": 123,
        "head_sha": "abc",
        "head_branch": "main",
        "created_at": "2026-07-21T12:00:00Z",
        "run_started_at": "2026-10-21T12:00:00Z",
        "html_url": "https://github.com/huggingface/lerobot/actions/runs/123",
    }
    for key, value in {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "huggingface/lerobot",
        "GITHUB_TOKEN": "test",
        "GRADIENT_CHECKPOINTING": "true",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(run).encode()))
    (tmp_path / "results.json").write_text(json.dumps([{"policy": "act", "status": "failed"}]))
    (tmp_path / "README.md").write_text("Measured results")
    (tmp_path / "comparison.json").write_text("[]")
    (tmp_path / "slack.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("Do not publish")
    folder = ci.archive(tmp_path)
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["key"] == "123-2"
    assert metadata["conclusion"] == "failure"
    assert metadata["profiling_variant"] == "checkpointing_trial"
    assert metadata["created_at"] == run["created_at"]
    assert metadata["run_started_at"] == run["run_started_at"]
    with zipfile.ZipFile(folder / "report.zip") as report:
        assert set(report.namelist()) == {"results.json", "README.md", "comparison.json", "slack.json"}


def test_runner_propagates_failure_and_writes_row(monkeypatch, tmp_path):
    monkeypatch.setattr(ci.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 2))
    assert ci.run_policy("act", tmp_path, []) == 2
    row = json.loads((tmp_path / "act/result.json").read_text())
    assert row["status"] == "failed"


def test_timeout_keeps_completed_inference_and_marks_training(tmp_path):
    run = tmp_path / "lingbot_va" / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "policy": "lingbot_va",
                "status": "failed",
                "inference_context": "cached_episode_prompt",
                "inference": {"status": "success", "median_ms": 12.5},
            }
        )
    )
    ci.mark_timeout("lingbot_va", tmp_path, "both")
    row = json.loads((tmp_path / "lingbot_va/result.json").read_text())
    assert row["inference"]["median_ms"] == 12.5
    assert row["training"]["status"] == "failed"
    assert ci.metric_label(row, "training", {"median_ms": None}) == "Time limit"
    ci.summary(tmp_path, ["lingbot_va"])
    payload = json.loads((tmp_path / "slack.json").read_text())
    assert "LingBot inference reuses its episode prompt" in json.dumps(payload)


def test_timeout_during_setup_marks_both_operations(tmp_path):
    ci.mark_timeout("act", tmp_path, "both")
    row = json.loads((tmp_path / "act/result.json").read_text())
    assert row["policy"] == "act"
    assert all("Time limit" in row[mode]["error"] for mode in ("inference", "training"))


def test_timeout_in_inference_only_does_not_report_training_timeout(tmp_path):
    ci.mark_timeout("act", tmp_path, "inference")
    row = json.loads((tmp_path / "act/result.json").read_text())
    assert row["profile_mode"] == "inference"
    assert "training" not in row
    assert ci.metric_label(row, "training", {"median_ms": None}) == "—"


def test_inference_comparison_excludes_different_prompt_context(measured_policy):
    row = deepcopy(measured_policy)
    row["inference_context"] = "cached_episode_prompt"
    old = history_point(measured_policy)
    now = datetime(2026, 10, 26, 1, tzinfo=UTC)
    assert ci.comparison(row, "inference", [old], now)["baseline"] is None
    assert ci.comparison(row, "training", [old], now)["baseline"] is not None


@pytest.fixture
def benchmark():
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from benchmarks import profile_policies

    return profile_policies


def test_flux3_profiles_after_the_padded_action_history(benchmark):
    cfg = SimpleNamespace(type="flux3", conditioning="history", n_obs_steps=1)
    assert benchmark.profile_sample_indices(cfg, 1) == [1]
    cfg.conditioning = "frame"
    assert benchmark.profile_sample_indices(cfg, 1) == [0]
    assert benchmark.profile_sample_indices(SimpleNamespace(type="act"), 2) == [0, 1]


def test_lingbot_inference_resets_chunk_but_reuses_episode_prompt(benchmark):
    class Policy:
        config = SimpleNamespace(type="lingbot_va")

        def __init__(self):
            self.encodes = 0
            self.chunks = 0
            self._prompt = self._prompt_embeds = self._negative_prompt_embeds = None
            self.queue = []

        def reset(self):
            self._prompt = self._prompt_embeds = self._negative_prompt_embeds = None
            self.queue.clear()

        def select_action(self):
            if self._prompt_embeds is None:
                self.encodes += 1
                self._prompt = "task"
                self._prompt_embeds = object()
                self._negative_prompt_embeds = object()
            if not self.queue:
                self.chunks += 1
                self.queue.extend([1, 2])
            return self.queue.pop(0)

    policy = Policy()
    for _ in range(3):
        benchmark.reset_for_inference(policy)
        assert policy.select_action() == 1
    assert policy.encodes == 1
    assert policy.chunks == 3


def test_matrix_matches_runtime_registry(benchmark):
    from lerobot.configs import PreTrainedConfig

    assert {row["policy"] for row in ci.policy_matrix()} == set(PreTrainedConfig.get_known_choices())


@pytest.mark.parametrize("policy", [row["policy"] for row in ci.policy_matrix()])
def test_every_recipe_parses_and_validates(benchmark, policy, tmp_path):
    import draccus

    cfg = draccus.parse(
        benchmark.ProfileConfig, args=[*ci.policy_args(policy), f"--output_dir={tmp_path / policy}"]
    )
    cfg.validate()
    assert cfg.policy.type == policy


@pytest.mark.parametrize(
    ("policy", "flag"), [("pi05", "gradient_checkpointing"), ("fastwam", "use_gradient_checkpointing")]
)
def test_checkpointing_variant_preserves_optimizer_defaults(benchmark, policy, flag):
    import draccus

    original = draccus.parse(benchmark.ProfileConfig, args=ci.policy_args(policy))
    variant = draccus.parse(benchmark.ProfileConfig, args=ci.policy_args(policy, gradient_checkpointing=True))
    assert not getattr(original.policy, flag)
    assert getattr(variant.policy, flag)
    assert original.policy.get_optimizer_preset() == variant.policy.get_optimizer_preset()


@pytest.mark.parametrize("policy", ["flux3", "lawam", "fastwam", "lingbot_va"])
def test_world_model_recipes_accept_dataset_features(benchmark, policy, tmp_path):
    import draccus

    from lerobot.configs import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features

    cfg = draccus.parse(
        benchmark.ProfileConfig, args=[*ci.policy_args(policy), f"--output_dir={tmp_path / policy}"]
    )
    cameras = (
        ["observation.image"]
        if policy == "flux3"
        else ["observation.images.front", "observation.images.wrist"]
    )
    features = {
        name: {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]}
        for name in cameras
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [2 if policy == "flux3" else 8],
                "names": None,
            },
            "action": {"dtype": "float32", "shape": [2 if policy == "flux3" else 7], "names": None},
        }
    )
    parsed = dataset_to_policy_features(features)
    cfg.policy.output_features = {k: v for k, v in parsed.items() if v.type is FeatureType.ACTION}
    cfg.policy.input_features = {k: v for k, v in parsed.items() if v.type is not FeatureType.ACTION}
    if hasattr(cfg.policy, "set_dataset_feature_metadata"):
        cfg.policy.set_dataset_feature_metadata(features)
    cfg.policy.validate_features()
    if hasattr(cfg.policy, "obs_cam_keys"):
        assert set(cfg.policy.obs_cam_keys) <= set(features)


@pytest.mark.parametrize("policy", ["dm05", "pi05", "molmoact2"])
def test_recipes_normalize_dataset_without_quantile_stats(benchmark, policy):
    import draccus
    import torch
    from datasets import Dataset

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor import NormalizerProcessorStep

    cfg = draccus.parse(benchmark.ProfileConfig, args=ci.policy_args(policy))
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    cfg.policy.input_features = {"observation.state": features["observation.state"]}
    cfg.policy.output_features = {"action": features["action"]}
    original_stats = {key: {"mean": torch.ones(2), "std": torch.full((2,), 2.0)} for key in features}
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats=original_stats),
        hf_dataset=Dataset.from_dict(
            {key: [[float(i), float(i * 2)] for i in range(100)] for key in features}
        ),
    )
    stats = benchmark.profiling_stats(cfg, SimpleNamespace(config=cfg.policy), dataset)
    assert cfg.policy.normalization_mapping["STATE"] == "QUANTILES"
    assert all("q01" in stats[key] and "q99" in stats[key] for key in features)
    assert all("q01" not in original_stats[key] for key in features)
    assert -1e-6 <= stats["action"]["q01"][0] < 3
    assert 95 < stats["action"]["q99"][0] <= 100
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map=cfg.policy.normalization_mapping,
        stats=stats,
    )
    result = normalizer(
        {
            TransitionKey.OBSERVATION: {"observation.state": torch.full((1, 2), 2.0)},
            TransitionKey.ACTION: torch.full((1, 2), 2.0),
        }
    )
    assert torch.isfinite(result[TransitionKey.OBSERVATION]["observation.state"]).all()
    assert torch.isfinite(result[TransitionKey.ACTION]).all()


def test_flux_preparation_uses_command_deltas_not_raw_action_stats(benchmark, monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    import draccus
    import numpy as np
    from datasets import Dataset

    from lerobot.datasets import lerobot_dataset

    cfg = draccus.parse(benchmark.ProfileConfig, args=ci.policy_args("flux3"))
    dataset = SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(total_episodes=1, episodes={"tasks": [["push"]]}),
        hf_dataset=Dataset.from_dict(
            {
                "episode_index": [0, 0, 0],
                "action": [[10.0, 20.0], [13.0, 24.0], [15.0, 25.0]],
                "observation.state": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            }
        ),
    )
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", lambda *args, **kwargs: dataset)
    benchmark.prepare_policy_config(cfg, dataset)
    np.testing.assert_allclose(cfg.policy.normalization_stats["action"]["q01"], [0.04, 0.02])
    np.testing.assert_allclose(cfg.policy.normalization_stats["state"]["q01"], [1.02, 2.02])


def test_current_observation_preserves_singleton_video_channels(benchmark):
    import torch

    batch = {
        "observation.images.front": torch.randn(1, 3, 16, 16),
        "observation.state": torch.randn(1, 1, 8),
    }
    cfg = SimpleNamespace(observation_delta_indices=[0])
    result = benchmark.current_observation(batch, cfg)
    assert torch.equal(result["observation.images.front"], batch["observation.images.front"])
    assert torch.equal(result["observation.state"], batch["observation.state"][:, 0])


def test_autoregressive_checkpoint_uses_exact_local_snapshot(benchmark, monkeypatch, tmp_path):
    from pathlib import Path

    weights = tmp_path / "pinned-snapshot" / "model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"")
    cfg = SimpleNamespace(
        policy=SimpleNamespace(
            type="pi0_fast", pretrained_path=Path("lerobot/pi0fast-base"), pretrained_revision="revision"
        )
    )
    calls = []

    def download(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(weights)

    monkeypatch.setattr(benchmark, "hf_hub_download", download)
    benchmark.prepare_policy_config(cfg, None)
    assert calls == [("lerobot/pi0fast-base", "model.safetensors", "revision")]
    assert cfg.policy.pretrained_path == weights.parent


def test_unavailable_checkpoint_fails_before_policy_allocation(benchmark, monkeypatch):
    from pathlib import Path

    cfg = SimpleNamespace(
        policy=SimpleNamespace(
            type="pi0_fast", pretrained_path=Path("lerobot/pi0fast-base"), pretrained_revision="revision"
        )
    )

    def unavailable(*args, **kwargs):
        raise PermissionError("checkpoint unavailable")

    monkeypatch.setattr(benchmark, "hf_hub_download", unavailable)
    with pytest.raises(PermissionError, match="checkpoint unavailable"):
        benchmark.prepare_policy_config(cfg, None)


def test_xvla_recipe_loads_required_backbone_architecture(benchmark, monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    import draccus

    architecture = tmp_path / "config.json"
    architecture.write_text(json.dumps({"florence_config": {"vision_config": {}, "text_config": {}}}))
    monkeypatch.setattr(benchmark, "hf_hub_download", lambda *args, **kwargs: str(architecture))
    cfg = draccus.parse(benchmark.ProfileConfig, args=ci.policy_args("xvla"))
    benchmark.prepare_policy_config(cfg, None)
    assert cfg.policy.get_florence_config().vision_config is not None


def test_measure_excludes_warmup_and_preserves_batch(benchmark, monkeypatch):
    import torch

    clock = iter([0, 0.01, 1, 1.02])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    original = {"value": torch.tensor([1.0])}
    seen = []

    def step(batch):
        seen.append(batch["value"].item())
        batch["value"].add_(1)

    result = benchmark.measure(step, original, torch.device("cpu"), warmup=2, iterations=2)
    assert seen == [1.0] * 4
    assert result["samples_ms"] == pytest.approx([10, 20])
    assert result["median_ms"] == pytest.approx(15)
    assert result["p95_ms"] == pytest.approx(20)


def test_trace_is_separate_from_latency_samples(benchmark, tmp_path):
    import torch

    calls = []

    def step(batch):
        calls.append(1)
        return batch["value"].square()

    trace = tmp_path / "trace.json"
    result = benchmark.measure(step, {"value": torch.ones(4)}, torch.device("cpu"), 1, 2, trace)
    assert len(calls) == 4
    assert len(result["samples_ms"]) == 2
    assert json.loads(trace.read_text())["traceEvents"]


def test_current_observation_selects_present_not_last_future_frame(benchmark):
    import torch

    batch = {
        "observation.images.front": torch.arange(4).reshape(1, 4, 1, 1, 1),
        "observation.state": torch.arange(3).reshape(1, 3, 1),
        "action": torch.ones(1, 2),
    }
    cfg = SimpleNamespace(observation_delta_indices=[-1, 0, 1], image_observation_delta_indices=[0, 1, 2, 3])
    observation = benchmark.current_observation(batch, cfg)
    assert observation["observation.images.front"].item() == 0
    assert observation["observation.state"].item() == 1
    assert "action" not in observation


def test_real_act_inference_and_training_write_results(benchmark, monkeypatch, tmp_path):
    import torch

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.default import DatasetConfig
    from lerobot.policies import ACTConfig

    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,)),
    }
    policy_cfg = ACTConfig(
        device="cpu",
        push_to_hub=False,
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        chunk_size=2,
        n_action_steps=2,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        optimizer_fused=False,
        dropout=0.0,
    )

    class Dataset(list):
        num_episodes = 1
        meta = SimpleNamespace(
            camera_keys=[],
            features={
                key: {"dtype": "float32", "shape": feature.shape, "names": None}
                for key, feature in {**features, **policy_cfg.output_features}.items()
            },
            stats={
                key: {"mean": torch.zeros(2), "std": torch.ones(2)} for key in ("observation.state", "action")
            },
        )

    dataset = Dataset(
        [
            {
                "observation.state": torch.randn(2),
                "observation.environment_state": torch.randn(3),
                "action": torch.randn(2, 2),
                "action_is_pad": torch.zeros(2, dtype=torch.bool),
                "episode_index": torch.tensor(0),
            }
        ]
        * 2
    )
    monkeypatch.setattr(benchmark, "make_dataset", lambda cfg: dataset)
    captured = {}
    real_make_policy = benchmark.make_policy

    def make_policy(*args, **kwargs):
        policy = real_make_policy(*args, **kwargs)
        captured["policy"] = policy
        captured["before"] = {k: v.clone() for k, v in policy.state_dict().items()}
        return policy

    monkeypatch.setattr(benchmark, "make_policy", make_policy)
    cfg = benchmark.ProfileConfig(
        dataset=DatasetConfig(repo_id="test/fixed"),
        policy=policy_cfg,
        output_dir=tmp_path / "act",
        batch_size=1,
        warmup=1,
        iterations=2,
    )
    benchmark.main(cfg)
    row = json.loads((cfg.output_dir / "result.json").read_text())
    assert row["status"] == "success"
    assert len(row["config_fingerprint"]) == 64
    assert row["configuration"]["policy"]["type"] == "act"
    assert all(len(row[mode]["samples_ms"]) == 2 for mode in ("inference", "training"))
    assert any(not torch.equal(captured["before"][k], v) for k, v in captured["policy"].state_dict().items())


def test_real_gaussian_actor_sac_update(benchmark, tmp_path):
    import torch

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.default import DatasetConfig
    from lerobot.policies import GaussianActorConfig

    policy_cfg = GaussianActorConfig(
        device="cpu",
        push_to_hub=False,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (2,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    cfg = benchmark.ProfileConfig(
        dataset=DatasetConfig(repo_id="test/fixed"),
        policy=policy_cfg,
        output_dir=tmp_path / "sac",
        batch_size=1,
    )
    cfg.validate()
    metadata = SimpleNamespace(
        features={
            key: {"dtype": "float32", "shape": feature.shape, "names": None}
            for key, feature in {**policy_cfg.input_features, **policy_cfg.output_features}.items()
        },
        stats={},
    )
    policy = benchmark.make_policy(policy_cfg, ds_meta=metadata)
    dataset = [
        {
            "observation.state": torch.randn(2),
            "action": torch.randn(2).tanh(),
            "next.reward": torch.tensor(0.5),
            "episode_index": torch.tensor(0),
        }
    ] * 2

    def preprocess(batch):
        # Real processors also return metadata that SAC must not treat as tensors.
        return {**batch, "task": ["push the block"], "processed": True}

    batch = preprocess(benchmark.default_collate(dataset[:1]))
    before = policy.actor.mean_layer.weight.detach().clone()
    step = benchmark.make_training_step(cfg, policy, dataset, preprocess, None)
    result = step(batch)
    assert "loss_actor" in result.losses
    assert torch.isfinite(torch.tensor(result.losses["loss_critic"]))
    assert not torch.equal(before, policy.actor.mean_layer.weight)
