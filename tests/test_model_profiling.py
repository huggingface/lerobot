#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest
import torch
from huggingface_hub.errors import HfHubHTTPError

from lerobot.utils import model_profiling as mp

# ---------------------------------------------------------------------------
# Policy spec matrix
# ---------------------------------------------------------------------------


def test_policy_specs_cover_expected_policies():
    assert set(mp.POLICY_SPECS) == {
        "act",
        "diffusion",
        "groot",
        "multi_task_dit",
        "pi0",
        "pi0_fast",
        "pi05",
        "smolvla",
        "wall_x",
        "xvla",
    }
    # Sanity: excluded policies should stay out of the matrix.
    for excluded in ("sac", "sarm", "tdmpc", "vqbet", "reward_classifier"):
        assert excluded not in mp.POLICY_SPECS


def test_pretrained_libero_specs_match_expected_camera_keys_and_normalization():
    base_rgb_rename = (
        '--rename_map={"observation.images.front": "observation.images.base_0_rgb", '
        '"observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
    )
    for name in ("pi0", "pi0_fast", "pi05"):
        assert base_rgb_rename in mp.POLICY_SPECS[name]["train_args"]
    assert any(
        arg.startswith('--policy.normalization_mapping={"ACTION": "MEAN_STD"')
        for arg in mp.POLICY_SPECS["pi05"]["train_args"]
    )
    assert (
        '--rename_map={"observation.images.front": "observation.images.camera1", '
        '"observation.images.wrist": "observation.images.camera2"}'
        in mp.POLICY_SPECS["smolvla"]["train_args"]
    )


# ---------------------------------------------------------------------------
# CI orchestrator helpers
# ---------------------------------------------------------------------------


def test_build_train_command_includes_profiling_outputs(tmp_path):
    cmd = mp.build_train_command("act", tmp_path / "run", "trace")
    assert cmd[:3] == ["uv", "run", "lerobot-train"]
    assert any(a.startswith("--output_dir=") for a in cmd)
    assert any(a.startswith("--profile_output_dir=") for a in cmd)
    assert "--profile_mode=trace" in cmd
    assert "--eval_freq=0" in cmd


def test_build_artifact_index_collects_tables_and_traces(tmp_path):
    run_dir = tmp_path / "act" / "20260415T000000Z__act"
    profiling = run_dir / "profiling"
    (profiling / "torch_tables").mkdir(parents=True)
    (profiling / "torch_traces").mkdir(parents=True)
    (profiling / "step_timing_summary.json").write_text("{}")
    (profiling / "deterministic_forward.json").write_text(
        json.dumps({"operator_fingerprint": "ops", "output_fingerprint": "out"})
    )
    (profiling / "torch_tables" / "cpu_time_total.txt").write_text("cpu table")
    (profiling / "torch_traces" / "trace_step_9.json").write_text("{}")
    (run_dir / "stdout.txt").write_text("stdout")
    (run_dir / "stderr.txt").write_text("stderr")

    paths, urls, targets, row_in_repo = mp.build_artifact_index(
        repo_id="lerobot/model-profiling-history",
        run_dir=run_dir,
        policy_name="act",
        run_id="20260415T000000Z__act",
    )

    assert row_in_repo == "rows/act/20260415T000000Z__act.json"
    assert paths["stdout"].endswith("/stdout.txt")
    assert paths["step_timing_summary"].endswith("/profiling/step_timing_summary.json")
    assert "cpu_time_total.txt" in paths["torch_tables"]
    assert "trace_step_9.json" in paths["trace_files"]
    assert urls["row"].startswith("https://huggingface.co/datasets/lerobot/model-profiling-history/")
    # stdout + stderr + 4 profiling files
    assert len(targets) == 6


def test_upload_targets_batches_preview_publish_into_single_hf_pr(monkeypatch, tmp_path):
    local_path = tmp_path / "profiling_row.json"
    local_path.write_text("{}")
    captured: dict[str, object] = {}

    class _FakeCommit:
        pr_url = "https://huggingface.co/datasets/lerobot/model-profiling-history/discussions/42"

    class _FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def create_commit(self, **kwargs):
            captured.update(kwargs)
            return _FakeCommit()

    monkeypatch.setattr(mp, "HfApi", _FakeApi)

    result = mp.upload_targets(
        repo_id="lerobot/model-profiling-history",
        targets=[mp.UploadTarget(local_path, "rows/act/run.json")],
        create_pr=True,
        token="hf_test_token",
    )

    assert captured["repo_id"] == "lerobot/model-profiling-history"
    assert captured["repo_type"] == "dataset"
    assert captured["create_pr"] is True
    assert result.pr_url == _FakeCommit.pr_url
    assert result.uploaded_paths["rows/act/run.json"].endswith("/resolve/refs/pr/42/rows/act/run.json")


def test_parse_discussion_num_handles_hf_discussion_urls():
    assert (
        mp.parse_discussion_num(
            "https://huggingface.co/datasets/lerobot/model-profiling-history/discussions/42"
        )
        == 42
    )
    assert mp.parse_discussion_num("https://huggingface.co/datasets/lerobot/model-profiling-history") is None
    assert mp.parse_discussion_num(None) is None


# ---------------------------------------------------------------------------
# main() smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_args(tmp_path):
    """Shared argparse namespace for main() smoke tests — overridden per-test."""
    return argparse.Namespace(
        policies=["act"],
        output_dir=tmp_path / "results",
        hub_org="lerobot",
        results_repo="model-profiling-history",
        publish=False,
        profile_mode="summary",
        git_commit="",
        git_ref="codex/model-profiling",
        pr_number="3389",
    )


def _stub_train_subprocess(mp_module, *, returncode: int = 0, write_artifacts: bool = True):
    """Build a fake subprocess.run that writes the profiling artifacts main() expects."""

    def _fake_run(cmd, capture_output, text):
        assert capture_output is True
        assert text is True
        profile_dir = Path(next(a.split("=", 1)[1] for a in cmd if a.startswith("--profile_output_dir=")))
        profile_dir.mkdir(parents=True, exist_ok=True)
        if write_artifacts:
            (profile_dir / "torch_tables").mkdir(parents=True, exist_ok=True)
            (profile_dir / "step_timing_summary.json").write_text(
                json.dumps({"total_update_s": {"count": 1, "mean": 0.3}, "peak_memory_allocated_bytes": 1024})
            )
            (profile_dir / "deterministic_forward.json").write_text(
                json.dumps(
                    {"operator_fingerprint": "ops-fingerprint", "output_fingerprint": "output-fingerprint"}
                )
            )
            (profile_dir / "torch_tables" / "cpu_time_total.txt").write_text("cpu time table")
        return subprocess.CompletedProcess(cmd, returncode, "stdout ok", "")

    return _fake_run


def test_main_smoke_writes_row(monkeypatch, fake_args):
    monkeypatch.setattr(mp, "parse_args", lambda: fake_args)
    monkeypatch.setattr(mp.subprocess, "check_output", lambda *a, **k: "deadbeef\n")
    monkeypatch.setattr(mp.subprocess, "run", _stub_train_subprocess(mp))

    assert mp.main() == 0

    row_paths = list(fake_args.output_dir.rglob("profiling_row.json"))
    assert len(row_paths) == 1
    row = json.loads(row_paths[0].read_text())
    assert row["policy"] == "act"
    assert row["status"] == "success"
    assert row["git_commit"] == "deadbeef"
    assert row["git_ref"] == "codex/model-profiling"
    assert row["pr_number"] == 3389
    assert row["step_timing_summary"]["total_update_s"]["mean"] == 0.3
    assert row["deterministic_forward"]["operator_fingerprint"] == "ops-fingerprint"


def test_main_records_publish_failure_without_failing(monkeypatch, fake_args):
    fake_args.publish = True
    fake_args.git_commit = "deadbeef"
    monkeypatch.setattr(mp, "parse_args", lambda: fake_args)
    monkeypatch.setattr(mp.subprocess, "run", _stub_train_subprocess(mp, write_artifacts=False))

    def _fail_upload(**kwargs):
        resp = type("Resp", (), {"status_code": 403, "headers": {}, "request": None})()
        raise HfHubHTTPError("403 Forbidden: Authorization error.", response=resp)

    monkeypatch.setattr(mp, "upload_profile_run", _fail_upload)

    assert mp.main() == 0
    row = json.loads(next(fake_args.output_dir.rglob("profiling_row.json")).read_text())
    assert row["status"] == "success"
    assert row["publish_status"] == "failed"
    assert "Authorization error" in row["publish_error"]


def test_main_returns_nonzero_when_training_subprocess_fails(monkeypatch, fake_args):
    monkeypatch.setattr(mp, "parse_args", lambda: fake_args)
    monkeypatch.setattr(mp.subprocess, "check_output", lambda *a, **k: "deadbeef\n")
    monkeypatch.setattr(mp.subprocess, "run", _stub_train_subprocess(mp, returncode=3))

    assert mp.main() == 1

    row = json.loads(next(fake_args.output_dir.rglob("profiling_row.json")).read_text())
    assert row["status"] == "failed"
    assert row["return_code"] == 3


# ---------------------------------------------------------------------------
# TrainingProfiler behavior
# ---------------------------------------------------------------------------


def test_deterministic_forward_artifacts_preserve_policy_mode(tmp_path):
    class _TrainingOnlyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, batch):
            self.forward_calls += 1
            assert self.training
            return batch["value"].sum(), {"value": batch["value"]}

    dataset = [{"value": torch.tensor([1.0, 2.0])}]
    policy = _TrainingOnlyPolicy()
    policy.train()

    mp.write_deterministic_forward_artifacts(
        policy=policy,
        dataset=dataset,
        batch_size=2,
        preprocessor=lambda b: b,
        output_dir=tmp_path,
        device_type="cpu",
    )

    payload = json.loads((tmp_path / "deterministic_forward.json").read_text())
    assert policy.training is True
    assert policy.forward_calls == 1
    assert payload["reference_batch_size"] == 2
    assert "operator_fingerprint" in payload
    assert payload["outputs"]["loss"]["numel"] == 1


def test_deterministic_forward_artifacts_infers_image_keys_without_dataset_meta(tmp_path):
    class _ImagePolicy(torch.nn.Module):
        def forward(self, batch):
            image = batch["observation.images.front"]
            assert image.dtype == torch.float32
            assert torch.all((image >= 0.0) & (image <= 1.0))
            return image.sum(), {"image": image}

    dataset = [{"observation.images.front": torch.tensor([[[0, 255]]], dtype=torch.uint8)}]

    mp.write_deterministic_forward_artifacts(
        policy=_ImagePolicy(),
        dataset=dataset,
        batch_size=1,
        preprocessor=lambda b: b,
        output_dir=tmp_path,
        device_type="cpu",
    )

    payload = json.loads((tmp_path / "deterministic_forward.json").read_text())
    assert payload["outputs"]["loss"]["numel"] == 1
    assert payload["outputs"]["output_dict"]["image"]["dtype"] == "torch.float32"


def test_training_profiler_section_records_forward_backward_optimizer(tmp_path):
    profiler = mp.TrainingProfiler(mode="summary", output_dir=tmp_path, device=torch.device("cpu"))
    profiler.start()
    for _ in range(3):
        with profiler.section("forward"):
            pass
        with profiler.section("backward"):
            pass
        with profiler.section("optimizer"):
            pass
    profiler.step(1, argparse.Namespace(update_s=0.5, dataloading_s=0.01))
    profiler.finalize()

    payload = json.loads((tmp_path / "step_timing_summary.json").read_text())
    assert payload["forward_s"]["count"] == 3
    assert payload["backward_s"]["count"] == 3
    assert payload["optimizer_s"]["count"] == 3
    assert payload["total_update_s"]["mean"] == 0.5


def test_training_profiler_accepts_metric_like_values(tmp_path):
    class _MetricLike:
        def __init__(self, v):
            self.val = v

    profiler = mp.TrainingProfiler(mode="summary", output_dir=tmp_path, device=torch.device("cpu"))
    profiler.start()
    profiler.step(1, argparse.Namespace(update_s=_MetricLike(0.6), dataloading_s=_MetricLike(0.05)))
    profiler.finalize()

    payload = json.loads((tmp_path / "step_timing_summary.json").read_text())
    assert payload["total_update_s"]["mean"] == 0.6
    assert payload["dataloading_s"]["mean"] == 0.05


def test_profiler_device_time_uses_generic_attr_first():
    class _Event:
        self_device_time_total = 12.3456

    assert mp._get_profiler_device_time_us(_Event()) == 12.3456
