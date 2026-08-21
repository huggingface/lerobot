# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import JobConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.jobs.kubeflow import _build_trainjob_manifest, submit_to_kubeflow


def _minimal_cfg(**overrides):
    args = [
        "--dataset.repo_id",
        "u/d",
        "--policy.type",
        "act",
        "--job.target",
        "kubeflow",
        "--job.gpus",
        "0",
    ]
    for key, val in overrides.items():
        args.extend([f"--job.{key}", str(val)])
    return draccus.parse(TrainPipelineConfig, args=args)


# -- CPU-only (gpus=0) --------------------------------------------------------


def test_zero_gpus_omits_resource_key():
    assert JobConfig(gpus=0).gpu_resource_key is None


def test_zero_gpus_skips_vendor_resolution(monkeypatch):
    """gpus=0 is CPU-only regardless of gpu_vendor; auto-detection must not even run."""

    def _boom():
        raise AssertionError("detect_gpu_vendor should not be called when gpus == 0")

    monkeypatch.setattr("lerobot.utils.device_utils.detect_gpu_vendor", _boom)
    cfg = JobConfig(gpu_vendor="auto", gpus=0)
    assert cfg.resolved_gpu_vendor == "cpu"
    assert cfg.gpu_resource_key is None
    assert cfg.resolved_runtime == "torch-distributed"


def test_submit_to_kubeflow_cpu_only_does_not_probe_vendor(monkeypatch, capsys):
    """gpus=0 must not trigger gpu_vendor auto-detection anywhere in the submit path.

    Regression test: the status banner used to call resolved_gpu_vendor unconditionally,
    so a CPU-only submission from a machine with no accelerator raised the
    "no local accelerator was detected" error meant for gpus > 0 requests.
    """

    def _boom():
        raise AssertionError("detect_gpu_vendor should not be called for a CPU-only submission")

    monkeypatch.setattr("lerobot.utils.device_utils.detect_gpu_vendor", _boom)
    monkeypatch.setattr(
        "lerobot.jobs.kubeflow._create_trainjob", lambda manifest, kubeconfig=None: "job-name"
    )
    monkeypatch.setattr("lerobot.jobs.kubeflow.follow_job", lambda *args, **kwargs: True)
    cfg = _minimal_cfg()

    submit_to_kubeflow(cfg)

    assert "GPUs/node: 0 (cpu)" in capsys.readouterr().out


# -- Multi-node ----------------------------------------------------------------


def test_manifest_multi_node():
    cfg = _minimal_cfg(nodes="3", namespace="training")
    manifest = _build_trainjob_manifest(cfg, "j", ["cmd"])

    assert manifest["metadata"]["namespace"] == "training"
    assert manifest["spec"]["trainer"]["numNodes"] == 3


# -- kubeconfig -----------------------------------------------------------------


def test_draccus_parses_kubeconfig_field():
    parsed = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--job.target",
            "kubeflow",
            "--job.kubeconfig",
            "/path/to/kube.yaml",
        ],
    )
    assert parsed.job.kubeconfig == "/path/to/kube.yaml"


def test_submit_to_kubeflow_passes_kubeconfig(monkeypatch):
    """cfg.job.kubeconfig must reach both the create and the log/poll-follow calls."""
    calls = {}

    def _fake_create_trainjob(manifest, kubeconfig=None):
        calls["create_kubeconfig"] = kubeconfig
        return "job-name"

    def _fake_follow_job(*args, **kwargs):
        calls["follow_kubeconfig"] = kwargs.get("kubeconfig")
        return True

    monkeypatch.setattr("lerobot.jobs.kubeflow._create_trainjob", _fake_create_trainjob)
    monkeypatch.setattr("lerobot.jobs.kubeflow.follow_job", _fake_follow_job)
    cfg = _minimal_cfg(kubeconfig="/path/to/kube.yaml")

    submit_to_kubeflow(cfg)

    assert calls["create_kubeconfig"] == "/path/to/kube.yaml"
    assert calls["follow_kubeconfig"] == "/path/to/kube.yaml"
