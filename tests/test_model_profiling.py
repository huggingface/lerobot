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
                "inference": {"median_ms": 12.5},
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
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(run).encode()))
    (tmp_path / "results.json").write_text(json.dumps([{"policy": "act", "status": "failed"}]))
    (tmp_path / "README.md").write_text("Measured results")
    (tmp_path / "unrelated.txt").write_text("Do not publish")
    folder = ci.archive(tmp_path)
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["key"] == "123-2"
    assert metadata["conclusion"] == "failure"
    assert metadata["created_at"] == run["created_at"]
    assert metadata["run_started_at"] == run["run_started_at"]
    with zipfile.ZipFile(folder / "report.zip") as report:
        assert set(report.namelist()) == {"results.json", "README.md"}


def test_runner_propagates_failure_and_writes_row(monkeypatch, tmp_path):
    monkeypatch.setattr(ci.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 2))
    assert ci.run_policy("act", tmp_path, []) == 2
    row = json.loads((tmp_path / "act/result.json").read_text())
    assert row["status"] == "failed"


@pytest.fixture
def benchmark():
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from benchmarks import profile_policies

    return profile_policies


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
