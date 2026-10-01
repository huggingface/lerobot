# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""SO-101 contract, trainable parameter, EMA and command-integration regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("datasets")

from examples.flux3 import export_base as exporter
from lerobot.configs.default import DatasetConfig, EMAConfig, PeftConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import factory
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3 import packing
from lerobot.policies.flux3.f3.video_vae import VideoVAE
from lerobot.policies.flux3.processor_flux3 import PAST_ACTIONS, ObservationHistoryNormalizerProcessorStep
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.scripts.lerobot_train import make_dataloaders, train
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.flux3.helpers import TaskVideoVAE, rich_task_config, single_config, task_config


def reference_range(x, config, stream, *, inverse=False):
    stats = config.normalization_stats[stream]
    lo, hi = (x.new_tensor(stats[q]) for q in ("q01", "q99"))
    span = torch.where(hi - lo > 1e-6, hi - lo, torch.ones_like(lo))
    if inverse:
        return (x + 1) * span / 2 + lo
    return (2 * (x - lo) / span - 1).clamp(-config.normalization_clip, config.normalization_clip)


def reference_training_actions(commands, config):
    delta = commands[:, 1:] - commands[:, :-1]
    delta[..., -1] = commands[:, 1:, -1]
    normalized = reference_range(delta, config, "action")
    past = torch.cat([torch.zeros_like(normalized[:, :1]), normalized[:, : config.n_obs_steps - 1]], 1)
    return normalized[:, config.n_obs_steps - 1 :], past


def test_native_task_vae_disables_droid_chunking():
    def encode(video, *, chunked_encode):
        assert chunked_encode is False
        return video

    vae = VideoVAE(SimpleNamespace(encode=encode))
    for frames in (1, 32):
        video = torch.zeros(1, 3, frames, 32, 32)
        assert torch.equal(vae.encode_task(video), video)


@pytest.fixture(autouse=True)
def task_vae(monkeypatch, fake_text_encoder):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, cfg: TaskVideoVAE())


def training_batch(cfg, b=1):
    return {
        "observation.images.top": torch.rand(b, cfg.window_frames, 3, 64, 96),
        OBS_STATE: torch.rand(b, cfg.n_obs_steps, cfg.action_dim),
        ACTION: torch.rand(b, cfg.window_frames, cfg.action_dim),
        "task": ["pick up box"] * b,
    }


def test_peft_defaults_match_shared_example():
    defaults = Flux3Config()
    preset = json.loads((Path(__file__).parents[3] / "examples/flux3/lora.json").read_text())["policy"]
    explicit = {"path", "device", "push_to_hub"}
    for key, value in preset.items():
        if key not in explicit:
            actual = getattr(defaults, key)
            assert (list(actual) if isinstance(actual, tuple) else actual) == value, key
    assert defaults.delta_absolute_dims == []
    assert defaults.action_channel_weights is None
    assert defaults.camera_keys is None


@pytest.mark.parametrize("rich", [False, True])
def test_history_config_and_indices(rich):
    cfg = rich_task_config() if rich else task_config()
    history = 8 if rich else 1
    assert cfg.window_frames == 32 + history
    assert cfg.image_observation_delta_indices == list(range(1 - history, 33))
    assert cfg.state_observation_delta_indices == list(range(1 - history, 1))
    assert cfg.action_delta_indices == list(range(-history, 32))
    policy = Flux3Policy(cfg)
    assert policy.dit.in_channels["action_cond"] == (12 if rich else 6)
    pre, _ = make_flux3_pre_post_processors(cfg)
    prepared = pre(training_batch(cfg))
    assert (PAST_ACTIONS in prepared) == rich
    assert prepared[OBS_STATE].shape == (1, history, 6)
    assert torch.isfinite(policy(prepared)[0])
    raw = json.loads((Path(__file__).parents[3] / "examples/flux3/lora.json").read_text())
    assert raw["steps"] % raw["accelerator"]["gradient_accumulation"]["steps"] == 0
    with pytest.raises(ValueError, match="state-relative"):
        task_config(use_relative_actions=True)
    with pytest.raises(ValueError, match="quantiles"):
        make_flux3_pre_post_processors(task_config(normalization_stats=None))


def test_command_delta_alignment_and_no_target_leak():
    cfg = rich_task_config()
    commands = torch.arange(40).float()[None, :, None].repeat(1, 1, 6)
    commands[..., -1] = 0.7
    targets, past = reference_training_actions(commands, cfg)
    assert targets.shape == (1, 32, 6) and past.shape == (1, 8, 6)
    assert torch.equal(past[:, 0], torch.zeros(1, 6))
    assert torch.allclose(targets[..., :-1], torch.full((1, 32, 5), 0.5))
    assert torch.allclose(targets[..., -1], torch.full((1, 32), 0.35))
    changed = commands.clone()
    changed[:, 8:] += 10
    _, past_changed = reference_training_actions(changed, cfg)
    assert torch.equal(past, past_changed)
    value = torch.tensor([[[0.2] * 6]])
    assert torch.allclose(
        reference_range(reference_range(value, cfg, "action"), cfg, "action", inverse=True), value
    )


def test_history_encoding_and_positions():
    task_config()
    videos = torch.zeros(1, 3, 40, 64, 96)
    videos[:, :, 7] = 1
    result = packing.pack_history_video(TaskVideoVAE(), videos, 8, 2, 24, (2, 3))
    assert result["x_video_cond"].shape == (1, 12, 96)
    assert result["x_video"].shape == (1, 48, 96)
    assert result["x_video_cond_ids"][0, :, 0].unique().tolist() == [0, 29]
    assert result["x_video_ids"][0, 0, 0] == 33
    states = torch.ones(1, 8, 6)
    act = packing.pack_history_actions(
        states, torch.zeros_like(states), torch.ones(1, 32, 6), "action", 30, 2
    )
    assert torch.equal(act["x_action_cond"][..., 6:], states)  # state is deliberately unscaled
    assert act["x_action_ids"][0, 0, 0] == 0
    assert act["x_action_cond_ids"][0, -1, 0] == 0
    assert torch.all(act["x_action_cond_ids"][..., -1] == -1)


def test_loss_weight_is_rms_normalized():
    pred = {"x_video": torch.ones(1, 3, 96), "x_action": torch.ones(1, 32, 6)}
    targets = {k: torch.zeros_like(v) for k, v in pred.items()}
    losses = packing.flow_loss(
        pred, targets, "action", 0.5, 1.0, reduction="modalities", channel_weights=[1, 1, 1, 1, 1, 2]
    )
    assert torch.allclose(losses["loss"], torch.tensor(1.5))
    pred["x_action"][..., :-1] = 0
    losses = packing.flow_loss(
        pred, targets, "action", 0.5, 1.0, reduction="modalities", channel_weights=[1, 1, 1, 1, 1, 2]
    )
    assert torch.allclose(losses["loss"], torch.tensor(1 + 0.5 * 4 / 9))


@pytest.mark.parametrize("rich", [False, True])
def test_command_queue_keeps_anchor_and_records_every_tick(monkeypatch, rich):
    cfg = rich_task_config() if rich else task_config()
    policy = Flux3Policy(cfg)
    pre, post = make_flux3_pre_post_processors(cfg)
    seen = []

    def predict(batch, **kwargs):
        seen.append(batch)
        deltas = torch.ones(1, cfg.chunk_size, 6)
        deltas[..., -1] = 0.7
        return reference_range(deltas, cfg, "action")

    monkeypatch.setattr(policy, "predict_action_chunk", predict)
    batch = {"observation.images.top": torch.rand(1, 3, 64, 96), OBS_STATE: torch.zeros(1, 6)}
    for i in range(cfg.n_action_steps + 1):
        # Measured state stays behind: never re-anchor queued deltas to it.
        result = post(policy.select_action(pre(batch)))
        assert torch.allclose(result[:, :-1], torch.full((1, 5), float(i + 1)))
        assert torch.allclose(result[:, -1], torch.tensor([0.7]))
    assert len(seen) == 2
    if rich:
        assert seen[-1][PAST_ACTIONS][0, -1, 0] == 0.5
    else:
        assert all(PAST_ACTIONS not in batch for batch in seen)
        assert all(batch[OBS_STATE].shape == (1, 1, 6) for batch in seen)
    policy.reset()
    pre.reset()
    post.reset()
    assert post(policy.select_action(pre(batch)))[0, 0] == 1


@pytest.mark.parametrize("pipeline_source", ["fresh", "saved"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
        ),
    ],
)
def test_history_processors_keep_device_across_ticks(tmp_path, pipeline_source, device):
    cfg = rich_task_config(device=device)
    if pipeline_source == "fresh":
        pre, post = make_flux3_pre_post_processors(cfg)
    else:
        pre, post = make_flux3_pre_post_processors(rich_task_config())
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
        saved_pre = (tmp_path / "policy_preprocessor.json").read_bytes()
        pre, post = make_pre_post_processors(
            cfg,
            pretrained_path=tmp_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        assert (tmp_path / "policy_preprocessor.json").read_bytes() == saved_pre

    normalizer = next(
        step for step in pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)
    )
    raw = {"observation.images.top": torch.zeros(1, 3, 64, 96), OBS_STATE: torch.full((1, 6), 2.0)}
    for tick in range(3):
        batch = pre(raw)
        assert all(value.device.type == device for value in batch.values() if isinstance(value, torch.Tensor))
        if tick:
            torch.testing.assert_close(batch[PAST_ACTIONS][:, -1], torch.full((1, 6), 0.25, device=device))
        command = post(torch.full((1, 6), 0.25, device=device))
        assert command.device.type == "cpu"
        torch.testing.assert_close(command[:, :-1], torch.full((1, 5), 2.0 + 0.5 * (tick + 1)))
        torch.testing.assert_close(command[:, -1], torch.tensor([0.5]))

    assert all(value.device.type == device for value in normalizer.commands)
    assert all(value.device.type == device for obs in normalizer.observations for value in obs.values())
    assert normalizer.last_command.device.type == device
    transfer = next(step for step in pre.steps if isinstance(step, DeviceProcessorStep))
    assert pre.steps.index(transfer) < pre.steps.index(normalizer)
    assert raw[OBS_STATE].device.type == "cpu"


def test_offline_inference_history_required():
    cfg = rich_task_config()
    policy = Flux3Policy(cfg)
    batch = training_batch(cfg)
    batch["observation.images.top"] = batch["observation.images.top"][:, :8]
    pre, _ = make_flux3_pre_post_processors(cfg)
    commands = batch.pop(ACTION)
    with pytest.raises(ValueError, match="command_history"):
        pre(batch)
    batch["observation.command_history"] = commands[:, :8]
    assert policy.predict_action_chunk(pre(batch)).shape == (1, 32, 6)


def test_separate_timesteps_and_conditioning_noise():
    video = {
        "x_video": torch.ones(2, 3, 96),
        "x_video_ids": torch.zeros(2, 3, 4),
        "x_video_cond": torch.ones(2, 6, 96),
        "x_video_cond_ids": torch.zeros(2, 6, 4),
    }
    action = {
        "x_action": torch.ones(2, 32, 6),
        "x_action_ids": torch.zeros(2, 32, 4),
        "x_action_cond": torch.ones(2, 8, 12),
        "x_action_cond_ids": torch.zeros(2, 8, 4),
    }
    kwargs, _ = packing.build_forward_kwargs(
        video,
        action,
        {},
        torch.tensor([0.2, 0.3]),
        "action",
        action_timesteps=torch.tensor([0.8, 0.9]),
        conditioning_noise_max=0.2,
    )
    assert torch.allclose(kwargs["x_video_timesteps"][:, 0], torch.tensor([0.2, 0.3]))
    assert torch.allclose(kwargs["x_action_timesteps"][:, 0], torch.tensor([0.8, 0.9]))
    assert torch.equal(kwargs["x_action_cond"], action["x_action_cond"])
    assert kwargs["x_video_cond_timesteps"].max() <= 0.2
    assert not torch.equal(kwargs["x_video_cond"], video["x_video_cond"])


@pytest.mark.parametrize("use_defaults", [False, True])
def test_exporter_creates_cli_config_without_training(tmp_path, monkeypatch, use_defaults):
    cfg = task_config()
    features = {**cfg.input_features, **cfg.output_features}
    monkeypatch.setattr(
        exporter, "LeRobotDatasetMetadata", lambda *a, **kw: SimpleNamespace(fps=30, features={}, stats={})
    )
    monkeypatch.setattr(exporter, "dataset_to_policy_features", lambda _: features)
    monkeypatch.setattr(Flux3Policy, "forward", lambda *a, **kw: pytest.fail("export must not train"))
    # A resolved training config must still load the explicitly requested trunk.
    trunk_state = {k: torch.full_like(v, 0.125) for k, v in Flux3Policy(cfg).dit.state_dict().items()}
    trunk = tmp_path / "trunk.safetensors"
    save_file(trunk_state, str(trunk))
    cfg.pretrained_path = "previous/base"
    cfg.pretrained_revision = "previous-revision"
    recipe = json.loads((Path(__file__).parents[3] / "examples/flux3/lora.json").read_text())
    recipe["policy"] = draccus.encode(cfg)
    if use_defaults:
        for key in ("fps", "action_representation", "delta_absolute_dims", "normalization_stats"):
            recipe["policy"].pop(key)

        def compute_statistics(*args, action_representation, absolute_dims, **kwargs):
            assert action_representation == "delta"
            assert absolute_dims == []
            return cfg.normalization_stats, [0]

        monkeypatch.setattr(exporter, "compute_statistics", compute_statistics)
    source = tmp_path / "recipe.json"
    source.write_text(json.dumps(recipe))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(cfg.normalization_stats))
    output = exporter.export_base(
        source, "test/so101", None if use_defaults else stats, str(trunk), tmp_path / "base"
    )
    restored = Flux3Policy.from_pretrained(output)
    assert restored.config.pretrained_revision is None
    assert all(torch.equal(restored.dit.state_dict()[key], value) for key, value in trunk_state.items())
    assert restored.config.normalization_stats is None
    assert restored.config.conditioning == "history"
    assert restored.config.fps == 30 and restored.config.action_representation == "delta"
    assert restored.config.delta_absolute_dims == ([] if use_defaults else [-1])

    pre, post = make_pre_post_processors(restored.config, pretrained_path=output)
    normalizer = next(
        step for step in pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)
    )
    assert torch.equal(
        normalizer.state_dict()["state.q01"], torch.tensor(cfg.normalization_stats["state"]["q01"])
    )
    raw = json.loads((output / "so101_train.json").read_text())
    train = draccus.decode(TrainPipelineConfig, raw)
    assert train.policy.pretrained_path == str(output.resolve()) or str(train.policy.pretrained_path) == str(
        output.resolve()
    )
    assert not train.policy.use_peft
    assert train.peft.r == 32 and train.ema.enable
    assert (output / "policy_preprocessor.json").exists()
    with pytest.raises(FileExistsError):
        exporter.export_base(source, "test/so101", stats, None, output)


def test_actual_trainer_peft_ema_checkpoint_and_resume(tmp_path, monkeypatch):
    for package in ("accelerate", "peft", "diffusers"):
        pytest.importorskip(package)

    # Restore the global precision setting changed by train() at teardown.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)

    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/so101",
        fps=30,
        root=root,
        features={
            "action": {"dtype": "float32", "shape": (6,), "names": None},
            OBS_STATE: {"dtype": "float32", "shape": (6,), "names": None},
            "observation.images.top": {
                "dtype": "image",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channel"],
            },
        },
    )
    for t in range(66):
        dataset.add_frame(
            {
                ACTION: np.full(6, t / 100, dtype=np.float32),
                OBS_STATE: np.zeros(6, dtype=np.float32),
                "observation.images.top": np.full((64, 96, 3), t, dtype=np.uint8),
                "task": "pick box",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    cfg = task_config(push_to_hub=False)
    base = tmp_path / "base"
    pre, post = make_flux3_pre_post_processors(cfg, dataset.meta.stats)
    policy = Flux3Policy(cfg)
    cfg.normalization_stats = None
    policy.save_pretrained(base)
    pre.save_pretrained(base)
    post.save_pretrained(base)

    def config(policy_config, steps):
        result = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="test/so101", root=str(root)),
            policy=policy_config,
            output_dir=tmp_path / "train",
            batch_size=2,
            num_workers=0,
            steps=steps,
            save_freq=4,
            env_eval_freq=0,
            log_freq=0,
            seed=42,
            peft=PeftConfig(r=32, lora_alpha=32),
            ema=EMAConfig(enable=True, decay=0.999),
        )
        result.accelerator.gradient_accumulation.steps = 2
        result.optimizer = policy_config.get_optimizer_preset()
        result.scheduler = policy_config.get_scheduler_preset()
        result.validate = lambda: None
        return result

    cfg.pretrained_path = str(base)

    loader_config = config(cfg, 4)
    loader_config.max_eval_samples = 3
    loaded_dataset = LeRobotDataset("test/so101", root=root)
    _, eval_loader = make_dataloaders(
        loader_config, loaded_dataset, loaded_dataset, 0, SimpleNamespace(device_type="cpu", dp_world_size=1)
    )
    assert len(eval_loader.dataset) == 3
    assert [int(eval_loader.dataset[i]["frame_index"]) for i in range(3)] == [1, 2, 3]
    train(config(cfg, 4))
    checkpoint = tmp_path / "train/checkpoints/000004"
    train_config = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    assert train_config["policy"]["use_peft"] is True
    saved = torch.load(checkpoint / "training_state/ema_state.pt", weights_only=True)
    assert saved["optimization_step"] == 2
    assert (checkpoint / "pretrained_model_ema/config.json").exists()
    assert (checkpoint / "pretrained_model_ema/adapter_model.safetensors").exists()
    resume_policy = Flux3Config.from_pretrained(checkpoint / "pretrained_model")
    resume_policy.pretrained_path = str(checkpoint / "pretrained_model")
    resume = config(resume_policy, 12)
    resume.resume = True
    resume.checkpoint_path = checkpoint
    train(resume)
    saved = torch.load(tmp_path / "train/checkpoints/000012/training_state/ema_state.pt", weights_only=True)
    assert saved["optimization_step"] == 6
    # Compare against an uninterrupted run with the same seed/data order.

    uninterrupted_policy = Flux3Config.from_pretrained(base)
    uninterrupted_policy.pretrained_path = str(base)
    uninterrupted = config(uninterrupted_policy, 12)
    uninterrupted.output_dir = tmp_path / "uninterrupted"
    train(uninterrupted)
    for variant in ("pretrained_model", "pretrained_model_ema"):
        resumed_weights = load_file(
            tmp_path / "train/checkpoints/000012" / variant / "adapter_model.safetensors"
        )
        continuous_weights = load_file(
            tmp_path / "uninterrupted/checkpoints/000012" / variant / "adapter_model.safetensors"
        )
        assert all(torch.equal(value, continuous_weights[key]) for key, value in resumed_weights.items())


def test_normalization_uses_training_episodes_only(monkeypatch):
    metadata = SimpleNamespace(total_episodes=4, episodes={"tasks": [["blue"], ["blue"], ["red"], ["red"]]})
    selected = []
    rows = []
    for episode in (0, 2):
        for value in (0.0, 1.0, 2.0):
            rows.append({"episode_index": episode, "action": [value] * 6, OBS_STATE: [value] * 6})

    def dataset(repo, *, root, episodes, download_videos):
        selected.extend(episodes)
        return SimpleNamespace(hf_dataset=SimpleNamespace(select_columns=lambda _: rows))

    monkeypatch.setattr(exporter, "LeRobotDataset", dataset)
    stats, train = exporter.compute_statistics("test/so101", metadata, {"eval_split": 0.5})
    assert selected == train == [0, 2]
    # Reset the delta at each episode boundary, and leave gripper absolute.
    assert stats["action"]["q99"][:5] == [1.0] * 5
    assert stats["action"]["q99"][-1] == 2.0
    assert stats["state"]["q99"] == [2.0] * 6


def test_fixed_text_padding_is_part_of_policy_contract():
    cfg = task_config(text_fixed_length=160)
    policy = Flux3Policy(cfg)
    assert policy._context("pick box", torch.device("cpu"))[0].shape[1] == 160
    assert policy._context("", torch.device("cpu"))[0].shape[1] == 160


def test_custom_processor_training_matches_reference_and_preserves_stats(tmp_path):
    legacy = rich_task_config()
    cfg = rich_task_config()
    old_policy = Flux3Policy(legacy)
    pre, post = make_flux3_pre_post_processors(cfg)
    raw = training_batch(cfg)
    raw[ACTION][:, :2, 0] = 100  # Exercise clipping and consecutive deltas.
    prepared = pre(raw)
    targets, past = reference_training_actions(raw[ACTION], legacy)
    assert torch.equal(prepared[ACTION], targets)
    assert torch.equal(prepared[PAST_ACTIONS], past)
    assert torch.equal(prepared[OBS_STATE], reference_range(raw[OBS_STATE], legacy, "state"))
    with pytest.raises(ValueError, match="already been processed"):
        pre(prepared)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    cfg.normalization_stats = None
    conflicting = {"q01": torch.full((6,), -1000.0), "q99": torch.full((6,), 1000.0)}
    loaded_pre, loaded_post = make_pre_post_processors(
        cfg,
        pretrained_path=tmp_path,
        preprocessor_overrides={
            "normalizer_processor": {"stats": {OBS_STATE: conflicting, ACTION: conflicting}}
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": {ACTION: conflicting}}},
    )
    assert torch.equal(loaded_pre(raw)[ACTION], targets)
    assert torch.equal(loaded_pre(raw)[OBS_STATE], prepared[OBS_STATE])
    assert len(list(tmp_path.glob("*.safetensors"))) == 3
    # Tiny real policies receive exactly the same training tensors and random draws.
    old_policy.save_pretrained(tmp_path)
    new_policy = Flux3Policy.from_pretrained(tmp_path, config=cfg).train()
    torch.manual_seed(9)
    old_loss, _ = old_policy(prepared)
    torch.manual_seed(9)
    new_loss, _ = new_policy(loaded_pre(raw))
    assert torch.equal(old_loss, new_loss)
    with pytest.raises(ValueError, match="preprocessor"):
        new_policy(raw)
    # A processor-format config cannot silently initialize quantiles from a dataset.
    with pytest.raises(ValueError, match="dataset stats"):
        make_flux3_pre_post_processors(cfg, {ACTION: conflicting})
    config_path = tmp_path / "policy_preprocessor.json"
    saved = json.loads(config_path.read_text())
    custom = next(
        step for step in saved["steps"] if step["registry_name"] == "flux3_observation_history_normalizer"
    )
    state_path = tmp_path / custom.pop("state_file")
    state_path.unlink()
    config_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="missing saved quantiles"):
        make_pre_post_processors(cfg, pretrained_path=tmp_path)


def test_custom_processors_preserve_robot_commands_history_and_reload(tmp_path, monkeypatch):
    legacy = rich_task_config()
    cfg = rich_task_config()
    new_policy = Flux3Policy(cfg)
    pre, post = make_flux3_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    cfg.normalization_stats = None
    pre, post = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    new_seen = []

    def predict(batch, seen):
        seen.append(batch)
        delta = torch.ones(1, cfg.chunk_size, 6)
        delta[..., -1] = 0.7
        return reference_range(delta, legacy, "action")

    monkeypatch.setattr(new_policy, "predict_action_chunk", lambda b, **kw: predict(b, new_seen))
    raw = {"observation.images.top": torch.rand(1, 3, 64, 96), OBS_STATE: torch.zeros(1, 6)}
    for tick in range(2 * cfg.n_action_steps + 2):  # three chunk predictions
        actual = post(new_policy.select_action(pre(raw)))
        assert torch.equal(actual[:, :-1], torch.full((1, 5), float(tick + 1)))
        assert torch.allclose(actual[:, -1], torch.tensor([0.7]))
    assert len(new_seen) == 3
    assert torch.equal(new_seen[-1][PAST_ACTIONS][:, 1:, :5], torch.full((1, 7, 5), 0.5))
    new_policy.reset()
    pre.reset()
    post.reset()
    assert post(new_policy.select_action(pre(raw)))[0, 0] == 1
    # Offline chunks use an explicit anchor and do not depend on online queue state.
    offline = {
        "observation.images.top": torch.rand(1, 8, 3, 64, 96),
        OBS_STATE: torch.zeros(1, 8, 6),
        "observation.command_history": torch.full((1, 8, 6), 10.0),
    }
    pre.reset()
    post.reset()
    chunk = post(new_policy.predict_action_chunk(pre(offline)))
    assert torch.equal(chunk[0, :, 0], torch.arange(11, 43).float())
    assert torch.allclose(chunk[0, :, -1], torch.full((32,), 0.7))


def test_two_camera_base_supports_scene_only_peft_and_reload(tmp_path, monkeypatch):
    pytest.importorskip("peft")

    scene, wrist = "observation.images.top", "observation.images.front"
    inputs = task_config().input_features
    cfg = task_config(
        camera_layout="side_by_side",
        camera_keys=[scene, wrist],
        input_features={**inputs, wrist: inputs[scene]},
    )
    base = tmp_path / "base"
    policy = Flux3Policy(cfg)
    pre, post = make_flux3_pre_post_processors(cfg)
    cfg.normalization_stats = None
    policy.save_pretrained(base)
    pre.save_pretrained(base)
    post.save_pretrained(base)

    cfg = Flux3Config.from_pretrained(
        base, cli_overrides=["--camera_layout=single", f"--camera_keys={json.dumps([scene])}"]
    )
    cfg.pretrained_path = str(base)
    features = {
        scene: cfg.input_features[scene],
        OBS_STATE: cfg.input_features[OBS_STATE],
        **cfg.output_features,
    }
    monkeypatch.setattr(factory, "dataset_to_policy_features", lambda _: features)
    metadata = SimpleNamespace(features={}, stats={})
    policy = factory.make_policy(cfg, ds_meta=metadata).wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": 2, "lora_alpha": 2}
    )
    pre, post = factory.make_pre_post_processors(cfg, pretrained_path=base)
    normalizer = next(
        step for step in pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)
    )
    assert normalizer.camera_keys == [scene]
    quantiles = normalizer.state_dict()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
    policy.train()
    loss, _ = policy(pre(training_batch(cfg)))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer.step()

    adapter = tmp_path / "adapter"
    policy.save_pretrained(adapter)
    policy.config.save_pretrained(adapter)
    pre.save_pretrained(adapter)
    post.save_pretrained(adapter)
    restored_cfg = Flux3Config.from_pretrained(adapter)
    restored_cfg.pretrained_path = str(adapter)
    restored = factory.make_policy(restored_cfg, ds_meta=metadata)
    restored_pre, restored_post = factory.make_pre_post_processors(restored_cfg, pretrained_path=adapter)
    restored_normalizer = next(
        step for step in restored_pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)
    )
    assert restored_cfg.camera_order == restored_normalizer.camera_keys == [scene]
    assert all(torch.equal(value, restored_normalizer.state_dict()[key]) for key, value in quantiles.items())
    obs = {scene: torch.rand(1, 3, 64, 96), OBS_STATE: torch.rand(1, 6)}
    policy.eval()
    restored.eval()
    pre.reset()
    post.reset()
    assert torch.equal(
        post(policy.select_action(pre(obs))), restored_post(restored.select_action(restored_pre(obs)))
    )
    # Adapting a subset of cameras does not rewrite the original base package.
    base_cfg = Flux3Config.from_pretrained(base)
    base_pre, _ = factory.make_pre_post_processors(base_cfg, pretrained_path=base)
    base_normalizer = next(
        step for step in base_pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)
    )
    assert base_normalizer.camera_keys == base_cfg.camera_order == [scene, wrist]


def test_frame_conditioning_uses_standard_processors():
    cfg = single_config()
    pre, post = make_flux3_pre_post_processors(cfg)
    assert not any(isinstance(step, ObservationHistoryNormalizerProcessorStep) for step in pre.steps)
    assert cfg.normalization_stats is None
    assert not any("History" in type(step).__name__ for step in post.steps)


def test_custom_quantiles_validation_and_constant_channels():
    processor = ObservationHistoryNormalizerProcessorStep(action_dim=6)
    values = {
        f"{stream}.{q}": torch.full((6,), 3.0) for stream in ("state", "action") for q in ("q01", "q99")
    }
    processor.load_state_dict(values)
    assert torch.equal(processor.scale(torch.full((1, 6), 3.5), "state"), torch.zeros(1, 6))
    assert torch.equal(processor.scale(torch.full((1, 6), 100.0), "action"), torch.full((1, 6), 6.0))
    for invalid in (torch.full((6,), float("nan")), torch.zeros(5), torch.full((6,), 2.0)):
        with pytest.raises(ValueError, match="quantiles"):
            processor.load_state_dict({**values, "state.q99": invalid})


@pytest.mark.parametrize(
    "dim,condition_on_past_actions,representation,history",
    [
        (3, False, "absolute", 8),
        (7, True, "absolute", 8),
        (7, False, "delta", 8),
        (7, True, "delta", 8),
        (6, False, "delta", 1),
    ],
)
def test_history_supports_other_dimensions_and_action_representations(
    tmp_path, dim, condition_on_past_actions, representation, history
):
    cfg = task_config(
        n_obs_steps=history,
        history_snapshots=min(2, history),
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))},
        condition_on_past_actions=condition_on_past_actions,
        action_representation=representation,
        delta_absolute_dims=[1, -1] if representation == "delta" else [],
        action_channel_weights=None,
        normalization_stats={name: {"q01": [-2.0] * dim, "q99": [2.0] * dim} for name in ("action", "state")},
    )
    policy = Flux3Policy(cfg)
    pre, post = make_flux3_pre_post_processors(cfg)
    batch = training_batch(cfg)
    prepared = pre(batch)
    assert prepared[ACTION].shape == (1, 32, dim)
    assert (PAST_ACTIONS in prepared) == condition_on_past_actions
    assert policy.dit.in_channels["action_cond"] == dim * (2 if condition_on_past_actions else 1)
    loss, _ = policy(prepared)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())
    # Processors and the expanded/unchanged head shape survive a complete export.
    cfg.normalization_stats = None
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    restored = Flux3Policy.from_pretrained(tmp_path)
    pre, post = make_pre_post_processors(restored.config, pretrained_path=tmp_path)
    obs = {
        "observation.images.top": batch["observation.images.top"][:, :history],
        OBS_STATE: batch[OBS_STATE],
    }
    if condition_on_past_actions or representation == "delta":
        obs["observation.command_history"] = torch.full((1, history, dim), 10.0)
    prepared = pre(obs)
    predicted = restored.predict_action_chunk(prepared)
    assert post(predicted).shape == (1, 32, dim)
    # A normalized value of 0.5 means +1 under these quantiles.
    commands = post(torch.full((1, 32, dim), 0.5))
    if representation == "delta":
        assert torch.equal(commands[0, :, 0], torch.arange(11, 43).float())
        assert torch.equal(commands[0, :, 1], torch.ones(32))
        assert torch.equal(commands[0, :, -1], torch.ones(32))
    else:
        assert torch.equal(commands, torch.ones(1, 32, dim))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"conditioning": "robot"}, "conditioning"),
        ({"condition_on_past_actions": True}, "require history"),
        ({"action_representation": "delta"}, "require history"),
        ({"delta_absolute_dims": [-1]}, "requires action_representation"),
    ],
)
def test_invalid_frame_conditioning_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        single_config(**overrides)


def test_history_processor_rejects_mismatched_absolute_channels(tmp_path):
    cfg = task_config()
    pre, post = make_flux3_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    cfg.delta_absolute_dims = [0]
    with pytest.raises(ValueError, match="settings disagree"):
        make_pre_post_processors(cfg, pretrained_path=tmp_path)
