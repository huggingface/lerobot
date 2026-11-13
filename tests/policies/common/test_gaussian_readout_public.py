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

"""Public inference, artifact and fitting contracts on CPU."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.common import gaussian_readout as gar
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from tests.policies.common.test_gaussian_readout import tiny_flow


@pytest.fixture
def readout() -> gar.GaussianReadout:
    return gar.GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=-1,
        close_normalized=1,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )


def test_public_helpers_accept_policy_or_core(tmp_path: Path, readout: gar.GaussianReadout) -> None:
    core = nn.Module()
    core.action_out_proj = nn.Linear(8, 8)
    core.config = SimpleNamespace(chunk_size=4, max_action_dim=8)
    policy = nn.Module()
    policy.model = core
    gar.bind_processor_sha256(policy, "processors")
    assert core._gar_processor_sha256 == "processors"
    assert gar.state_dict_sha256(policy) == gar.state_dict_sha256(core)
    path = tmp_path / "readout.safetensors"
    gar.save_readout(
        readout, path, teacher_sha256=gar.state_dict_sha256(policy), processor_sha256="processors"
    )
    assert isinstance(gar.load_gaussian_readout(path, policy, "processors"), gar.GaussianReadout)


def test_from_pretrained_with_own_processors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readout: gar.GaussianReadout
) -> None:
    pytest.importorskip("transformers")
    core, _ = tiny_flow.__wrapped__(SimpleNamespace(param="pi05"), monkeypatch)

    def make_core(config, **kwargs):
        model = deepcopy(core)
        model.config = config
        return model

    class TinyPolicy(PI05Policy):
        use_native_pretrained_loader = True
        model_class = staticmethod(make_core)

        def _preprocess_images(self, batch):
            return [], []

    class Tokenizer:
        def get_vocab(self):
            return {"tiny": 0}

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text("{}")

        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.zeros(len(text), 2, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
            }

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    config = core.config
    config.input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (8,))}
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    config.normalization_mapping = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
    config.use_relative_actions = False
    config.text_tokenizer_name = "tiny-local-tokenizer"
    policy = TinyPolicy(config)
    policy.save_pretrained(tmp_path)
    stats = {
        key: {"mean": torch.zeros(dim), "std": torch.ones(dim)}
        for key, dim in (("observation.state", 8), ("action", 7))
    }
    pre, post = make_pre_post_processors(policy.config, dataset_stats=stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    path = tmp_path / "readout.safetensors"
    gar.save_readout(
        readout,
        path,
        teacher_sha256=gar.state_dict_sha256(policy),
        processor_sha256=policy.config._gar_processor_sha256,
    )
    loaded = TinyPolicy.from_pretrained(tmp_path, local_files_only=True)
    loaded.config.gaussian_readout_path = str(path)
    pre, post = make_pre_post_processors(loaded.config, pretrained_path=tmp_path)
    batch = pre({"observation.state": torch.zeros(8), "task": "pick"})
    with torch.inference_mode():
        assert post(loaded.predict_action_chunk(batch)).shape == (1, 4, 7)
        loaded.reset()
        assert post(loaded.select_action(batch)).shape == (1, 7)
    from lerobot.scripts.lerobot_fit_gaussian_readout import fit_readout

    loaded.config.gaussian_readout_path = None
    fitted_path = tmp_path / "fitted.safetensors"
    contexts = [{"observation.state": torch.full((8,), float(i)), "task": "pick"} for i in range(5)]
    report = fit_readout(loaded, pre, post, contexts[:3], contexts[3:], fitted_path, draws=2, updates=2)
    assert report["fit_contexts"] == 3 and report["tune_contexts"] == 2
    assert report["tune_motion_mean_mse"] >= 0
    assert report["one_step_tune_motion_mean_mse"] >= 0
    loaded.config.gaussian_readout_path = str(fitted_path)
    with torch.inference_mode():
        assert post(loaded.predict_action_chunk(pre(contexts[0]))).shape == (1, 4, 7)


@pytest.mark.parametrize("dimension", ["horizon", "padding", "features", "raw"])
def test_density_rejects_broadcast_shapes(readout: gar.GaussianReadout, dimension: str) -> None:
    features, raw = torch.ones(1, 4, 8), torch.zeros(1, 4, 8)
    action = readout.sample_action(features, raw)
    if dimension == "horizon":
        action = action[:, :1]
    elif dimension == "padding":
        action = action[..., :7]
    elif dimension == "features":
        features = features[:, :1]
    else:
        raw = raw[:, :1]
    with pytest.raises(ValueError, match="shape"):
        readout.log_prob(features, raw, action)


def test_density_accepts_float32_processor_roundtrip() -> None:
    mean = torch.full((7,), -0.1940586119890213)
    std = torch.full((7,), 0.6129239797592163)
    readout = gar.GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=float(gar.normalized_command(-1, mean[6], std[6])),
        close_normalized=float(gar.normalized_command(1, mean[6], std[6])),
        action_mean=mean,
        action_std=std,
    )
    features, raw = torch.ones(1, 4, 8), torch.zeros(1, 4, 8)
    with torch.no_grad():
        readout.gripper_bias.fill_(-10)
    action = readout.sample_action(features, raw)
    roundtrip = action.clone()
    roundtrip[..., :7] = ((action[..., :7] * std + mean) - mean) / std
    assert not torch.equal(action[..., 6], roundtrip[..., 6])
    assert torch.isfinite(readout.log_prob(features, raw, roundtrip)).all()
    torch.testing.assert_close(
        readout.log_prob(features, raw, action), readout.log_prob(features, raw, roundtrip)
    )


@pytest.mark.parametrize(("motion", "gripper", "active"), [(0, 0, 1), (5, 6, 7), (6, 6, 8)])
def test_rejects_unsupported_action_layouts(motion: int, gripper: int, active: int) -> None:
    with pytest.raises(ValueError, match="layout"):
        gar.GaussianReadout(
            8,
            chunk_size=4,
            motion_dim=motion,
            gripper_index=gripper,
            max_action_dim=8,
            action_dim=active,
            open_normalized=-1,
            close_normalized=1,
            action_mean=torch.zeros(active),
            action_std=torch.ones(active),
        )


def test_loader_checks_active_action_feature(tmp_path: Path, readout: gar.GaussianReadout) -> None:
    teacher = nn.Module()
    teacher.action_out_proj = nn.Linear(8, 8)
    teacher.config = SimpleNamespace(
        chunk_size=4, max_action_dim=8, output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))}
    )
    path = tmp_path / "readout.safetensors"
    gar.save_readout(
        readout, path, teacher_sha256=gar.state_dict_sha256(teacher), processor_sha256="processors"
    )
    with pytest.raises(ValueError, match="active action"):
        gar.load_gaussian_readout(path, teacher, "processors")


def test_standalone_sampling_reports_device_mismatch(readout: gar.GaussianReadout) -> None:
    readout.to("meta")
    with pytest.raises(ValueError, match="device"):
        readout.sample_action(torch.ones(1, 4, 8), torch.zeros(1, 4, 8))


@pytest.mark.parametrize("head", ["variance", "gripper"])
def test_fit_preserves_rng_and_trainability(readout: gar.GaussianReadout, head: str) -> None:
    features = torch.randn(8, 4, 8)
    target = torch.ones(8, 4, 6) if head == "variance" else torch.full((8, 4), 0.5)
    readout.affine_weight.requires_grad_(False)
    flags = [p.requires_grad for p in readout.parameters()]
    readout.train()
    before = torch.random.get_rng_state().clone()
    fit = gar.fit_variance_head if head == "variance" else gar.fit_gripper_head
    fit(readout, features, target, features, target, updates=2)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert [p.requires_grad for p in readout.parameters()] == flags
    assert readout.training


def test_variance_loss_balances_each_coordinate() -> None:
    target = torch.tensor([[[1.0, 10.0]]])
    predicted = 2 * target
    scale = target.mean(dim=(0, 1), keepdim=True).square()
    expected = torch.tensor(1.0) + gar.VARIANCE_LOG_WEIGHT * torch.log(torch.tensor(2.0)).square()
    loss = gar.variance_loss(predicted, target, scale)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("field", ["min_period", "max_period"])
def test_artifact_binds_timestep_periods(tmp_path, readout, field):
    teacher = nn.Module()
    teacher.action_out_proj = nn.Linear(8, 8)
    teacher.config = SimpleNamespace(chunk_size=4, max_action_dim=8, min_period=0.004, max_period=4.0)
    gar.bind_processor_sha256(teacher, "processors")
    path = str(tmp_path / "periods.safetensors")
    gar.save_readout(
        readout, path, teacher_sha256=gar.state_dict_sha256(teacher), processor_sha256="processors"
    )
    gar._cached_readout(teacher, path)
    setattr(teacher.config, field, getattr(teacher.config, field) * 2)
    with pytest.raises(ValueError, match="period"):
        gar.load_gaussian_readout(path, teacher, "processors")
    with pytest.raises(ValueError, match="period"):
        gar._cached_readout(teacher, path)


def test_smolvla_requires_prefix_cache(monkeypatch):
    model, _ = tiny_flow.__wrapped__(SimpleNamespace(param="smolvla"), monkeypatch)
    model.config.use_cache = False
    with pytest.raises(ValueError, match="use_cache"):
        gar.encode_prefix(
            model,
            [],
            [],
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 8),
        )


def test_fineart_feature_helpers_preserve_causal_language_marks(monkeypatch):
    model, backbone = tiny_flow.__wrapped__(SimpleNamespace(param="fineart_vla"), monkeypatch)
    tokens, masks = torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)
    marks = torch.tensor([[False, True, True]])
    seen = []
    forward = backbone.forward

    def capture(**kwargs):
        if kwargs["inputs_embeds"][0] is not None:
            seen.append(kwargs["attention_mask"].clone())
        return forward(**kwargs)

    monkeypatch.setattr(backbone, "forward", capture)
    previous = torch.zeros_like(marks)
    model._lang_causal_marks = previous
    gar.evaluate_time_one_features(
        model, [], [], tokens, masks, None, None, gar.make_epsilon_star(4, 8), lang_causal_marks=marks
    )
    assert model._lang_causal_marks is previous
    gar.encode_prefix(model, [], [], tokens, masks)
    assert not torch.equal(seen[0], seen[1])


def test_save_nondefault_teacher_periods(tmp_path, readout):
    teacher = nn.Module()
    teacher.action_out_proj = nn.Linear(8, 8)
    teacher.config = SimpleNamespace(chunk_size=4, max_action_dim=8, min_period=0.008, max_period=8.0)
    path = tmp_path / "periods.safetensors"
    gar.save_readout(
        readout,
        path,
        teacher_sha256=gar.state_dict_sha256(teacher),
        processor_sha256="processors",
        teacher=teacher,
    )
    assert isinstance(gar.load_gaussian_readout(path, teacher, "processors"), gar.GaussianReadout)


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
@pytest.mark.parametrize("fail", [False, True])
def test_prefix_restores_attention_backend(monkeypatch, policy_type, fail):
    model, backbone = tiny_flow.__wrapped__(SimpleNamespace(param=policy_type), monkeypatch)
    config = backbone.paligemma.model.language_model.config
    config._attn_implementation = "sdpa"
    forward = backbone.forward

    def capture(**kwargs):
        assert config._attn_implementation == "eager"
        if fail:
            raise RuntimeError("prefill failed")
        return forward(**kwargs)

    monkeypatch.setattr(backbone, "forward", capture)
    args = (model, [], [], torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
    if fail:
        with pytest.raises(RuntimeError, match="prefill failed"):
            gar.encode_prefix(*args)
    else:
        gar.encode_prefix(*args)
    assert config._attn_implementation == "sdpa"


@pytest.mark.parametrize("change", ["same", "nan", "seed", "std", "mean", "codes"])
def test_constructor_rejects_invalid_artifact_contract(change):
    kwargs = {
        "feature_dim": 8,
        "chunk_size": 4,
        "motion_dim": 6,
        "gripper_index": 6,
        "max_action_dim": 8,
        "action_dim": 7,
        "open_normalized": -1,
        "close_normalized": 1,
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
    }
    if change == "same":
        kwargs["open_normalized"] = 1
    elif change == "nan":
        kwargs["open_normalized"] = float("nan")
    elif change == "seed":
        kwargs["epsilon_seed"] = 42
    elif change == "std":
        kwargs["action_std"][0] = 0
    elif change == "mean":
        kwargs["action_mean"][0] = float("inf")
    else:
        kwargs["open_normalized"] = -2
    with pytest.raises(ValueError):
        gar.GaussianReadout(**kwargs)


def test_save_rejects_mutated_codes(tmp_path, readout):
    readout.open_normalized.copy_(readout.close_normalized)
    path = tmp_path / "invalid.safetensors"
    with pytest.raises(ValueError):
        gar.save_readout(readout, path, teacher_sha256="teacher", processor_sha256="processors")
    assert not path.exists()


def test_processor_digest_ignores_tokenizer_runtime_padding():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    backend = Tokenizer(WordLevel({"tiny": 0}, unk_token="tiny"))
    step = SimpleNamespace(
        input_tokenizer=SimpleNamespace(backend_tokenizer=backend),
        get_config=lambda: {"padding": "max_length", "max_length": 4},
        state_dict=dict,
    )
    pipeline = SimpleNamespace(steps=[step])
    before = gar.processor_sha256(pipeline)
    backend.enable_padding(length=4)
    backend.enable_truncation(max_length=4)
    assert gar.processor_sha256(pipeline) == before


def test_fitting_cli_rejects_overlapping_episodes(monkeypatch, tmp_path):
    from lerobot.scripts.lerobot_fit_gaussian_readout import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "fit-readout",
            "--policy",
            "unused",
            "--dataset",
            "unused",
            "--fit-episodes",
            "1",
            "--tune-episodes",
            "1",
            "--output",
            str(tmp_path / "readout.safetensors"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "readout.safetensors").exists()


def test_cache_records_head_after_device_placement(monkeypatch, tmp_path, readout):
    from tests.policies.common.test_gaussian_readout import _call_euler, _ReadoutFake

    path = str(tmp_path / "placed.safetensors")
    model = _ReadoutFake(path)
    gar.bind_processor_sha256(model, "processors")
    gar.save_readout(
        readout, path, teacher_sha256=gar.state_dict_sha256(model), processor_sha256="processors"
    )
    placements = []
    original_to = gar.GaussianReadout.to

    def place(head, *args, **kwargs):
        # Model the tensor identity change caused by moving a cold CPU head to a GPU.
        head.affine_weight = nn.Parameter(head.affine_weight.detach().clone())
        placements.append(head)
        return original_to(head, *args, **kwargs)

    monkeypatch.setattr(gar.GaussianReadout, "to", place)
    _call_euler(model)
    cached = model._gar_cache[2]
    _call_euler(model)
    assert model._gar_cache[2] is cached
    assert placements == [cached]
