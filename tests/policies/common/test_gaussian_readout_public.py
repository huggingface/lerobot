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
