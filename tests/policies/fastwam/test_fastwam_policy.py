#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import torch
from safetensors.torch import save_model
from torch import nn

from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.modular_fastwam import ActionDiT, MoT
from lerobot.policies.fastwam.wan_video_dit import (
    FastWAMAttentionBlock,
    WanVideoDiT,
    fastwam_masked_attention,
    precompute_freqs_cis,
)
from lerobot.policies.pretrained import PreTrainedPolicy


class FakeFastWAMCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(2, 2)

    def training_loss(self, sample):
        assert sample["video"].ndim == 5
        assert sample["context"].ndim == 3
        return sample["action"].sum() * 0.0 + torch.tensor(1.0), {"loss_action": 1.0}

    def infer_action(self, **kwargs):
        horizon = kwargs["action_horizon"]
        return {"action": torch.ones(horizon, 3)}


def _patch_core_builder(monkeypatch):
    monkeypatch.setattr(
        FastWAMPolicy,
        "_build_core_model",
        lambda self, config: FakeFastWAMCore(),
    )


def test_action_attention_block_supports_mot_attention_dim_larger_than_hidden_dim():
    block = FastWAMAttentionBlock(hidden_dim=16, attn_head_dim=8, num_heads=4, ffn_dim=32)
    x = torch.zeros(1, 2, 16)
    context = torch.zeros(1, 3, 16)
    t_mod = torch.zeros(1, 6, 16)
    freqs = precompute_freqs_cis(8, end=2).view(2, 1, -1)

    output = block(x, context, t_mod, freqs)

    assert output.shape == x.shape
    assert block.self_attn.q.out_features == 32
    assert block.self_attn.o.out_features == 16


def test_fastwam_masked_attention_accepts_rope_float32_qk_with_bfloat16_values():
    q = torch.zeros(1, 2, 32, dtype=torch.float32)
    k = torch.zeros(1, 2, 32, dtype=torch.float32)
    v = torch.zeros(1, 2, 32, dtype=torch.bfloat16)

    out = fastwam_masked_attention(q=q, k=k, v=v, num_heads=4)

    assert out.dtype == torch.float32
    assert out.shape == v.shape


def test_fastwam_masked_attention_runs_fp32_when_cache_promotes_keys():
    q = torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 32, dtype=torch.float32)
    v = torch.zeros(1, 4, 32, dtype=torch.bfloat16)
    mask = torch.ones(2, 4, dtype=torch.bool)

    out = fastwam_masked_attention(q=q, k=k, v=v, num_heads=4, ctx_mask=mask)

    assert out.dtype == torch.float32
    assert out.shape == q.shape


def test_attention_post_projection_casts_fp32_attention_to_block_dtype():
    block = FastWAMAttentionBlock(hidden_dim=16, attn_head_dim=8, num_heads=4, ffn_dim=32).to(
        dtype=torch.bfloat16
    )
    residual = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
    mixed_attn = torch.zeros(1, 2, 32, dtype=torch.float32)
    gate_msa = torch.ones(1, 16, dtype=torch.bfloat16)
    shift_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)
    scale_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)
    gate_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)

    out = MoT._apply_expert_post_block(
        block=block,
        residual_x=residual,
        mixed_attn_out=mixed_attn,
        gate_msa=gate_msa,
        shift_mlp=shift_mlp,
        scale_mlp=scale_mlp,
        gate_mlp=gate_mlp,
        context_payload=None,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == residual.shape


def test_attention_cross_projection_casts_fp32_attention_to_block_dtype():
    block = FastWAMAttentionBlock(hidden_dim=16, attn_head_dim=8, num_heads=4, ffn_dim=32).to(
        dtype=torch.bfloat16
    )
    x = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
    context = torch.zeros(1, 3, 16, dtype=torch.bfloat16)

    out = block.apply_cross_attention(x, context)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_attention_norm3_handles_bfloat16_affine_weights():
    block = FastWAMAttentionBlock(hidden_dim=16, attn_head_dim=8, num_heads=4, ffn_dim=32).to(
        dtype=torch.bfloat16
    )
    x = torch.zeros(1, 2, 16, dtype=torch.bfloat16)

    out = block.apply_norm3(x)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_attention_post_block_handles_bfloat16_cross_attention_norm():
    block = FastWAMAttentionBlock(hidden_dim=16, attn_head_dim=8, num_heads=4, ffn_dim=32).to(
        dtype=torch.bfloat16
    )
    residual = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
    mixed_attn = torch.zeros(1, 2, 32, dtype=torch.float32)
    gate_msa = torch.ones(1, 16, dtype=torch.bfloat16)
    shift_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)
    scale_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)
    gate_mlp = torch.zeros(1, 16, dtype=torch.bfloat16)
    context_payload = {"context": torch.zeros(1, 3, 16, dtype=torch.bfloat16), "mask": None}

    out = MoT._apply_expert_post_block(
        block=block,
        residual_x=residual,
        mixed_attn_out=mixed_attn,
        gate_msa=gate_msa,
        shift_mlp=shift_mlp,
        scale_mlp=scale_mlp,
        gate_mlp=gate_mlp,
        context_payload=context_payload,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == residual.shape


def test_video_dit_pre_dit_casts_double_latents_to_model_dtype():
    model = WanVideoDiT(
        hidden_dim=4,
        in_dim=48,
        ffn_dim=8,
        out_dim=48,
        text_dim=6,
        freq_dim=4,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        attn_head_dim=4,
        num_layers=0,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    ).to(dtype=torch.bfloat16)

    state = model.pre_dit(
        x=torch.zeros(1, 48, 1, 2, 2, dtype=torch.float64),
        timestep=torch.zeros(1, dtype=torch.float64),
        context=torch.zeros(1, 2, 6, dtype=torch.float64),
        fuse_vae_embedding_in_latents=True,
    )

    assert state["tokens"].dtype == torch.bfloat16
    assert state["context"].dtype == torch.bfloat16
    assert state["t_mod"].dtype == torch.bfloat16


def test_action_dit_pre_dit_casts_double_inputs_to_model_dtype():
    model = ActionDiT(
        hidden_dim=16,
        action_dim=3,
        ffn_dim=32,
        text_dim=6,
        freq_dim=4,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=0,
    ).to(dtype=torch.bfloat16)

    state = model.pre_dit(
        action_tokens=torch.zeros(1, 2, 3, dtype=torch.float64),
        timestep=torch.zeros(1, dtype=torch.float64),
        context=torch.zeros(1, 2, 6, dtype=torch.float64),
    )

    assert state["tokens"].dtype == torch.bfloat16
    assert state["context"].dtype == torch.bfloat16
    assert state["t_mod"].dtype == torch.bfloat16


def test_forward_adapts_lerobot_batch_to_fastwam_sample(monkeypatch):
    _patch_core_builder(monkeypatch)
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    policy = FastWAMPolicy(cfg)
    batch = {
        "observation.images.image": torch.zeros(1, 3, 16, 16),
        "observation.state": torch.zeros(1, 2),
        "action": torch.zeros(1, 4, 3),
        "context": torch.zeros(1, 5, 4096),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
    }

    output = policy.forward(batch)

    assert set(output) == {"loss", "loss_action"}
    assert output["loss"].item() == 1.0
    assert output["loss_action"].item() == 1.0


def test_get_optim_params_returns_lerobot_optimizer_dict(monkeypatch):
    _patch_core_builder(monkeypatch)
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    policy = FastWAMPolicy(cfg)

    optim_params = policy.get_optim_params()

    assert isinstance(optim_params, dict)
    assert set(optim_params) == {"params"}
    assert list(optim_params["params"])


def test_select_action_uses_action_queue(monkeypatch):
    _patch_core_builder(monkeypatch)
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    policy = FastWAMPolicy(cfg)
    batch = {
        "input_image": torch.zeros(1, 3, 16, 16),
        "observation.state": torch.zeros(1, 2),
        "context": torch.zeros(1, 5, 4096),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
    }

    first = policy.select_action(batch)
    second = policy.select_action(batch)

    assert first.shape == (1, 3)
    assert second.shape == (1, 3)


def test_predict_action_prepares_lerobot_libero_observation(monkeypatch):
    captured = {}

    class CapturingCore(FakeFastWAMCore):
        def infer_action(self, **kwargs):
            captured.update(kwargs)
            return {"action": torch.ones(1, 4, 3)}

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: CapturingCore())
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        image_size=(16, 32),
        input_features={
            "observation.images.image": {"type": "VISUAL", "shape": (3, 16, 32)},
            "observation.state": {"type": "STATE", "shape": (2,)},
        },
    )
    policy = FastWAMPolicy(cfg)
    batch = {
        "observation.images.image": torch.ones(1, 3, 20, 20),
        "observation.images.image2": torch.zeros(1, 3, 20, 20),
        "observation.state": torch.zeros(1, 2),
        "task": ["pick up the bowl"],
    }

    action = policy.predict_action_chunk(batch)

    assert action.shape == (1, 4, 3)
    assert captured["prompt"] == [cfg.prompt_template.format(task="pick up the bowl")]
    assert tuple(captured["input_image"].shape) == (1, 3, 16, 32)
    assert captured["input_image"].amin().item() == -1.0
    assert captured["input_image"].amax().item() == 1.0
    assert "num_video_frames" not in captured


def test_predict_action_splits_parallel_eval_batch_into_single_infer_calls(monkeypatch):
    captured = []

    class CapturingCore(FakeFastWAMCore):
        def infer_action(self, **kwargs):
            captured.append(
                {
                    "input_image_shape": tuple(kwargs["input_image"].shape),
                    "input_image_sum": float(kwargs["input_image"].sum()),
                    "proprio_shape": tuple(kwargs["proprio"].shape),
                    "proprio_sum": float(kwargs["proprio"].sum()),
                    "prompt": kwargs["prompt"],
                }
            )
            action = torch.full((1, kwargs["action_horizon"], 3), float(len(captured)))
            return {"action": action}

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: CapturingCore())
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        image_size=(16, 16),
        input_features={
            "observation.images.image": {"type": "VISUAL", "shape": (3, 16, 16)},
            "observation.state": {"type": "STATE", "shape": (2,)},
        },
    )
    policy = FastWAMPolicy(cfg)
    batch = {
        "observation.images.image": torch.stack(
            [
                torch.zeros(3, 16, 16),
                torch.ones(3, 16, 16),
                torch.full((3, 16, 16), 2.0),
            ]
        ),
        "observation.state": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        "task": ["task 0", "task 1", "task 2"],
    }

    action = policy.predict_action_chunk(batch)

    assert action.shape == (3, 4, 3)
    assert action[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert len(captured) == 3
    assert [item["input_image_shape"] for item in captured] == [(1, 3, 16, 16)] * 3
    assert [item["proprio_shape"] for item in captured] == [(1, 2)] * 3
    assert [item["prompt"] for item in captured] == [
        cfg.prompt_template.format(task="task 0"),
        cfg.prompt_template.format(task="task 1"),
        cfg.prompt_template.format(task="task 2"),
    ]


def test_from_pretrained_does_not_initialize_wan_backbone(monkeypatch, tmp_path):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    cfg.save_pretrained(tmp_path)
    _patch_core_builder(monkeypatch)
    reference_policy = FastWAMPolicy(cfg)
    save_model(reference_policy, str(tmp_path / "model.safetensors"))

    def fail_if_wan_pretrained_is_loaded(*args, **kwargs):
        raise AssertionError("from_pretrained must not initialize or download Wan2.2 backbone components")

    monkeypatch.setattr(
        "lerobot.policies.fastwam.modular_fastwam.FastWAM.from_wan22_pretrained",
        fail_if_wan_pretrained_is_loaded,
    )
    monkeypatch.setattr(
        modeling_fastwam,
        "_build_core_model_from_architecture",
        lambda config: FakeFastWAMCore(),
        raising=False,
    )
    loaded_components_from = []
    monkeypatch.setattr(
        FastWAMPolicy,
        "load_wan_components_from_pretrained",
        lambda self, path: loaded_components_from.append(path),
    )

    policy = FastWAMPolicy.from_pretrained(tmp_path, strict=False)

    assert isinstance(policy.model, FakeFastWAMCore)
    assert loaded_components_from == [tmp_path]


def test_from_pretrained_resolves_hub_repo_to_snapshot_before_loading_sidecars(monkeypatch, tmp_path):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    cfg.save_pretrained(tmp_path)
    snapshot_calls = []

    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(tmp_path)

    @classmethod
    def fake_base_from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        assert pretrained_name_or_path == tmp_path
        assert kwargs.pop("_skip_wan_init") is True
        assert kwargs["strict"] is False
        return cls(config, _skip_wan_init=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", fake_base_from_pretrained)
    monkeypatch.setattr(
        modeling_fastwam,
        "_build_core_model_from_architecture",
        lambda config: FakeFastWAMCore(),
        raising=False,
    )
    loaded_components_from = []
    monkeypatch.setattr(
        FastWAMPolicy,
        "load_wan_components_from_pretrained",
        lambda self, path: loaded_components_from.append(path),
    )

    FastWAMPolicy.from_pretrained("org/fastwam", strict=False, local_files_only=True, revision="main")

    assert snapshot_calls[0]["repo_id"] == "org/fastwam"
    assert snapshot_calls[0]["local_files_only"] is True
    assert snapshot_calls[0]["revision"] == "main"
    assert loaded_components_from == [tmp_path]


def test_save_pretrained_copies_wan_components(monkeypatch, tmp_path):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    source = tmp_path / "source"
    tokenizer = source / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    vae = source / "Wan2.2_VAE.pth"
    text_encoder = source / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_file = tokenizer / "tokenizer.json"
    vae.write_bytes(b"vae")
    text_encoder.write_bytes(b"text")
    tokenizer_file.write_text("{}")
    core = FakeFastWAMCore()
    core.model_paths = {
        "vae": str(vae),
        "text_encoder": str(text_encoder),
        "tokenizer": str(tokenizer),
    }
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: core)
    policy = FastWAMPolicy(cfg)

    save_dir = tmp_path / "saved"
    policy.save_pretrained(save_dir)

    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "Wan2.2_VAE.pth").read_bytes() == b"vae"
    assert (save_dir / "models_t5_umt5-xxl-enc-bf16.pth").read_bytes() == b"text"
    assert (save_dir / "google" / "umt5-xxl" / "tokenizer.json").read_text() == "{}"
