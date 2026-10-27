#!/usr/bin/env python

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

"""CPU tests for Zero-WAM on a tiny transformer with fake frozen components."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    get_policy_class,
    make_policy_config,
    make_pre_post_processors,  # noqa: E402
)
from lerobot.policies.zero_wam.configuration_zero_wam import ZeroWAMConfig  # noqa: E402
from lerobot.policies.zero_wam.modeling_zero_wam import ZeroWAMPolicy, read_video_at_fps  # noqa: E402
from lerobot.policies.zero_wam.transformer import ICL_CACHE_TYPE  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

CAMERAS = ["observation.images.base", "observation.images.left_wrist", "observation.images.right_wrist"]
REBOT_CHANNELS = [*range(14, 20), 28, *range(21, 27), 29]
Z_DIM = 4
TEXT_DIM = 16


class FakeVAE(torch.nn.Module):
    """Deterministic stand-in for the Wan2.2 VAE with its 16x spatial / 4x causal temporal downsample."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv3d(3, 2 * Z_DIM, 1)
        self.config = SimpleNamespace(latents_mean=[0.0] * Z_DIM, latents_std=[1.0] * Z_DIM, z_dim=Z_DIM)

    def moments(self, video, first_chunk=True):
        t = video.shape[2]
        frames = 1 + (t - 1) // 4 if first_chunk else t // 4
        out = torch.nn.functional.adaptive_avg_pool3d(
            video.float(), (frames, video.shape[3] // 16, video.shape[4] // 16)
        )
        return self.proj(out).to(video.dtype)

    def encode(self, video):
        mean, logvar = self.moments(video).chunk(2, dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean, logvar=logvar))


class FakeStreamingVAE:
    def __init__(self, vae):
        self.vae = vae
        self.first = True

    def clear_cache(self):
        self.first = True

    def encode_chunk(self, video):
        out = self.vae.moments(video, first_chunk=self.first)
        self.first = False
        return out


class FakeTokenizer:
    def __call__(self, prompts, max_length, **kwargs):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            n = min(len(p.split()) + 1, max_length)
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, TEXT_DIM)

    def forward(self, ids, mask):
        return SimpleNamespace(last_hidden_state=self.emb(ids % 64))


def tiny_config(**overrides) -> ZeroWAMConfig:
    kwargs = {
        "num_attention_heads": 2,
        "attention_head_dim": 24,
        "in_channels": Z_DIM,
        "out_channels": Z_DIM,
        "text_dim": TEXT_DIM,
        "freq_dim": 16,
        "ffn_dim": 64,
        "num_layers": 4,
        "action_inner_dim": 48,
        "action_ffn_dim": 64,
        "mcp_hidden_collect_layers": (0, 1, 2, 3),
        "rope_max_seq_len": 64,
        "attention_backend": "sdpa",
        "gradient_checkpointing": False,
        "dtype": torch.float32,
        "frozen_dtype": torch.float32,
        "device": "cpu",
        "text_encoder_device": "cpu",
        "max_sequence_length": 8,
        "obs_cam_keys": CAMERAS,
        "height": 32,
        "width": 32,
        "video_frame_stride": 1,
        "frame_chunk_size": 2,
        "used_action_channel_ids": REBOT_CHANNELS,
        "train_num_latent_frames": 5,
        "num_inference_steps": 2,
        "action_num_inference_steps": 2,
        "icl_height": 32,
        "icl_width": 64,
        "input_features": {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)) for k in CAMERAS},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    }
    kwargs.update(overrides)
    return ZeroWAMConfig(**kwargs)


def make_policy(config: ZeroWAMConfig) -> ZeroWAMPolicy:
    torch.manual_seed(0)
    policy = ZeroWAMPolicy(config)
    vae = FakeVAE()
    policy._frozen = {
        "vae": vae,
        "streaming_vae": FakeStreamingVAE(vae),
        "text_encoder": FakeTextEncoder(),
        "tokenizer": FakeTokenizer(),
    }
    return policy


def training_batch(config: ZeroWAMConfig, batch_size: int = 1) -> dict:
    frames = len(config.observation_delta_indices)
    steps = len(config.action_delta_indices)
    batch = {k: torch.rand(batch_size, frames, 3, 48, 64) for k in CAMERAS}
    batch[ACTION] = torch.randn(batch_size, steps, 14)
    batch[f"{ACTION}_is_pad"] = torch.zeros(batch_size, steps, dtype=torch.bool)
    batch["task"] = ["pick up the red cup | put the red cup in the bin"] * batch_size
    batch["episode_index"] = torch.zeros(batch_size, dtype=torch.long)
    batch["frame_index"] = torch.full((batch_size,), 10)
    return batch


def write_icl_latent(tmp_path, name="human.pth"):
    path = tmp_path / name
    torch.save({"latent": torch.randn(1, Z_DIM, 3, 2, 4), "text_emb": torch.randn(8, TEXT_DIM)}, path)
    return path


def test_registered_and_resolved() -> None:
    assert isinstance(make_policy_config("zero_wam"), ZeroWAMConfig)
    assert get_policy_class("zero_wam") is ZeroWAMPolicy


def test_temporal_layout() -> None:
    config = ZeroWAMConfig(video_frame_stride=2, train_num_latent_frames=11, frame_chunk_size=2)
    assert config.action_per_frame == 8
    assert config.chunk_size == 16
    assert config.observation_delta_indices == list(range(0, 82, 2))
    assert config.action_delta_indices == list(range(80))


def test_invalid_channels_raise() -> None:
    with pytest.raises(ValueError):
        ZeroWAMConfig(used_action_channel_ids=[0, 0])
    with pytest.raises(ValueError):
        ZeroWAMConfig(used_action_channel_ids=[30])


@pytest.mark.parametrize("with_icl", [False, True])
def test_training_loss_backward(tmp_path, with_icl: bool) -> None:
    overrides = {"drop_icl": 0.0, "droptext_target": 0.0}
    if with_icl:
        latent = write_icl_latent(tmp_path)
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text(
            json.dumps({"episode_index": 0, "start_frame": 0, "end_frame": 50, "latent_path": str(latent)})
            + "\n"
        )
        overrides["icl_pairs_path"] = str(pairs)
    policy = make_policy(tiny_config(**overrides))
    policy.train()
    loss, logs = policy.forward(training_batch(policy.config, batch_size=2))
    assert torch.isfinite(loss)
    assert set(logs) == {"latent_loss", "action_loss", "mcp_loss"}
    loss.backward()
    grads = [p.grad for p in policy.get_optim_params() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert policy.transformer.action_embedder.weight.grad.abs().sum() > 0
    before = policy.transformer.action_embedder.weight.detach().clone()
    torch.optim.AdamW(policy.get_optim_params(), lr=1e-3).step()
    assert not torch.equal(before, policy.transformer.action_embedder.weight)


def test_training_preserves_selected_task(monkeypatch) -> None:
    policy = make_policy(tiny_config(droptext_target=0.0))
    batch = training_batch(policy.config)
    supplied = "pick up the red cup | put the red cup in the bin"
    seen = []
    encode = policy._encode_text

    def capture(prompt):
        seen.append(prompt)
        return encode(prompt)

    monkeypatch.setattr(policy, "_encode_text", capture)
    policy.forward(batch)
    assert seen == [supplied]


@pytest.mark.parametrize("stride", [1, 2])
def test_icl_pair_requires_the_entire_training_window(stride) -> None:
    policy = make_policy(tiny_config(video_frame_stride=stride))
    # A conditioning frame plus 16*stride future frames fit exactly at frame 100.
    stop = 101 + 16 * stride
    policy._icl_pairs = {7: [{"start_frame": 100, "end_frame": stop, "latent_path": "matched.pth"}]}
    assert policy._find_icl_latent(7, 100) == "matched.pth"
    assert policy._find_icl_latent(7, 101) is None
    assert policy._find_icl_latent(7, 99) is None
    assert policy._find_icl_latent(8, 100) is None


def test_training_from_compact_prompt_pool(tmp_path, monkeypatch) -> None:
    latent = write_icl_latent(tmp_path, "other-episode.pth")
    pools = tmp_path / "pools.jsonl"
    pools.write_text(
        json.dumps(
            {
                "pool_id": "task",
                "dataset": "abc",
                "split": "train",
                "members": [
                    {
                        "episode_index": 0,
                        "start_frame": 0,
                        "end_frame": 50,
                        "latent_path": "must-not-load.pth",
                    },
                    {"episode_index": 1, "start_frame": 0, "end_frame": 50, "latent_path": str(latent)},
                ],
            }
        )
        + "\n"
    )
    policy = make_policy(tiny_config(icl_pools_path=str(pools), drop_icl=0.0, droptext_target=0.0))
    loaded = []
    original = policy._load_icl_latent

    def capture(path):
        loaded.append(path)
        return original(path)

    monkeypatch.setattr(policy, "_load_icl_latent", capture)
    loss, _ = policy(training_batch(policy.config))
    loss.backward()
    assert loaded == [str(latent)]
    assert torch.isfinite(loss) and policy.transformer.action_embedder.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="only one of icl_pairs_path and icl_pools_path"):
        tiny_config(icl_pairs_path="pairs.jsonl", icl_pools_path="pools.jsonl")


def test_processor_quantiles_survive_save_reload(tmp_path) -> None:
    config = tiny_config()
    q01 = torch.arange(14, dtype=torch.float32)
    q99 = q01 + 2
    q99[-1] = q01[-1]  # Constant gripper channel must stay finite.
    pre, post = make_pre_post_processors(config, dataset_stats={ACTION: {"q01": q01, "q99": q99}})
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=tmp_path)
    batch = {**{k: torch.zeros(3, 48, 64) for k in CAMERAS}, ACTION: q01.clone(), "task": "move"}
    normalized = loaded_pre(batch)[ACTION]
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[..., :-1], -torch.ones(1, 13))
    torch.testing.assert_close(loaded_post(normalized), q01[None])


def test_inference_rejects_multiple_environments() -> None:
    policy = make_policy(tiny_config())
    with pytest.raises(ValueError, match="batch size 1"):
        policy.select_action({k: torch.zeros(2, 3, 48, 64) for k in CAMERAS})


@pytest.mark.parametrize("no_mcp", [False, True])
def test_conversion_strict_reload_and_normalization(tmp_path, monkeypatch, no_mcp) -> None:
    from safetensors.torch import save_file

    policy = make_policy(tiny_config())
    source = tmp_path / "source" / "transformer"
    source.mkdir(parents=True)
    policy.transformer.save_config(source)
    save_file(
        {k: v.detach().clone() for k, v in policy.transformer.state_dict().items()},
        source / "diffusion_pytorch_model.safetensors",
    )
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({ACTION: {"q01": [1.0] * 14, "q99": [3.0] * 14}}))
    output = tmp_path / "converted"
    argv = [
        "convert_checkpoint.py",
        "--source",
        str(source.parent),
        "--output",
        str(output),
        "--cameras",
        *CAMERAS,
        "--action-channels",
        *map(str, REBOT_CHANNELS),
        "--dtype",
        "float32",
        "--action-stats",
        str(stats),
    ]
    if no_mcp:
        argv.append("--no-mcp")
    monkeypatch.setattr("sys.argv", argv)
    script = Path(__file__).parents[3] / "examples/zero_wam/convert_checkpoint.py"
    runpy.run_path(str(script), run_name="__main__")
    loaded = ZeroWAMPolicy.from_pretrained(output, strict=True)
    assert loaded.config.enable_mcp is not no_mcp
    for name, value in loaded.transformer.state_dict().items():
        torch.testing.assert_close(value, policy.transformer.state_dict()[name])
    _, post = make_pre_post_processors(loaded.config, pretrained_path=output)
    torch.testing.assert_close(post(torch.zeros(1, 14)), torch.full((1, 14), 2.0))
    report = json.loads((output / "conversion_report.json").read_text())
    assert not report["missing"] and not report["unexpected"] and not report["shape_mismatches"]
    assert bool(report["dropped_mcp"]) is no_mcp


def test_gradient_checkpointing_matches() -> None:
    batch = training_batch(tiny_config())
    losses = []
    for checkpointing in (False, True):
        policy = make_policy(tiny_config(gradient_checkpointing=checkpointing))
        policy.train()
        torch.manual_seed(1)
        losses.append(policy.forward(batch)[0])
    torch.testing.assert_close(losses[0], losses[1])


def test_gradient_checkpointing_gradients_match_across_batch() -> None:
    # Every sample draws its own chunk size and window, so each needs its own masks on recompute.
    batch = training_batch(tiny_config(train_frame_chunk_size=0, train_attn_window=0), batch_size=3)
    grads = []
    for checkpointing in (False, True):
        policy = make_policy(tiny_config(gradient_checkpointing=checkpointing))
        policy.train()
        torch.manual_seed(3)
        policy.forward(batch)[0].backward()
        grads.append({n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], msg=name)


def test_save_load_roundtrip(tmp_path) -> None:
    policy = make_policy(tiny_config())
    policy.save_pretrained(tmp_path)
    loaded = ZeroWAMPolicy.from_pretrained(tmp_path, strict=True)
    for (name, a), (_, b) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=name)
    # attn2 action K/V alias the video projections and must stay tied after loading.
    block = loaded.transformer.blocks[0].attn2
    assert block.action_to_k is block.to_k and block.action_norm_k is block.norm_k


def run_rollout(policy: ZeroWAMPolicy, steps: int) -> list[torch.Tensor]:
    torch.manual_seed(0)
    actions = []
    for _ in range(steps):
        obs = {k: torch.rand(1, 3, 48, 64) for k in CAMERAS}
        obs["task"] = ["pick up the red cup"]
        actions.append(policy.select_action(obs))
    return actions


@pytest.mark.parametrize("with_icl", [False, True])
def test_streaming_inference(tmp_path, with_icl: bool) -> None:
    overrides = {"icl_latent_path": str(write_icl_latent(tmp_path))} if with_icl else {}
    policy = make_policy(tiny_config(**overrides))
    apf = policy.config.action_per_frame
    # First chunk skips the conditioning frame; later chunks yield frame_chunk_size frames each.
    first, later = apf, policy.config.chunk_size
    actions = run_rollout(policy, first + 2 * later + 1)
    assert all(a.shape == (1, 14) and torch.isfinite(a).all() for a in actions)
    assert policy._chunk_idx == 3
    counts = policy.transformer.cache_counts()
    assert (counts[ICL_CACHE_TYPE] > 0) == with_icl
    policy.reset()
    assert policy.transformer.cache_counts() == {}
    assert not policy._action_queue


def test_icl_video_file_is_encoded_as_a_whole_clip(tmp_path) -> None:
    av = pytest.importorskip("av")
    import numpy as np

    video = tmp_path / "human_demo.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(90):  # 3 s: 37 frames at the 12 fps ICL rate, not a streaming-chunk length
            frame = np.full((48, 64, 3), (i * 3) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    policy = make_policy(tiny_config(icl_video_path=str(video), icl_width=64))
    streamed_widths = []
    streaming = policy._frozen["streaming_vae"]
    original = streaming.encode_chunk

    def tracking_encode_chunk(clip):
        streamed_widths.append(clip.shape[-1])
        return original(clip)

    streaming.encode_chunk = tracking_encode_chunk
    actions = run_rollout(policy, policy.config.action_per_frame + policy.config.chunk_size + 1)
    assert all(torch.isfinite(a).all() for a in actions)
    assert policy.transformer.cache_counts()[ICL_CACHE_TYPE] > 0
    # Only camera keyframes (width 32) are streamed; the ICL clip (width 64) is encoded whole.
    assert streamed_widths and set(streamed_widths) == {policy.config.width}


def test_bounded_prompt_keeps_the_selected_episode_and_final_motion(tmp_path) -> None:
    av = pytest.importorskip("av")
    import numpy as np

    video = tmp_path / "three_episodes.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(90):
            color = (230, 0, 0) if i < 30 else (0, 80 + (i - 30) * 5, 0) if i < 60 else (0, 0, 230)
            rgb = np.empty((48, 64, 3), dtype=np.uint8)
            rgb[:] = color
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    clip = read_video_at_fps(str(video), 12, start_time_s=1, end_time_s=2, max_frames=5)
    assert clip.shape == (5, 3, 48, 64)
    assert clip[:, [0, 2]].max() < 0.05  # Neither adjacent red/blue episode is included.
    assert clip[-1, 1].mean() > clip[0, 1].mean() + 0.4  # Sample the whole motion, not its first 5 frames.
    with pytest.raises(ValueError, match="past the available video"):
        read_video_at_fps(str(video), 12, start_time_s=2, end_time_s=4, max_frames=5)


def test_episode_prompt_uses_video_offset_and_source_episode_id(tmp_path) -> None:
    example = Path(__file__).resolve().parents[3] / "examples/zero_wam/prepare_episode_prompts.py"
    prepare = runpy.run_path(str(example))["episode_video_spec"]
    info = {"fps": 30, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    episode = {
        "episode_index": 700,
        "length": 180,
        "videos/top/from_timestamp": 10.0,
        "videos/top/chunk_index": 2,
        "videos/top/file_index": 3,
    }
    prompt = prepare(tmp_path, info, episode, "top", 30, 90)
    assert prompt["prompt_episode_index"] == 700
    assert prompt["prompt_start_s"] == 11.0 and prompt["prompt_end_s"] == 13.0
    assert prompt["prompt_video"] == str(tmp_path / "videos/top/chunk-002/file-003.mp4")
    with pytest.raises(ValueError, match="Invalid prompt frame interval"):
        prepare(tmp_path, info, episode, "top", 0, 181)


def test_prompt_encoding_identity_preserves_crops_and_captions(tmp_path) -> None:
    example = Path(__file__).resolve().parents[3] / "examples/zero_wam/encode_human_videos.py"
    spec = runpy.run_path(str(example))["prompt_spec"]
    base = {
        "prompt_video": str(tmp_path / "episodes.mp4"),
        "prompt_text": "fold the cloth",
        "prompt_start_s": 10,
        "prompt_end_s": 20,
    }
    assert spec(base, 81) != spec({**base, "prompt_start_s": 12}, 81)
    assert spec(base, 81) != spec({**base, "prompt_text": "stack the cloth"}, 81)
    assert spec({"human_video": "demo.mp4", "human_text": "fold"})["text"] == "fold"


def test_episode_pair_preparation_preserves_ids_and_rejects_self_pairs(tmp_path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    meta = root / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": ep,
                    "length": 90,
                    "tasks": ["fold"],
                    "videos/top/from_timestamp": offset,
                    "videos/top/chunk_index": 0,
                    "videos/top/file_index": file,
                }
                for ep, offset, file in [(10, 4.0, 4), (20, 7.0, 5)]
            ]
        ),
        meta / "episodes/chunk-000/file-000.parquet",
    )
    pair = {
        "dataset": "abc",
        "split": "train",
        "target_episode": 10,
        "prompt_episode": 20,
        "target_from_frame": 0,
        "target_to_frame": 90,
        "prompt_from_frame": 30,
        "prompt_to_frame": 90,
        "detail_raw": "fold",
    }
    source, output = tmp_path / "input.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(pair) + "\n")
    example = Path(__file__).resolve().parents[3] / "examples/zero_wam/prepare_episode_prompts.py"
    main = runpy.run_path(str(example))["main"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(example),
            "--dataset-root",
            str(root),
            "--camera",
            "top",
            "--pairs",
            str(source),
            "--dataset-id",
            "abc",
            "--metadata-only",
            "--output",
            str(output),
        ],
    )
    main()
    saved = output.read_text()
    row = json.loads(saved)
    assert row["episode_index"] == 10 and row["prompt_episode_index"] == 20
    assert row["prompt_start_s"] == 8 and row["prompt_end_s"] == 10
    source.write_text(json.dumps({**pair, "prompt_episode": "10"}) + "\n")
    with pytest.raises(ValueError, match="another episode"):
        main()
    assert output.read_text() == saved
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_encoder_writes_distinct_crops_and_standalone_prompt_index(tmp_path, monkeypatch) -> None:
    example = Path(__file__).resolve().parents[3] / "examples/zero_wam/encode_human_videos.py"
    main = runpy.run_path(str(example))["main"]
    scope = main.__globals__
    monkeypatch.setitem(scope, "load_vae", lambda *a, **k: FakeVAE())
    monkeypatch.setitem(scope, "load_tokenizer", lambda *a, **k: FakeTokenizer())
    monkeypatch.setitem(scope, "load_text_encoder", lambda *a, **k: FakeTextEncoder())
    seen = []

    def decode(video, fps, **crop):
        seen.append(crop)
        return torch.full((5, 3, 48, 64), float(crop["start_time_s"]))

    monkeypatch.setitem(scope, "read_video_at_fps", decode)
    video = tmp_path / "source.mp4"
    video.touch()
    pairs = tmp_path / "source.jsonl"
    base = {"prompt_video": str(video), "prompt_text": "fold", "prompt_start_s": 0, "prompt_end_s": 1}
    # The first row is a standalone physical prompt. The second has a training target.
    pairs.write_text(
        json.dumps(base)
        + "\n"
        + json.dumps(
            {
                **base,
                "prompt_start_s": 1,
                "prompt_end_s": 2,
                "episode_index": 7,
                "start_frame": 0,
                "end_frame": 90,
            }
        )
        + "\n"
    )
    output = tmp_path / "latents"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(example),
            "--pairs",
            str(pairs),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--height",
            "32",
            "--width",
            "64",
            "--max-sequence-length",
            "8",
        ],
    )
    main()
    prompts = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert len(prompts) == len(seen) == 2 and len(train) == 1
    assert len({p["latent_path"] for p in prompts}) == 2
    assert train[0]["episode_index"] == 7
    for p in prompts:
        encoded = torch.load(p["latent_path"], weights_only=True)
        assert encoded["source_start_s"] == p["start_time_s"]


def test_pool_preparation_and_encoding_preserve_compact_members(tmp_path) -> None:
    base = Path(__file__).resolve().parents[3] / "examples/zero_wam"
    flatten = runpy.run_path(str(base / "prepare_episode_prompts.py"))["pool_member_rows"]
    encode = runpy.run_path(str(base / "encode_human_videos.py"))["encoded_pools"]
    import io

    raw = {
        "pool_id": "fold",
        "dataset": "abc",
        "split": "train",
        "text": "fold the shirt",
        "members": [
            {"episode_index": 5, "start_frame": 10, "end_frame": 100},
            {"episode_index": 8, "start_frame": 0, "end_frame": 120},
            {"episode_index": 12, "start_frame": 0, "end_frame": 140},
        ],
    }
    rows = list(flatten(io.StringIO(json.dumps(raw)), "abc", "train"))
    assert len(rows) == 3  # Three source members represent six possible directed assignments.
    resolved = [
        {
            **r,
            "prompt_episode_index": r["prompt_episode"],
            "prompt_start_frame": r["prompt_from_frame"],
            "prompt_end_frame": r["prompt_to_frame"],
        }
        for r in rows
    ]
    result = encode(resolved, ["5", "8", "12"], {"5": "5.pth", "8": "8.pth", "12": "12.pth"})
    assert len(result) == 1 and len(result[0]["members"]) == 3
    assert result[0]["split"] == "train" and result[0]["dataset"] == "abc"
    assert result[0]["members"][0] == {**raw["members"][0], "latent_path": "5.pth"}


def test_precomputed_text_embeddings_skip_text_encoder(tmp_path) -> None:
    cache_path = tmp_path / "text.pt"
    torch.save({"": torch.randn(1, TEXT_DIM), "pick up the red cup": torch.randn(5, TEXT_DIM)}, cache_path)
    policy = make_policy(tiny_config(text_embedding_cache_path=str(cache_path)))
    vae = policy._frozen["vae"]
    policy._frozen = {}
    policy._ensure_frozen_modules = lambda: None
    policy._frozen = {"vae": vae, "streaming_vae": FakeStreamingVAE(vae)}
    policy._precomputed_text = torch.load(cache_path)

    def fail():
        raise AssertionError("text encoder must not load for cached prompts")

    policy._ensure_text_encoder = fail
    emb = policy._encode_text("pick up the red cup")
    assert emb.shape == (1, policy.config.max_sequence_length, TEXT_DIM)
    assert torch.equal(emb[0, 5:], torch.zeros_like(emb[0, 5:]))
    run_rollout(policy, 2)


def test_instruction_change_applies_next_chunk() -> None:
    policy = make_policy(tiny_config())
    run_rollout(policy, 1)
    policy.drop_queued_actions()
    assert policy.count_queued_actions() == policy.config.action_per_frame - 1
    obs = {k: torch.rand(1, 3, 48, 64) for k in CAMERAS}
    obs["task"] = ["pick up the blue block"]
    for _ in range(policy.config.action_per_frame):
        policy.select_action(obs)
    assert policy._prompt == "pick up the blue block"


def test_feedback_cache_matches_clipped_actions() -> None:
    policy = make_policy(tiny_config(action_clip=0.01))
    obs = {k: torch.rand(1, 3, 48, 64) for k in CAMERAS}
    obs["task"] = ["move the cup"]
    first_action = policy.select_action(obs)
    dispatched = torch.cat([first_action, torch.stack(list(policy._action_queue))])
    cached = policy._last_actions[0, policy.config.used_action_channel_ids, 1:].flatten(1).T
    assert dispatched.abs().max() <= policy.config.action_clip
    torch.testing.assert_close(cached, dispatched)


def test_inference_chunk_needs_a_future_frame() -> None:
    with pytest.raises(ValueError, match="frame_chunk_size must be at least 2"):
        tiny_config(frame_chunk_size=1)
