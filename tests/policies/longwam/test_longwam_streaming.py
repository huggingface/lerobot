# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Causal VAE comparison adapted from NVlabs/LongLive @
# 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9, Long-WAM/tests/runtime/test_streaming_vae.py.
"""Real causal VAE numerics and incremental worker-window/reset contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import NormalizationMode
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.longwam._core.wan_video_vae import VideoVAE38_, VideoVAE_
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy
from lerobot.policies.longwam.streaming_longwam import StreamingVAE
from tests.policies.longwam.test_longwam_policy import ScriptedCore, config, observation


@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("past", [0, 16, 48])
def test_real_causal_vae_matches_full_window_and_reset(patched, past):
    torch.manual_seed(7)
    cls = VideoVAE38_ if patched else VideoVAE_
    vae = cls(
        dim=4,
        z_dim=2,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
    ).eval()
    frames = torch.randn(1, 3, past // 4 + 1, 16, 16)
    scale = [torch.zeros(2), torch.ones(2)]
    with torch.no_grad():
        expected = vae.encode(frames, scale)
    model = SimpleNamespace(
        vae=SimpleNamespace(temporal_downsample_factor=4, model=vae),
        num_clean_frames=past // 16 + 1,
        _stream_video_latents_start=vae.stream_encode_start,
        _stream_video_latents_step=vae.stream_encode_step,
        _stream_video_latents_finish=lambda: vae.stream_encode_finish(scale),
    )
    session = StreamingVAE(model, past_steps=past, sample_stride=4)
    session.begin(past, [(0, frames[0, :, 0])])
    for tick in range(4, past + 1, 4):
        session.feed(tick, frames[0, :, tick // 4])
    torch.testing.assert_close(session.finish(past), expected, atol=1e-5, rtol=1e-5)
    session.begin(past, [(i * 4, frame) for i, frame in enumerate(frames[0].unbind(1))])
    torch.testing.assert_close(session.finish(past), expected, atol=1e-5, rtol=1e-5)
    session.reset()
    assert session.anchor is None and not session._pending


def test_streaming_rejects_missing_frames_and_incomplete_windows():
    model = SimpleNamespace(
        vae=SimpleNamespace(temporal_downsample_factor=4, model=SimpleNamespace(clear_cache=lambda: None)),
        _stream_video_latents_start=lambda _: None,
    )
    with pytest.raises(ValueError, match="complete temporal"):
        StreamingVAE(model, past_steps=3, sample_stride=4)
    session = StreamingVAE(model, past_steps=16, sample_stride=4)
    frame = torch.zeros(3, 16, 16)
    with pytest.raises(ValueError, match="at least one"):
        session.begin(16, [])
    with pytest.raises(ValueError, match="Missing observed frame"):
        session.begin(16, [(0, frame), (16, frame)])
    session.begin(16, [(0, frame)])
    with pytest.raises(ValueError, match="Skipped sampled"):
        session.feed(8, frame)
    with pytest.raises(RuntimeError, match="incomplete"):
        session.finish(16)


class StreamingScriptedCore(ScriptedCore):
    """A frame-identity oracle, separate from the real VAE numerical checks."""

    def __init__(self, cfg):
        super().__init__()
        self.num_clean_frames = cfg.past_obs_steps // 16 + 1
        self.vae.temporal_downsample_factor = 4
        self.vae.model = SimpleNamespace(clear_cache=lambda: self.chunks.clear())
        self.chunks = []

    def _stream_video_latents_start(self, chunk):
        self.chunks = [chunk]

    def _stream_video_latents_step(self, chunk):
        self.chunks.append(chunk)

    def _stream_video_latents_finish(self):
        return torch.cat(self.chunks, 2)[:, :, ::4]

    def infer_joint_ar(self, **kwargs):
        window = kwargs["obs_window"]
        expected = kwargs["input_image"].unsqueeze(2) if window is None else window[:, :, ::4]
        torch.testing.assert_close(kwargs["clean_latents"], expected, atol=0, rtol=0)
        return super().infer_joint_ar(**kwargs)


@pytest.mark.parametrize("execution", ["sync", "async"])
@pytest.mark.parametrize("past", [0, 48])
def test_worker_streams_exact_planning_windows_and_resets(monkeypatch, execution, past):
    monkeypatch.setattr(
        LongWAMPolicy, "_build_core_model", staticmethod(lambda cfg, **_: StreamingScriptedCore(cfg))
    )
    cfg = config(
        execution=execution,
        trigger_stride=4 if execution == "async" else None,
        past_obs_steps=past,
        streaming_vae=True,
        normalization_mapping=dict.fromkeys(("VISUAL", "STATE", "ACTION"), NormalizationMode.IDENTITY),
    )
    with LongWAMPolicy(cfg).eval() as policy:
        pre, post = make_pre_post_processors(cfg)
        engine = SyncInferenceEngine(
            policy, pre, post, {}, ["episode", "anchor", "offset"], "pick up", "cpu", "test"
        )
        engine.start()
        for episode in (1, 2):
            engine.reset()
            for step in range(65):
                batch = observation(cfg, step, episode)
                raw = {
                    key: batch[key][0].permute(1, 2, 0).mul(255).round().byte().numpy()
                    for key in cfg.camera_keys
                }
                raw["observation.state"] = np.array([step, episode], dtype=np.float32)
                action = engine.get_action(raw)
                anchor = (
                    (step // 6) * 6 if execution == "sync" else (0 if step < 6 else ((step - 2) // 4) * 4)
                )
                torch.testing.assert_close(action, torch.tensor([episode, anchor, step - anchor]).float())
        engine.stop()
    assert not policy.model.chunks
