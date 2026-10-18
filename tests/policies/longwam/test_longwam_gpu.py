# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-checkpoint CUDA lifecycle and streaming VAE checks, without downloads.

Set LONGWAM_NATIVE_CHECKPOINT to a converted IDM or COD checkpoint to run.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy
from lerobot.policies.longwam.streaming_longwam import StreamingVAE

CHECKPOINT = os.environ.get("LONGWAM_NATIVE_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT or not torch.cuda.is_available(), reason="local CUDA checkpoint required"
)


@pytest.fixture(scope="module")
def policy():
    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    cfg.device, cfg.execution, cfg.trigger_stride, cfg.streaming_vae = "cuda", "async", 5, True
    with LongWAMPolicy.from_pretrained(CHECKPOINT, config=cfg).eval() as model:
        yield model


@pytest.fixture
def observation(policy):
    cfg = policy.config
    generator = torch.Generator().manual_seed(7)
    batch = {
        key: torch.rand(1, 3, *cfg.camera_size, generator=generator).to("cuda") for key in cfg.camera_keys
    }
    batch.update(
        {"observation.state": torch.zeros(1, cfg.proprio_dim, device="cuda"), "task": ["pick up the bowl"]}
    )
    return batch


def test_full_weight_streaming_vae_matches_window(policy, observation):
    policy.close()
    core, cfg = policy.model, policy.config
    frame = policy._images(observation, temporal=False)[0]
    frames = [(frame + i * 0.02).clamp(-1, 1) for i in range(cfg.past_obs_steps // cfg.video_stride + 1)]
    window = torch.stack(frames, dim=1).unsqueeze(0)
    with torch.no_grad():
        expected = core._encode_video_latents(window, tiled=False)
        session = StreamingVAE(core, past_steps=cfg.past_obs_steps, sample_stride=cfg.video_stride)
        session.begin(cfg.past_obs_steps, [(0, frames[0])])
        for step in range(cfg.video_stride, cfg.past_obs_steps + 1, cfg.video_stride):
            session.feed(step, frames[step // cfg.video_stride])
        actual = session.finish(cfg.past_obs_steps)
        error = (expected.float() - actual.float()).abs().max().item()
        print(f"FULL_VAE_MAX_ABS_ERROR={error}", flush=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        session.reset()


@pytest.mark.parametrize("transition", ["late_handoff", "reset_pending"])
def test_full_weight_async_pending_transition(policy, observation, monkeypatch, transition):
    policy.close()
    entered, release = threading.Event(), threading.Event()
    chunks = []
    original = policy.model.infer_joint_ar

    def delayed(**kwargs):
        if len(chunks) == 1:
            entered.set()
            if not release.wait(60):
                raise TimeoutError("Lifecycle test did not release the full-model inference")
        result = original(**kwargs)
        chunks.append(result["action"].detach().float().cpu())
        return result

    monkeypatch.setattr(policy.model, "infer_joint_ar", delayed)

    def at_step(step):
        # Distinct inputs make stale chunks distinguishable with real weights too.
        batch = dict(observation)
        for key in policy.config.camera_keys:
            batch[key] = (observation[key] + step * 0.01).clamp(0, 1)
        batch["observation.state"] = observation["observation.state"] + step * 0.03
        return batch

    try:
        first = policy.select_action(at_step(0)).cpu()
        last = 10 if transition == "late_handoff" else 6
        for step in range(1, last):
            policy.select_action(at_step(step))
        assert entered.wait(30)
        if transition == "reset_pending":
            policy.reset()
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(policy.select_action, at_step(10 if transition == "late_handoff" else 0))
            try:
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.1)
            finally:
                release.set()
            actual = future.result(120).cpu()
        assert not torch.equal(chunks[0][5], chunks[1][5]), "Fixture must distinguish old and new chunks"
        if transition == "late_handoff":
            torch.testing.assert_close(actual[0], chunks[1][5], atol=0, rtol=0)
            assert policy._runtime.metrics["late_handoffs"] >= 1
        else:
            torch.testing.assert_close(actual, first, atol=0, rtol=0)
            assert policy._runtime.episode == 2
        assert torch.isfinite(actual).all()
    finally:
        release.set()
        policy.close()
    assert policy._runtime is None
    assert not any(t.name.startswith("longwam") for t in threading.enumerate())
