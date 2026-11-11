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
"""Packer boundaries, matching train/sample clocks, and custom-layout checkpoint reload."""

import json

import pytest
import torch
from draccus.utils import ParsingError

from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.flux3.f3 import packing, times_to_ids
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.flux3.helpers import FakeVideoVAE, droid_features, single_config, task_config


class RecordingVAE(FakeVideoVAE):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, video):
        self.calls.append(("frame", video.shape[2]))
        return super().encode(video)

    def encode_task(self, video):
        self.calls.append(("history", video.shape[2]))
        return super().encode(video[:, :, : 1 + 4 * ((video.shape[2] - 1) // 4)])


@pytest.mark.parametrize(
    "layout,past,history,snapshots",
    [("frame", False, 1, 1), ("history", False, 1, 1), ("history", False, 8, 2), ("history", True, 8, 2)],
)
@pytest.mark.parametrize("video_position_fps", [None, 24.0])
def test_builtin_packers_preserve_vae_boundaries_and_token_clocks(
    layout, past, history, snapshots, video_position_fps
):
    cfg = (
        single_config(video_position_fps=video_position_fps)
        if layout == "frame"
        else task_config(
            condition_on_past_actions=past,
            video_position_fps=video_position_fps,
            n_obs_steps=history,
            history_snapshots=snapshots,
        )
    )
    packer = packing.build_packer(cfg)
    vae = RecordingVAE()
    batch = 2
    videos = torch.zeros(batch, 3, cfg.window_frames, *cfg.canvas_hw)
    video = packer.pack_video(cfg, vae, videos, targets=True)
    if layout == "frame":
        assert vae.calls == [("frame", 45)] * batch
    else:
        assert vae.calls == [("history", 1)] * (batch * cfg.history_snapshots) + [("history", 32)] * batch
    vae.calls.clear()
    observed = packer.pack_video(cfg, vae, videos[:, :, : cfg.n_obs_steps], targets=False)
    assert vae.calls == [(layout, 1)] * (batch * (cfg.history_snapshots if layout == "history" else 1))
    assert set(observed) == {"x_video_cond", "x_video_cond_ids"}
    frames = packer.predicted_latent_frames(cfg)
    assert video["x_video"].shape[1] == frames * cfg.latent_hw[0] * cfg.latent_hw[1]
    video_times = video["x_video_ids"].reshape(batch, frames, -1, 4)[:, :, 0, 0]
    assert torch.equal(video_times, packer.predicted_video_times(cfg, batch))

    states = torch.ones(batch, cfg.n_obs_steps, cfg.action_dim)
    previous = torch.full_like(states, 3) if past else None
    action = packer.pack_actions(cfg, states, previous, torch.ones(batch, cfg.chunk_size, cfg.action_dim))
    cond = action["x_action_cond"]
    assert cond.shape == (batch, cfg.n_obs_steps, packer.conditioning_channels(cfg))
    # Frame state tokens are scaled; history conditioning is unscaled, with past actions first.
    assert torch.equal(cond[..., -cfg.action_dim :], states * (cfg.action_scale if layout == "frame" else 1))
    if past:
        assert torch.equal(cond[..., : cfg.action_dim], previous)
    assert action["x_action_ids"][0, 0, 0] == (int(100 / cfg.fps) if layout == "frame" else 0)
    assert torch.equal(action["x_action_ids"][..., 0], times_to_ids(packer.action_times(cfg, batch)))
    observed_action = packer.pack_actions(cfg, states, previous, targets=False)
    assert set(observed_action) == {"x_action_cond", "x_action_cond_ids"}
    assert all(torch.equal(value, action[key]) for key, value in observed_action.items())


def test_history_droid_crops_padding_for_training_and_inference():
    inputs, outputs = droid_features()
    cfg = task_config(
        input_features=inputs,
        output_features=outputs,
        camera_layout="droid",
        canvas_hw=(544, 736),
        chunk_size=17,
        n_action_steps=17,
        action_channel_weights=None,
        normalization_stats=None,
    )
    packer = packing.build_packer(cfg)
    vae = RecordingVAE()
    videos = torch.ones(1, 3, cfg.window_frames, *cfg.canvas_hw)
    # Give the padded columns distinct values to detect content leaking into tokens.
    videos[..., 640:] = 10
    training = packer.pack_video(cfg, vae, videos, targets=True)
    observed = packer.pack_video(cfg, vae, videos[:, :, : cfg.n_obs_steps], targets=False)
    assert cfg.latent_hw == (17, 20)
    assert training["x_video"].shape == (1, 1700, 96)
    assert training["x_video_cond"].shape == (1, 340, 96)
    assert all(torch.equal(value, training[key]) for key, value in observed.items())
    # A physically cropped canvas must produce exactly the same tokens and position IDs.
    reference = packer.pack_video(cfg, vae, videos[..., :640], targets=True)
    assert all(torch.equal(value, reference[key]) for key, value in training.items())
    times = training["x_video_ids"].reshape(1, packer.predicted_latent_frames(cfg), -1, 4)[:, :, 0, 0]
    assert torch.equal(times, packer.predicted_video_times(cfg, 1))


def test_custom_packer_controls_heads_training_sampling_and_reload(tmp_path, monkeypatch, fake_text_encoder):
    monkeypatch.setattr(packing, "PACKERS", dict(packing.PACKERS))
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, cfg: RecordingVAE())
    calls = []

    def pack_actions(cfg, states, past_actions, actions=None, *, targets=True):
        calls.append("actions")
        return packing.pack_actions(
            states.repeat(1, 1, 2),
            actions,
            packing.frame_action_times(cfg, states.shape[0]),
            cfg.action_modality,
            scale=cfg.action_scale,
            targets=targets,
        )

    def pack_video(*args, targets):
        calls.append("train_video" if targets else "infer_video")
        return packing.frame_pack_video(*args, targets=targets)

    def predicted_video_times(cfg, batch):
        calls.append("sample_video_times")
        return packing.frame_predicted_video_times(cfg, batch)

    def action_times(cfg, batch):
        calls.append("sample_action_times")
        return packing.frame_action_times(cfg, batch)

    custom = packing.FRAME._replace(
        conditioning_channels=lambda cfg: 2 * cfg.action_dim,
        pack_video=pack_video,
        pack_actions=pack_actions,
        predicted_video_times=predicted_video_times,
        action_times=action_times,
    )
    packing.register_packer("wide_frame", custom)
    cfg = single_config(packer="wide_frame", augment=False, caption_dropout=0)
    policy = Flux3Policy(cfg)
    assert policy.dit.in_channels["action_cond"] == 2 * cfg.action_dim
    batch = {
        "observation.images.top": torch.rand(1, cfg.window_frames, 3, 64, 96),
        OBS_STATE: torch.rand(1, 1, cfg.action_dim),
        ACTION: torch.rand(1, cfg.chunk_size, cfg.action_dim),
    }
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss) and "train_video" in calls
    prediction = policy.predict_action_chunk(batch)
    assert {"actions", "infer_video", "sample_video_times", "sample_action_times"} <= set(calls)
    policy.save_pretrained(tmp_path)
    restored = Flux3Policy.from_pretrained(tmp_path)
    assert restored.config.packer == "wide_frame" and restored.packer is custom
    assert torch.equal(restored.predict_action_chunk(batch), prediction)
    with pytest.raises(ValueError, match="already registered"):
        packing.register_packer("wide_frame", custom)
    monkeypatch.delitem(packing.PACKERS, "wide_frame")
    with pytest.raises(ParsingError) as error:
        Flux3Policy.from_pretrained(tmp_path)
    assert isinstance(error.value.__cause__, ValueError)
    assert "unknown packer 'wide_frame'" in str(error.value.__cause__)


def test_pre_packer_config_loads_and_explicit_packer_overrides_conditioning(
    tmp_path, monkeypatch, fake_text_encoder
):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, cfg: RecordingVAE())
    policy = Flux3Policy(single_config())
    policy.save_pretrained(tmp_path)
    config_file = tmp_path / "config.json"
    raw = json.loads(config_file.read_text())
    raw.pop("packer")  # Existing configs have no packer field.
    config_file.write_text(json.dumps(raw))
    restored = Flux3Policy.from_pretrained(tmp_path)
    assert restored.config.packer is None and restored.packer is packing.FRAME
    for name, value in policy.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    assert packing.build_packer(single_config(packer="history")) is packing.HISTORY
    assert packing.build_packer(task_config(packer="frame")) is packing.FRAME
    with pytest.raises(ValueError, match="unknown packer"):
        single_config(packer="missing")
