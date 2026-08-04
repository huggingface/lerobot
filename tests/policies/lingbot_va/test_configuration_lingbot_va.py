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

from __future__ import annotations

import json

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import (
    EPISODE_ANCHOR_DELTA,
    LingBotVAConfig,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES


def make_config(**overrides) -> LingBotVAConfig:
    kwargs = {"device": "cpu"}
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


def test_registered_in_choice_registry() -> None:
    assert "lingbot_va" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("lingbot_va") is LingBotVAConfig


def test_type_property() -> None:
    assert make_config().type == "lingbot_va"


def test_chunk_size_and_action_steps() -> None:
    cfg = make_config(frame_chunk_size=4, action_per_frame=4)
    assert cfg.chunk_size == 16
    assert cfg.n_action_steps == 16
    # Action frame j holds the actions executed from latent frame j-1 to latent frame j (upstream's
    # retrospective convention), so the window starts one action frame *before* the clip.
    assert cfg.action_delta_indices == list(range(-4, 12))
    # 4 * (train_frames - 1) + 1 = 13 frames: exactly what the Wan VAE consumes to emit 4 latent
    # frames. Asking for 16 loaded 3 frames the encoder never reads.
    assert cfg.observation_delta_indices == list(range(13))
    assert cfg.reward_delta_indices is None


def test_train_latent_frames_extends_the_clip() -> None:
    apf, fcs, frames = 16, 2, 8
    cfg = make_config(frame_chunk_size=fcs, action_per_frame=apf, train_latent_frames=frames)
    # The inference chunk is untouched: only the training clip grows.
    assert cfg.chunk_size == fcs * apf
    assert cfg.train_frames == frames
    # 4 * (F - 1) + 1 loaded frames at stride apf/4 -> exactly F latent frames, none wasted.
    assert len(cfg.observation_delta_indices) == 4 * (frames - 1) + 1
    assert cfg.observation_delta_indices[-1] == (frames - 1) * apf
    assert cfg.action_delta_indices == list(range(-apf, (frames - 1) * apf))
    assert len(cfg.action_delta_indices) == frames * apf


def test_train_latent_frames_defaults_to_frame_chunk_size() -> None:
    cfg = make_config(frame_chunk_size=2, action_per_frame=16)
    assert cfg.train_frames == 2
    assert len(cfg.action_delta_indices) == 32


def test_train_latent_frames_below_chunk_raises() -> None:
    with pytest.raises(ValueError, match="must be >= frame_chunk_size"):
        make_config(frame_chunk_size=4, train_latent_frames=2)


def test_episode_anchor_prepends_sentinel_delta() -> None:
    apf = 16
    cfg = make_config(frame_chunk_size=2, action_per_frame=apf, action_anchor="episode")
    deltas = cfg.action_delta_indices
    # The sentinel is clamped to the episode's first frame by the dataset reader, so it must be more
    # negative than any plausible episode length.
    assert deltas[0] == EPISODE_ANCHOR_DELTA
    assert deltas[0] < -100_000
    assert deltas[1:] == list(range(-apf, apf))
    # One anchor row on top of the clip's actions.
    assert len(deltas) == 1 + 2 * apf


def test_unknown_action_anchor_raises() -> None:
    with pytest.raises(ValueError, match="action_anchor must be one of"):
        make_config(action_anchor="chunk")


def test_checkpoint_with_retired_relative_fields_still_loads(tmp_path) -> None:
    """The per-chunk relative path is gone, but checkpoints saved with its fields must still load.

    Every LingBot-VA checkpoint predating ``action_anchor`` -- including the folding absolute
    fine-tune the anchored runs warm-start from -- has ``use_relative_actions`` /
    ``relative_exclude_joints`` in its config.json. draccus raises on unknown keys, so
    ``from_pretrained`` drops keys the class no longer declares.
    """
    cfg = make_config()
    cfg.save_pretrained(tmp_path)
    config_file = tmp_path / "config.json"
    saved = json.loads(config_file.read_text())
    assert "use_relative_actions" not in saved  # retired: no longer serialized
    saved["use_relative_actions"] = False
    saved["relative_exclude_joints"] = ["gripper"]
    config_file.write_text(json.dumps(saved))

    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, LingBotVAConfig)
    assert loaded.action_anchor == "none"
    assert not hasattr(loaded, "use_relative_actions")


def test_optimizer_and_scheduler_presets() -> None:
    cfg = make_config()
    opt = cfg.get_optimizer_preset()
    assert opt.lr == cfg.optimizer_lr
    sched = cfg.get_scheduler_preset()
    assert sched.num_warmup_steps == cfg.scheduler_warmup_steps


def test_validate_features_sets_action_feature() -> None:
    cfg = make_config()
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))}
    cfg.output_features = {}
    cfg.validate_features()
    assert ACTION in cfg.output_features
    assert cfg.output_features[ACTION].shape == (len(cfg.used_action_channel_ids),)


def test_validate_features_no_visual_raises() -> None:
    cfg = make_config()
    cfg.input_features = {}
    cfg.output_features = {}
    with pytest.raises(ValueError, match="at least one visual input feature"):
        cfg.validate_features()


def test_invalid_attn_mode_raises() -> None:
    with pytest.raises(ValueError, match="attn_mode"):
        make_config(attn_mode="banana")
