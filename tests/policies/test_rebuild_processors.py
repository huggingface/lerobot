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

"""`make_pre_post_processors(..., rebuild_from_config=True)`.

The saved pipeline is normally authoritative, so a `--policy.*` flag that should add or reconfigure a
processor *step* silently does nothing when the checkpoint predates that step. Rebuilding takes the
structure from the config and the normalization stats from the checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.processor_lingbot_va import (
    LingBotEpisodeAnchorStep,
    make_lingbot_va_pre_post_processors,
)
from lerobot.processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

ACTION_NAMES = ["right_joint_1.pos", "right_gripper.pos", "left_joint_1.pos", "left_gripper.pos"]
Q01 = [-40.0, -60.0, -35.0, -55.0]
Q99 = [40.0, 15.0, 40.0, 16.0]


def _config(**overrides) -> LingBotVAConfig:
    cfg = LingBotVAConfig(device="cpu", used_action_channel_ids=[0, 1, 2, 3], **overrides)
    cfg.normalization_mapping = {
        **cfg.normalization_mapping,
        FeatureType.ACTION.value: NormalizationMode.QUANTILES,
    }
    cfg.input_features = {
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    cfg.output_features = {}
    cfg.validate_features()
    cfg.action_feature_names = ACTION_NAMES
    return cfg


def _stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        ACTION: {"q01": torch.tensor(Q01), "q99": torch.tensor(Q99)},
        OBS_STATE: {"q01": torch.tensor(Q01), "q99": torch.tensor(Q99)},
    }


@pytest.fixture
def checkpoint(tmp_path):
    """A checkpoint saved WITHOUT the anchor step, i.e. the situation this flag exists for."""
    pre, post = make_lingbot_va_pre_post_processors(_config(action_anchor="none"), dataset_stats=_stats())
    # Drop the (disabled) anchor step entirely so the saved pipeline genuinely predates it.
    pre.steps = [s for s in pre.steps if not isinstance(s, LingBotEpisodeAnchorStep)]
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    return tmp_path


def _step_names(pipeline) -> list[str]:
    return [type(s).__name__ for s in pipeline.steps]


def test_without_the_flag_the_saved_pipeline_wins(checkpoint) -> None:
    # Baseline: the config asks for anchoring, the checkpoint has no such step, and loading honours
    # the checkpoint -- so no anchoring happens. This is the failure mode being fixed.
    pre, _ = make_pre_post_processors(
        policy_cfg=_config(action_anchor="episode"), pretrained_path=str(checkpoint)
    )
    assert "LingBotEpisodeAnchorStep" not in _step_names(pre)


def test_rebuild_adds_the_step_the_checkpoint_lacks(checkpoint) -> None:
    pre, post = make_pre_post_processors(
        policy_cfg=_config(action_anchor="episode"),
        pretrained_path=str(checkpoint),
        rebuild_from_config=True,
    )
    assert "LingBotEpisodeAnchorStep" in _step_names(pre)
    anchor = next(s for s in pre.steps if isinstance(s, LingBotEpisodeAnchorStep))
    assert anchor.enabled
    assert anchor.action_names == ACTION_NAMES
    # And it lands before the normalizer, as the policy's factory arranges it.
    assert _step_names(pre).index("LingBotEpisodeAnchorStep") < _step_names(pre).index(
        "NormalizerProcessorStep"
    )
    assert "AbsoluteActionsProcessorStep" in _step_names(post)


def test_rebuild_preserves_checkpoint_stats(checkpoint) -> None:
    """The safety property: no dataset stats available, so the checkpoint's must be carried over."""
    pre, post = make_pre_post_processors(
        policy_cfg=_config(action_anchor="episode"),
        pretrained_path=str(checkpoint),
        rebuild_from_config=True,
    )
    normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    assert torch.allclose(normalizer._tensor_stats[ACTION]["q01"], torch.tensor(Q01))
    assert torch.allclose(normalizer._tensor_stats[ACTION]["q99"], torch.tensor(Q99))

    # The postprocessor must round-trip too: -1 -> q01. Anchoring is enabled on this pipeline, so
    # latch a zero anchor first (unanchoring then adds nothing and only the unnormalizer's stats are
    # under test). Without a latched anchor the unanchor step refuses to run, by design.
    anchor = next(s for s in pre.steps if isinstance(s, LingBotEpisodeAnchorStep))
    anchor.set_cached_state(torch.zeros(1, 4))
    out_lo = post(torch.full((1, 4), -1.0))
    assert torch.allclose(out_lo, torch.tensor(Q01).unsqueeze(0), atol=1e-3)


def test_dataset_stats_take_precedence_over_the_checkpoint(checkpoint) -> None:
    fresh = {ACTION: {"q01": torch.tensor([-1.0] * 4), "q99": torch.tensor([1.0] * 4)}}
    pre, _ = make_pre_post_processors(
        policy_cfg=_config(action_anchor="episode"),
        pretrained_path=str(checkpoint),
        rebuild_from_config=True,
        dataset_stats=fresh,
    )
    normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    assert torch.allclose(normalizer._tensor_stats[ACTION]["q99"], torch.tensor([1.0] * 4))


def test_rebuild_without_any_stats_raises(tmp_path) -> None:
    # A checkpoint with no stats at all: silently normalizing with nothing would be far worse than
    # failing, since it looks like a working policy that emits garbage.
    pre, post = make_lingbot_va_pre_post_processors(_config(action_anchor="none"), dataset_stats=None)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="needs normalization stats"):
        make_pre_post_processors(
            policy_cfg=_config(action_anchor="episode"),
            pretrained_path=str(tmp_path),
            rebuild_from_config=True,
        )


def test_rebuild_applies_step_overrides(checkpoint) -> None:
    """Step overrides must survive the rebuild.

    The rename map is a dataset->policy key mapping that exists only on the CLI, never in the policy
    config, and it reaches the pipeline as a step override. A rebuild that ignored overrides would
    produce a rename step with an empty map and every camera key would then miss.
    """
    rename_map = {f"{OBS_IMAGES}.base": f"{OBS_IMAGES}.cam"}
    pre, post = make_pre_post_processors(
        policy_cfg=_config(action_anchor="episode"),
        pretrained_path=str(checkpoint),
        rebuild_from_config=True,
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": rename_map},
            "device_processor": {"device": "cpu"},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    rename = next(s for s in pre.steps if type(s).__name__ == "RenameObservationsProcessorStep")
    assert rename.rename_map == rename_map

    # __post_init__ re-ran, so derived state is consistent rather than stale.
    device_step = next(s for s in pre.steps if type(s).__name__ == "DeviceProcessorStep")
    assert device_step.tensor_device.type == "cpu"

    # And the stats carried over from the checkpoint are still intact after the override pass.
    normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    assert torch.allclose(normalizer._tensor_stats[ACTION]["q01"], torch.tensor(Q01))
    assert "AbsoluteActionsProcessorStep" in _step_names(post)


def test_rebuild_is_off_by_default(checkpoint) -> None:
    pre_default, _ = make_pre_post_processors(
        policy_cfg=_config(action_anchor="none"), pretrained_path=str(checkpoint)
    )
    pre_explicit, _ = make_pre_post_processors(
        policy_cfg=_config(action_anchor="none"),
        pretrained_path=str(checkpoint),
        rebuild_from_config=False,
    )
    assert _step_names(pre_default) == _step_names(pre_explicit)
    assert "LingBotEpisodeAnchorStep" not in _step_names(pre_default)
