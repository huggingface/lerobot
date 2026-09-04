# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Config surface for the upstream native-depth / DINO-video distillation branch.

Upstream LingBot-VLA 2.0 exposes the distillation branch through a nested
``train.align_params`` dict (frozen MoGe / LingBot-Depth / DINO-video teachers
produce per-batch targets that the VLA aligns against). This port keeps the
same dict as the single source of truth: a valid dict passes strict schema
validation at config construction; a malformed one fails before any dataset /
model / teacher initialization with an actionable message.

These tests pin:

1. the action-only default (empty ``align_params``) keeps working;
2. the official RoboTwin align_params passes validation;
3. malformed variants (missing keys / bad mode / bad model_type / broken
   cross-field constraints) fail with precise ValueErrors;
4. ``enable_expert_vision`` is still rejected as a separate unimplemented
   branch (it is not the DINO-video teacher);
5. the single documented CLI shape — one quoted JSON object passed to
   ``--policy.align_params='...'`` — parses into the dict field and reaches
   the same validation.

Everything here is pure CPU: no teachers, no Hub access, no GPU, no dataset.
"""

import copy
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _make_config_cls():
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

    return LingbotVLAV2Config


# The official RoboTwin "Native Depth" recipe (upstream
# configs/vla/robotwin/robotwin.yaml align_params block), with local teacher
# paths. Passing this through config construction is the port's contract.
ROBOTWIN_ALIGN_PARAMS = {
    "mode": "query",
    "num_task_tokens": 8,
    "depth_loss_weight": 0.004,
    "future_depth_loss_weight": 0.004,
    "use_future_video": True,
    "llm": {"dim_out": 2560, "image_token_size": 8, "image_input_size": 224},
    "depth": {
        "model_type": "MoRGBD",
        "moge_path": "/tmp/teachers/moge/model.pt",
        "morgbd_path": "/tmp/teachers/depth/model.pt",
        "num_layers": 1,
        "num_heads": 4,
        "dim_head": 32,
        "ff_mult": 1,
        "num_backbone_tokens": 256,
        "token_size": 16,
        "dim_out": 1024,
        "input_size": 224,
        "use_future_depth": True,
        "block_future_depth_to_action": True,
        "detach_future_image_feats": True,
    },
    "video": {
        "ckpt_path": "/tmp/teachers/dino_video/teacher_step_10000.pth",
        "config_path": "/tmp/teachers/dino_video/config.yaml",
        "attention_mode": "flex_block_causal",
        "input_size": 256,
        "num_future_frames": 1,
        "use_warmup_frame": True,
        "effective_fps": 1.0,
        "share_future_depth_query": True,
        "use_shared_future_task_proj": True,
        "use_current_shared_task_proj": True,
        "use_patch_loss": True,
        "use_current_patch_loss": True,
        "use_cls_loss": False,
        "use_smooth_l1_loss": False,
        "use_mse_loss": True,
        "mse_loss_weight": 1.0,
        "use_cosine_loss": True,
        "cosine_loss_weight": 0.2,
        "future_video_loss_weight": 0.004,
        "num_layers": 1,
        "num_heads": 4,
        "dim_head": 32,
        "ff_mult": 1,
        "num_backbone_tokens": 256,
        "dim_out": 1024,
    },
    "visual_steps": 5000,
}


def _bad(**overrides):
    """A copy of the official recipe with top-level overrides applied."""
    params = copy.deepcopy(ROBOTWIN_ALIGN_PARAMS)
    params.update(overrides)
    return params


def _bad_depth(**overrides):
    params = copy.deepcopy(ROBOTWIN_ALIGN_PARAMS)
    params["depth"].update(overrides)
    return params


def _bad_video(**overrides):
    params = copy.deepcopy(ROBOTWIN_ALIGN_PARAMS)
    params["video"].update(overrides)
    return params


def test_default_action_only_config_remains_supported():
    config_cls = _make_config_cls()
    config = config_cls()
    assert config.align_params == {}
    assert config.enable_expert_vision is False


def test_empty_align_params_dict_is_still_action_only():
    """Automation layers that always pass the flag must not be locked out."""
    config_cls = _make_config_cls()
    config = config_cls(align_params={})
    assert config.align_params == {}


def test_official_robotwin_align_params_passes_validation():
    """The complete official recipe must survive config construction — this is
    the config the depth/DINO-video training path is opened with."""
    config_cls = _make_config_cls()
    config = config_cls(align_params=copy.deepcopy(ROBOTWIN_ALIGN_PARAMS))
    assert config.align_params["mode"] == "query"
    assert config.align_params["depth"]["model_type"] == "MoRGBD"


def test_depth_only_align_params_passes_validation():
    """use_future_video=False with no video block at all is the minimal depth-only branch."""
    config_cls = _make_config_cls()
    params = _bad(use_future_video=False)
    params.pop("video")
    config = config_cls(align_params=params)
    assert config.align_params["use_future_video"] is False


def test_future_image_sampling_uses_the_upstream_action_horizon():
    """RoboTwin samples current/future images at [0, chunk_size - 1]."""
    config_cls = _make_config_cls()
    config = config_cls(align_params=copy.deepcopy(ROBOTWIN_ALIGN_PARAMS), chunk_size=50)
    assert config.use_depth_align is True
    assert config.use_future_image is True
    assert config.observation_delta_indices == [0, 49]


def test_explicit_future_frame_offset_overrides_the_default_horizon():
    config_cls = _make_config_cls()
    config = config_cls(
        align_params=copy.deepcopy(ROBOTWIN_ALIGN_PARAMS), chunk_size=50, future_frame_offset=12
    )
    assert config.observation_delta_indices == [0, 12]


def test_missing_required_top_key_is_rejected():
    config_cls = _make_config_cls()
    params = _bad(depth_loss_weight=0.004)
    del params["depth_loss_weight"]
    with pytest.raises(ValueError, match="align_params.*missing required keys.*depth_loss_weight"):
        config_cls(align_params=params)


def test_missing_required_head_key_is_rejected():
    config_cls = _make_config_cls()
    params = _bad_depth()
    del params["depth"]["dim_head"]
    with pytest.raises(ValueError, match="align_params.depth.*missing required keys.*dim_head"):
        config_cls(align_params=params)


def test_non_query_mode_is_rejected():
    config_cls = _make_config_cls()
    with pytest.raises(ValueError, match="align_params.mode must be 'query'"):
        config_cls(align_params=_bad(mode="linear"))


def test_non_morgbd_model_type_is_rejected():
    config_cls = _make_config_cls()
    with pytest.raises(ValueError, match="align_params.depth.model_type must be 'MoRGBD'"):
        config_cls(align_params=_bad_depth(model_type="DepthAnything"))


def test_indivisible_backbone_tokens_is_rejected():
    config_cls = _make_config_cls()
    with pytest.raises(ValueError, match="num_backbone_tokens.*divisible"):
        config_cls(align_params=_bad_depth(num_backbone_tokens=255))


def test_future_video_requires_full_video_block():
    config_cls = _make_config_cls()
    params = _bad()
    params.pop("video")
    with pytest.raises(ValueError, match="align_params.video.*missing required keys"):
        config_cls(align_params=params)


def test_shared_query_requires_future_depth():
    config_cls = _make_config_cls()
    params = _bad_depth(use_future_depth=False)
    with pytest.raises(ValueError, match="share_future_depth_query=True requires"):
        config_cls(align_params=params)


def test_shared_task_proj_requires_shared_query():
    config_cls = _make_config_cls()
    with pytest.raises(ValueError, match="use_shared_future_task_proj=True requires"):
        config_cls(align_params=_bad_video(share_future_depth_query=False))


def test_expert_vision_rejected_as_distinct_unimplemented_path():
    config_cls = _make_config_cls()
    with pytest.raises(NotImplementedError) as excinfo:
        config_cls(enable_expert_vision=True, expert_vision_type="dinov3_vitb16")
    message = str(excinfo.value)
    assert "enable_expert_vision" in message
    assert "not available" in message
    # Must steer developers away from conflating it with the DINO-video teacher.
    assert "NOT the DINO-video" in message


def test_toplevel_use_depth_field_does_not_trigger_validation():
    """use_depth is a compatibility field the model never reads; the branch is
    keyed on align_params (matching the model's actual behavior)."""
    config_cls = _make_config_cls()
    config = config_cls(use_depth=True, num_task_tokens=8)
    assert config.use_depth is True


def test_json_dict_cli_shape_reaches_validation():
    """The one documented CLI form: a single quoted JSON object. ``lerobot-train``
    reaches ``LingbotVLAV2Config`` through draccus with the CLI argv, so the
    assertion has two parts:

    1. the mechanism itself — a plain-dict dataclass field receives a real dict
       from the JSON string (a minimal probe isolates this from the policy);
    2. the policy config hit through the same argv shape reaches align_params
       schema validation (here: a malformed dict is rejected during parsing).
    """
    draccus = pytest.importorskip("draccus")
    config_cls = _make_config_cls()
    argv = ['--align_params={"mode":"query","num_task_tokens":8,"depth":{"model_type":"MoRGBD"}}']

    @dataclass
    class _Probe:
        align_params: dict = field(default_factory=dict)

    probe = draccus.parse(_Probe, args=argv)
    assert probe.align_params == {"mode": "query", "num_task_tokens": 8, "depth": {"model_type": "MoRGBD"}}

    # draccus re-raises config __post_init__ errors wrapped in its ParsingError.
    with pytest.raises(Exception) as excinfo:
        draccus.parse(config_cls, args=argv)
    cause = excinfo.value.__cause__ or excinfo.value
    assert "align_params" in str(cause)
