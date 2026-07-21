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

"""Tests for the VLA-JEPA image-prep processor step and its back-compat contract with the model.

The step moves image resize + 1->3 channel-expand out of the model into the (serialized)
preprocessor. The model keeps the same ops as idempotent guards, so:
  - old checkpoints (JSON without the step) are unaffected — the model still does the prep;
  - new checkpoints (JSON with the step) get it done in the step, and the model guards no-op.
These tests pin the step's numerics (bit-identical to the model's F.interpolate(area)) and the
equivalence of the two paths on the Qwen image path.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from conftest import (  # noqa: E402
    BATCH_SIZE,
    IMAGE_SIZE,
    make_config,
    make_inference_batch,
    make_train_batch,
)

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402
from lerobot.policies.vla_jepa.processor_vla_jepa import (  # noqa: E402
    ImagePrepProcessorStep,
    make_vla_jepa_pre_post_processors,
)
from lerobot.processor import ProcessorStepRegistry  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

RESIZE = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)  # (4, 4)
IMG_KEY = f"{OBS_IMAGES}.laptop"


# ---------------------------------------------------------------------------
# Step numerics / shape handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3, IMAGE_SIZE, IMAGE_SIZE),  # [C, H, W] (raw single-sample inference)
        (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),  # [B, C, H, W]
        (BATCH_SIZE, 2, 3, IMAGE_SIZE, IMAGE_SIZE),  # [B, T, C, H, W] (video stack)
    ],
)
def test_image_prep_resize_shapes_and_area_numerics(shape: tuple[int, ...]) -> None:
    step = ImagePrepProcessorStep(resize_to=RESIZE)
    x = torch.rand(*shape)
    out = step.observation({IMG_KEY: x})[IMG_KEY]

    assert out.shape[:-2] == x.shape[:-2]  # leading + channel dims unchanged
    assert tuple(out.shape[-2:]) == RESIZE
    assert out.dtype == torch.float32

    # bit-identical to the model-side F.interpolate(mode="area"), no clamp
    ref = F.interpolate(x.float().reshape(-1, *x.shape[-3:]), size=RESIZE, mode="area").reshape(
        *x.shape[:-2], *RESIZE
    )
    assert torch.equal(out, ref)


def test_image_prep_channel_expand() -> None:
    step = ImagePrepProcessorStep(resize_to=None, expand_channels=True)
    x = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    out = step.observation({IMG_KEY: x})[IMG_KEY]
    assert out.shape[1] == 3
    # all three channels are copies of the single input channel
    assert torch.equal(out[:, 0], x[:, 0]) and torch.equal(out[:, 1], x[:, 0])


def test_image_prep_resize_skip_when_already_target_size() -> None:
    step = ImagePrepProcessorStep(resize_to=RESIZE)
    x = torch.rand(BATCH_SIZE, 3, *RESIZE)
    out = step.observation({IMG_KEY: x})[IMG_KEY]
    # size already matches -> only the float cast happens, values preserved exactly
    assert torch.equal(out, x)


def test_image_prep_leaves_non_image_keys_untouched() -> None:
    step = ImagePrepProcessorStep(resize_to=RESIZE)
    state = torch.randn(BATCH_SIZE, 4)
    out = step.observation({IMG_KEY: torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), OBS_STATE: state})
    assert torch.equal(out[OBS_STATE], state)


def test_image_prep_config_roundtrip_via_registry() -> None:
    step = ImagePrepProcessorStep(resize_to=RESIZE, expand_channels=True)
    cfg = step.get_config()
    assert cfg == {"resize_to": [RESIZE[0], RESIZE[1]], "expand_channels": True}
    rebuilt = ProcessorStepRegistry.get("vla_jepa_image_prep")(**cfg)
    assert rebuilt.resize_to == RESIZE
    assert rebuilt.expand_channels is True


def test_image_prep_transform_features() -> None:
    step = ImagePrepProcessorStep(resize_to=RESIZE, expand_channels=True)
    features = {
        PipelineFeatureType.OBSERVATION: {
            IMG_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            "observation.images.depth": PolicyFeature(
                type=FeatureType.VISUAL, shape=(1, IMAGE_SIZE, IMAGE_SIZE)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        }
    }
    out = step.transform_features(features)[PipelineFeatureType.OBSERVATION]
    assert out[IMG_KEY].shape == (3, *RESIZE)  # already 3-channel, only resized
    assert out["observation.images.depth"].shape == (3, *RESIZE)  # 1->3 expanded
    assert out[OBS_STATE].shape == (4,)  # non-image untouched


# ---------------------------------------------------------------------------
# Pipeline wiring + back-compat with the model
# ---------------------------------------------------------------------------


def test_image_prep_step_wired_into_preprocessor() -> None:
    cfg = make_config()
    cfg.resize_images_to = RESIZE
    preprocessor, _ = make_vla_jepa_pre_post_processors(cfg, dataset_stats=None)
    prep_steps = [s for s in preprocessor.steps if isinstance(s, ImagePrepProcessorStep)]
    assert len(prep_steps) == 1
    assert prep_steps[0].resize_to == RESIZE


@torch.no_grad()
@pytest.mark.parametrize("batch_fn", [make_inference_batch, make_train_batch])
def test_image_prep_matches_model_qwen_path(patch_vla_jepa_external_models: None, batch_fn) -> None:
    """The Qwen image path is identical whether the step resized (new ckpt) or the model does (old ckpt).

    Both use F.interpolate(mode="area"), so pre-resizing in the step then letting the model's
    size guard no-op yields byte-identical Qwen inputs to the pure model path. This is the
    contract that keeps already-uploaded checkpoints correct.
    """
    cfg = make_config()
    cfg.resize_images_to = RESIZE
    policy = VLAJEPAPolicy(cfg)
    policy.eval()
    training = batch_fn is make_train_batch

    batch = batch_fn()

    # Path A (old checkpoint, no processor step): the model resizes internally.
    imgs_a = policy._prepare_model_inputs(deepcopy(batch), training=training)["images"]

    # Path B (new checkpoint): the step resizes first; the model's guard becomes a no-op.
    step = ImagePrepProcessorStep(resize_to=RESIZE)
    resized = step.observation({IMG_KEY: batch[IMG_KEY]})
    batch_b = deepcopy(batch)
    batch_b[IMG_KEY] = resized[IMG_KEY]
    imgs_b = policy._prepare_model_inputs(batch_b, training=training)["images"]

    assert len(imgs_a) == len(imgs_b) == BATCH_SIZE
    for views_a, views_b in zip(imgs_a, imgs_b, strict=True):
        for a, b in zip(views_a, views_b, strict=True):
            assert torch.equal(a, b)
