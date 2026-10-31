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
"""A camera missing from the batch must not shift the other cameras out of their slots.

Each camera is filled with its own constant so that every slot can be traced back to the camera
it came from. Missing cameras are expected as a padded slot (masked out) at their own position.
"""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import OBS_IMAGES

FRONT, WRIST, SIDE = (f"{OBS_IMAGES}.{name}" for name in ("front", "wrist", "side"))
CAMERAS = [FRONT, WRIST, SIDE]
VALUES = {FRONT: 0.2, WRIST: 0.4, SIDE: 0.6}
BATCH_SIZE, SIZE = 2, 8


def make_batch(missing):
    return {
        key: torch.full((BATCH_SIZE, 3, SIZE, SIZE), VALUES[key]) for key in CAMERAS if key not in missing
    }


def run_paligemma(policy_cls, batch):
    policy = SimpleNamespace(
        config=SimpleNamespace(image_features=dict.fromkeys(CAMERAS), image_resolution=(SIZE, SIZE)),
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return policy_cls._preprocess_images(policy, batch)


def run_smolvla(batch, empty_cameras=len(CAMERAS)):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features=dict.fromkeys(CAMERAS), resize_imgs_with_padding=None, empty_cameras=empty_cameras
        )
    )
    return SmolVLAPolicy.prepare_images(policy, batch)


def run_xvla(batch):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features=dict.fromkeys(CAMERAS), resize_imgs_with_padding=None, num_image_views=len(CAMERAS)
        )
    )
    images, masks = XVLAPolicy._prepare_images(policy, batch)
    return list(images.unbind(dim=1)), list(masks.unbind(dim=1))


def run_evo1(batch):
    policy = SimpleNamespace(
        _camera_keys=list(CAMERAS), config=SimpleNamespace(max_views=len(CAMERAS), empty_cameras=len(CAMERAS))
    )
    images, masks = Evo1Policy._collect_image_batches(policy, batch)
    return images, list(masks.unbind(dim=1))


# How each policy scales a camera value, and the value it pads a missing camera with.
POLICIES = {
    "pi0": (lambda batch: run_paligemma(PI0Policy, batch), lambda v: v * 2 - 1, -1.0),
    "pi05": (lambda batch: run_paligemma(PI05Policy, batch), lambda v: v * 2 - 1, -1.0),
    "pi0_fast": (lambda batch: run_paligemma(PI0FastPolicy, batch), lambda v: v * 2 - 1, -1.0),
    "smolvla": (run_smolvla, lambda v: v * 2 - 1, -1.0),
    "xvla": (run_xvla, lambda v: v, 0.0),
    "evo1": (run_evo1, lambda v: v, 0.0),
}


def assert_slot(image, mask, value, present):
    torch.testing.assert_close(image, torch.full_like(image, value))
    assert mask.tolist() == [present] * BATCH_SIZE


@pytest.mark.parametrize("missing", [FRONT, WRIST], ids=["first", "middle"])
@pytest.mark.parametrize("name", list(POLICIES))
def test_missing_camera_keeps_other_cameras_in_their_slots(name, missing):
    run, scale, pad = POLICIES[name]

    images, masks = run(make_batch(missing=[missing]))

    assert len(images) == len(masks) == len(CAMERAS)
    for slot, key in enumerate(CAMERAS):
        if key == missing:
            assert_slot(images[slot], masks[slot], pad, present=False)
        else:
            assert_slot(images[slot], masks[slot], scale(VALUES[key]), present=True)


@pytest.mark.parametrize("name", list(POLICIES))
def test_slots_follow_image_features_not_batch_key_order(name):
    run, scale, pad = POLICIES[name]

    images, masks = run(dict(reversed(make_batch(missing=[WRIST]).items())))

    assert_slot(images[0], masks[0], scale(VALUES[FRONT]), present=True)
    assert_slot(images[1], masks[1], pad, present=False)
    assert_slot(images[2], masks[2], scale(VALUES[SIDE]), present=True)


@pytest.mark.parametrize("name", list(POLICIES))
def test_missing_last_camera_is_padded_at_the_end(name):
    run, scale, pad = POLICIES[name]

    images, masks = run(make_batch(missing=[SIDE]))

    assert_slot(images[0], masks[0], scale(VALUES[FRONT]), present=True)
    assert_slot(images[1], masks[1], scale(VALUES[WRIST]), present=True)
    assert masks[2].tolist() == [False] * BATCH_SIZE
    if name == "evo1":
        # Trailing absent views are left to the embedder, which zero-pads them.
        assert len(images) == 2
    else:
        assert_slot(images[2], masks[2], pad, present=False)


def test_paligemma_keeps_slots_with_several_missing_cameras():
    images, masks = run_paligemma(PI05Policy, make_batch(missing=[FRONT, SIDE]))

    assert_slot(images[0], masks[0], -1.0, present=False)
    assert_slot(images[1], masks[1], VALUES[WRIST] * 2 - 1, present=True)
    assert_slot(images[2], masks[2], -1.0, present=False)


def test_pi05_keeps_slots_for_temporal_inputs():
    # With proprioceptive memory, cameras come as [B, T, C, H, W] and masks as [B, T].
    frames = 4
    batch = {
        key: torch.full((BATCH_SIZE, frames, 3, SIZE, SIZE), VALUES[key]) for key in CAMERAS if key != FRONT
    }
    batch |= {f"{key}_is_pad": torch.zeros(BATCH_SIZE, frames, dtype=torch.bool) for key in (WRIST, SIDE)}

    images, masks = run_paligemma(PI05Policy, batch)

    expected = [(-1.0, False), (VALUES[WRIST] * 2 - 1, True), (VALUES[SIDE] * 2 - 1, True)]
    for image, mask, (value, present) in zip(images, masks, expected, strict=True):
        assert image.shape == (BATCH_SIZE, frames, 3, SIZE, SIZE)
        torch.testing.assert_close(image, torch.full_like(image, value))
        assert mask.shape == (BATCH_SIZE, frames)
        assert bool(mask.all()) if present else not bool(mask.any())


def test_smolvla_drops_missing_cameras_beyond_empty_cameras_budget():
    # Without a budget a missing camera is dropped, as before.
    images, masks = run_smolvla(make_batch(missing=[FRONT]), empty_cameras=0)
    assert len(images) == 2
    assert_slot(images[0], masks[0], VALUES[WRIST] * 2 - 1, present=True)
    assert_slot(images[1], masks[1], VALUES[SIDE] * 2 - 1, present=True)

    # With a budget of one, the first missing camera keeps its slot and the second one is dropped.
    images, masks = run_smolvla(make_batch(missing=[FRONT, WRIST]), empty_cameras=1)
    assert len(images) == 2
    assert_slot(images[0], masks[0], -1.0, present=False)
    assert_slot(images[1], masks[1], VALUES[SIDE] * 2 - 1, present=True)
