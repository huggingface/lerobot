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

import pytest
import torch

transformers = pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


@pytest.fixture
def small_policy(monkeypatch):
    """Keep the real vision, prefix attention, and action expert at a small CPU-test size."""
    paligemma_cls = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_paligemma(config):
        config.vision_config.projection_dim = 32
        return paligemma_cls(config)

    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma)
    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda variant: modeling_pi05.GemmaConfig(
            width=32, depth=2, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=4
        ),
    )
    monkeypatch.setattr(
        modeling_pi05,
        "CONFIG_MAPPING",
        {
            "paligemma": lambda: transformers.PaliGemmaConfig(
                vision_config={
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "patch_size": 8,
                }
            ),
            "gemma": transformers.GemmaConfig,
        },
    )
    config = PI05Config(
        device="cpu",
        image_resolution=(16, 16),
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        num_inference_steps=2,
        empty_cameras=1,
        memory_temporal_attention_every=2,
        input_features={
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))
            for key in ["observation.images.base", "observation.images.wrist"]
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    torch.manual_seed(0)
    return modeling_pi05.PI05Policy(config)


def make_batch(*, temporal=False, cameras=2):
    image_shape = (2, 3, 3, 16, 16) if temporal else (2, 3, 16, 16)
    batch = {
        "observation.images.base": torch.rand(image_shape),
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True, False], [True, True, False, False]]),
        ACTION: torch.randn(2, 3, 4),
    }
    if cameras >= 2:
        # An actual black camera frame must still be encoded.
        batch["observation.images.wrist"] = torch.zeros(image_shape)
    if cameras == 3:
        batch["observation.images.empty_camera_0"] = torch.rand(image_shape)
    if temporal:
        for key in [
            "observation.images.base",
            "observation.images.wrist",
            "observation.images.empty_camera_0",
        ][:cameras]:
            batch[f"{key}_is_pad"] = torch.tensor([[True, False, False], [True, True, False]])
    return batch


@pytest.mark.parametrize("cameras", [1, 2, 3])
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("compile_model", [False, True])
def test_inference_skips_missing_camera_encoders_without_changing_actions(
    small_policy, cameras, temporal, compile_model
):
    policy = small_policy.eval()
    # Exercise input selection without compiling the CPU-sized model in this unit test.
    policy.config.compile_model = compile_model
    batch = make_batch(temporal=temporal, cameras=cameras)
    noise = torch.randn(2, 3, 4)
    images, masks = policy._preprocess_images(batch)
    with torch.no_grad():
        expected = policy.model.sample_actions(
            images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], noise=noise
        )

    vision_calls = []
    hook = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector.register_forward_pre_hook(
        lambda module, args: vision_calls.append(1)
    )
    try:
        actual = policy.predict_action_chunk(batch, noise=noise)
    finally:
        hook.remove()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    expected_cameras = len(images) if compile_model else cameras
    assert len(vision_calls) == expected_cameras
    actual_images, actual_masks = policy._preprocess_images(batch, encode_missing_cameras=compile_model)
    assert len(actual_images) == len(images)
    assert sum(img is not None for img in actual_images) == expected_cameras
    for actual_mask, expected_mask in zip(actual_masks, masks, strict=True):
        torch.testing.assert_close(actual_mask, expected_mask)


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_missing_cameras_preserve_rtc_actions(small_policy, mode):
    policy = small_policy.eval()
    policy.config.rtc_config = RTCConfig(mode=mode, execution_horizon=2)
    policy.config.rtc_training_max_delay = 1
    policy.init_rtc_processor()
    batch = make_batch(cameras=1)
    kwargs = {
        "noise": torch.randn(2, 3, 4),
        "prev_chunk_left_over": torch.randn(2, 2, 4),
        "inference_delay": 1,
        "execution_horizon": 2,
    }
    images, masks = policy._preprocess_images(batch)
    with torch.no_grad():
        expected = policy.model.sample_actions(
            images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], **kwargs
        )
    actual = policy.predict_action_chunk(batch, **kwargs)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("training", [False, True])
def test_loss_forward_keeps_camera_padding(small_policy, training):
    policy = small_policy.train(training)
    batch = make_batch(cameras=1)
    vision_calls = []
    hook = policy.model.paligemma_with_expert.paligemma.model.vision_tower.register_forward_pre_hook(
        lambda module, args: vision_calls.append(1)
    )
    try:
        loss, _ = policy(batch)
        loss.backward()
    finally:
        hook.remove()
    assert len(vision_calls) == 3
    assert torch.isfinite(loss)
    assert torch.isfinite(policy.model.action_out_proj.weight.grad).all()
