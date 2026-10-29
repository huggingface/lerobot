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

"""End-to-end tests of the Cosmos3 policies with tiny backbones: training, inference, save and load."""

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.cosmos3.configuration_cosmos3 import (  # noqa: E402
    Nemotron3DenseVLMoTConfig,
    Qwen3VLMoTConfig,
    Wan2pt2VAEConfig,
)
from lerobot.policies.cosmos3.processor_cosmos3 import Cosmos3TokenizerProcessorStep  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    get_policy_class,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import DEVICE  # noqa: E402

BATCH_SIZE = 2
CHUNK_SIZE = 4
IMAGE_SIZE = 32
TASKS = ["pick up the black bowl", "open the top drawer and put the bowl inside"]
# Two layers of the released text backbones, with their vocabulary and special tokens for the real tokenizers.
TEXT_CONFIG = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
}

# One case per policy type: Cosmos3 Nano with the LIBERO layout, Cosmos3 Edge with the DROID layout (state
# action, idle frames over the state, classifier-free guidance).
CASES = [
    pytest.param(
        {
            "policy_type": "cosmos3_nano",
            "camera_keys": ["observation.images.image", "observation.images.wrist_image"],
            "action_names": [*(f"pos_{i}" for i in range(3)), *(f"rot6d_{i}" for i in range(6)), "gripper_0"],
            "recipe": {},
        },
        id="nano_libero",
    ),
    pytest.param(
        {
            "policy_type": "cosmos3_edge",
            "camera_keys": [
                "observation.images.wrist",
                "observation.images.exterior_1",
                "observation.images.exterior_2",
            ],
            "action_names": [*(f"joint_{i}" for i in range(7)), "gripper_0"],
            "recipe": {"use_state": True, "idle_state": True, "guidance_scale": 2.0},
        },
        id="edge_droid",
    ),
]


def make_mot_backbone(policy_type: str) -> Qwen3VLMoTConfig | Nemotron3DenseVLMoTConfig:
    if policy_type == "cosmos3_nano":
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": 5_000_000,
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        }
        return Qwen3VLMoTConfig(
            text_config={**TEXT_CONFIG, "rope_parameters": rope_parameters},
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 64,
                "deepstack_visual_indexes": [0],
            },
        )
    return Nemotron3DenseVLMoTConfig(
        text_config=TEXT_CONFIG,
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
        },
        projector_config={"input_hidden_size": 32, "merger_intermediate_size": 64, "out_hidden_size": 64},
    )


def make_policy_and_processors(case: dict):
    action_dim = len(case["action_names"])
    config = make_policy_config(
        case["policy_type"],
        mot_backbone=make_mot_backbone(case["policy_type"]),
        vae=Wan2pt2VAEConfig(
            base_dim=16,
            decoder_base_dim=16,
            z_dim=8,
            num_res_blocks=1,
            latents_mean=[0.0] * 8,
            latents_std=[1.0] * 8,
        ),
        device=DEVICE,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        num_inference_steps=2,
        max_action_dim=16,
        fps=15.0,
        action_feature_names=case["action_names"],
        **case["recipe"],
    )
    config.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
        for key in case["camera_keys"]
    }
    if config.use_state:
        config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(action_dim,))
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    policy = get_policy_class(case["policy_type"])(config).to(DEVICE)
    dataset_stats = {ACTION: {"q01": -torch.ones(action_dim), "q99": torch.ones(action_dim)}}
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)
    return policy, preprocessor, postprocessor


def make_batch(case: dict, training: bool) -> dict:
    """Raw training batch (clips of `CHUNK_SIZE + 1` frames and actions) or observation (one frame)."""
    action_dim = len(case["action_names"])
    frames = (CHUNK_SIZE + 1,) if training else ()
    batch = {key: torch.rand(BATCH_SIZE, *frames, 3, IMAGE_SIZE, IMAGE_SIZE) for key in case["camera_keys"]}
    batch[OBS_STATE] = torch.randn(BATCH_SIZE, action_dim)
    batch["task"] = TASKS
    if training:
        batch[ACTION] = torch.randn(BATCH_SIZE, CHUNK_SIZE, action_dim)
    return batch


@pytest.mark.parametrize("case", CASES)
def test_cosmos3_training_step(case):
    set_seed(0)
    policy, preprocessor, _ = make_policy_and_processors(case)
    policy.train()

    loss, losses = policy.forward(preprocessor(make_batch(case, training=True)))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(losses) == {"video_loss", "action_loss"}

    loss.backward()
    params = [param for param in policy.parameters() if param.requires_grad]
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in params)

    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    before = [param.detach().clone() for param in params]
    optimizer.step()
    assert all(not torch.equal(old, param) for old, param in zip(before, params, strict=True))


@pytest.mark.parametrize("case", CASES)
def test_cosmos3_inference(case):
    set_seed(0)
    policy, preprocessor, postprocessor = make_policy_and_processors(case)
    observation = make_batch(case, training=False)
    action_dim = len(case["action_names"])

    torch.manual_seed(1)
    chunk = policy.predict_action_chunk(preprocessor(observation))
    assert chunk.shape == (BATCH_SIZE, CHUNK_SIZE, action_dim)

    # `select_action` samples one chunk and returns its actions one by one.
    policy.reset()
    torch.manual_seed(1)
    actions = [policy.select_action(preprocessor(observation)) for _ in range(CHUNK_SIZE)]
    torch.testing.assert_close(torch.stack(actions, dim=1), chunk, rtol=0, atol=0)

    action = postprocessor(actions[0])
    assert action.shape == (BATCH_SIZE, action_dim)
    assert torch.isfinite(action).all()


@pytest.mark.parametrize("case", CASES)
def test_cosmos3_save_and_load(case, tmp_path):
    set_seed(0)
    policy, preprocessor, postprocessor = make_policy_and_processors(case)
    policy.save_pretrained(tmp_path)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_policy = get_policy_class(case["policy_type"]).from_pretrained(tmp_path)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        loaded_policy.config, pretrained_path=tmp_path
    )
    torch.testing.assert_close(loaded_policy.state_dict(), policy.state_dict(), rtol=0, atol=0)
    tokenizer_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, Cosmos3TokenizerProcessorStep)
    )
    assert tokenizer_step.tokenizer_name == str(tmp_path / "tokenizer")

    observation = make_batch(case, training=False)
    torch.manual_seed(1)
    action = postprocessor(policy.select_action(preprocessor(observation)))
    torch.manual_seed(1)
    loaded_action = loaded_postprocessor(loaded_policy.select_action(loaded_preprocessor(observation)))
    torch.testing.assert_close(loaded_action, action, rtol=0, atol=0)
