#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import OBS_IMAGES, OBS_LANGUAGE, REWARD
from lerobot.policies.factory import make_processor
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig
from lerobot.policies.rlearn.modeling_rlearn import RLearNPolicy
from tests.utils import require_package


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_instantiation_and_forward_tensor_batch():
    """Instantiate RLearN and run a forward pass with a (B, T, C, H, W) tensor input using a real model and real text."""
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        push_to_hub=False,
        freeze_backbones=True,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    policy = RLearNPolicy(cfg)

    B, T, C, H, W = 2, 3, 3, 256, 256
    batch = {
        OBS_IMAGES: torch.rand(B, T, C, H, W),
        REWARD: torch.randint(low=0, high=1, size=(B, T)).float(),
        OBS_LANGUAGE: ["move the green cube into the box" for _ in range(B)],
    }

    loss, logs = policy.forward(batch)
    assert isinstance(loss, torch.Tensor)
    assert "loss" in logs


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_instantiation_and_forward_list_batch_with_language():
    """Instantiate RLearN and run a forward pass with a list-of-frames input and real language using a real model."""
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        push_to_hub=False,
        freeze_backbones=True,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    policy = RLearNPolicy(cfg)

    B, T, C, H, W = 2, 4, 3, 256, 256
    frames = [torch.rand(B, C, H, W) for _ in range(T)]
    batch = {
        OBS_IMAGES: frames,  # list[(B, C, H, W)]
        REWARD: torch.randint(low=0, high=2, size=(B, T)).float(),
        OBS_LANGUAGE: ["move the red cube into the box" for _ in range(B)],
    }

    loss, logs = policy.forward(batch)
    assert isinstance(loss, torch.Tensor)
    assert "loss" in logs


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_composite_loss_shapes_and_terms():
    """Smoke test composite loss: checks presence of terms and valid gradients."""
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        push_to_hub=False,
        freeze_backbones=True,
        use_video_rewind=True,
        rewind_prob=0.5,
        use_mismatch_loss=True,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    policy = RLearNPolicy(cfg)

    B, T, C, H, W = 2, 3, 3, 256, 256
    # Progress labels y in [0,1]
    y = torch.linspace(0, 1, T).unsqueeze(0).repeat(B, 1)
    batch = {
        OBS_IMAGES: torch.rand(B, T, C, H, W),
        REWARD: y.clone(),
        OBS_LANGUAGE: ["stack the blocks" for _ in range(B)],
    }

    loss, logs = policy.forward(batch)
    assert isinstance(loss, torch.Tensor) and torch.isfinite(loss)
    # Expect ReWiND loss terms (progress and mismatch)
    assert "loss_progress" in logs
    assert "loss_mismatch" in logs


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_preprocessor_tokenizes_and_copies_task():
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        device="cpu",
        push_to_hub=False,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    pre, post = make_processor(cfg, dataset_stats=None)

    B, C, H, W = 2, 3, 64, 64
    batch = {
        "observation.image": torch.rand(B, C, H, W),
        REWARD: torch.zeros(B),
        "task": ["pick the cube", "place it in the box"],
    }

    processed = pre(batch)

    assert isinstance(processed, dict)
    assert f"{OBS_LANGUAGE}.tokens" in processed
    assert f"{OBS_LANGUAGE}.attention_mask" in processed
    assert OBS_LANGUAGE in processed

    tokens = processed[f"{OBS_LANGUAGE}.tokens"]
    attn = processed[f"{OBS_LANGUAGE}.attention_mask"]
    assert tokens.dim() == 2 and attn.dim() == 2
    assert tokens.shape[0] == B and attn.shape[0] == B


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_preprocessor_string_task_and_to_batch():
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        device="cpu",
        push_to_hub=False,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    pre, post = make_processor(cfg, dataset_stats=None)

    # Unbatched image and single string task
    batch = {
        "observation.image": torch.rand(3, 64, 64),
        REWARD: torch.tensor(0.0),
        "task": "move the green cube into the box",
    }

    processed = pre(batch)

    # Image should have batch dim now
    assert processed["observation.image"].dim() == 4 and processed["observation.image"].shape[0] == 1
    # Language copy and tokenization should exist
    assert OBS_LANGUAGE in processed and isinstance(processed[OBS_LANGUAGE], list)
    assert f"{OBS_LANGUAGE}.tokens" in processed
    assert f"{OBS_LANGUAGE}.attention_mask" in processed


@require_package("transformers")
@require_package("sentence_transformers")
def test_rlearn_pipeline_end_to_end_forward():
    """End-to-end: preprocessor + model forward using RLearN pipeline on synthetic data."""
    cfg = RLearNConfig(
        vision_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        text_model_name="sentence-transformers/all-MiniLM-L12-v2",
        device="cpu",
        push_to_hub=False,
        freeze_backbones=True,
        use_video_rewind=True,
    )
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }

    # Build processors and model
    pre, post = make_processor(cfg, dataset_stats=None)
    policy = RLearNPolicy(cfg)

    B, T, C, H, W = 2, 3, 3, 256, 256
    y = torch.linspace(0, 1, T).unsqueeze(0).repeat(B, 1)
    raw = {
        # Provide as observation.image to let preprocessor map/normalize and batch
        "observation.image": torch.rand(B, C, H, W),  # not time-major to test ToBatch
        REWARD: y[:, :1].clone(),  # single step label; pipeline keeps structure
        "task": ["insert the peg", "insert the peg"],
    }

    processed = pre(raw)
    # Integrate preprocessor output with model forward
    loss, logs = policy.forward(
        {
            OBS_IMAGES: processed.get(OBS_IMAGES, processed.get("observation.image"))
            .unsqueeze(1)
            .repeat(1, T, 1, 1, 1),
            REWARD: y.clone(),
            OBS_LANGUAGE: processed[OBS_LANGUAGE],
        }
    )
    assert isinstance(loss, torch.Tensor) and torch.isfinite(loss)
