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

"""PI0Fast integration checks and a fixed-token decoding reference.

Generation from synthetic observations is checked for validity and reproducibility,
not exact parity with OpenPI or across Torch versions and GPU attention backends.
"""
# ruff: noqa: E402

import random
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.lerobot_types import PolicyAction  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.policies.pi0_fast.processor_pi0_fast import make_pi0_fast_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)  # noqa: E402
from tests.utils import require_cuda, require_hf_token  # noqa: E402

# Constants
DUMMY_ACTION_DIM = 7
DUMMY_STATE_DIM = 20
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
DEVICE = "cuda"
MODEL_PATH_LEROBOT = "lerobot/pi0fast-base"

# Expected action token shape: (batch_size, max_decoding_steps)
EXPECTED_ACTION_TOKENS_SHAPE = (1, 2)

# Fixed input for the decoding reference, independent of model generation.
REFERENCE_ACTION_TOKENS = torch.tensor([[255020, 255589]])

# Expected actions after detokenization
EXPECTED_ACTIONS_SHAPE = (1, 2, 32)  # (batch_size, n_action_steps, action_dim)
EXPECTED_ACTIONS_FIRST_5 = torch.tensor([0.0000, 0.3536, 0.0707, 0.0000, 0.0000])


def set_seed_all(seed: int) -> None:
    """Set random seed for all RNG sources to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@require_cuda
@require_hf_token
def instantiate_lerobot_pi0_fast(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH_LEROBOT,
) -> tuple[
    Any,  # Policy
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot PI0Fast policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = PI0FastPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=True,
        )
        policy.config.validate_action_token_prefix = False
        policy.config.max_action_tokens = 2
        policy.config.max_decoding_steps = 2
        policy.config.chunk_size = 2
        policy.config.n_action_steps = 2
    else:
        config = PI0FastConfig(
            n_action_steps=2,
            max_action_dim=DUMMY_ACTION_DIM,
            max_state_dim=DUMMY_STATE_DIM,
            device=DEVICE,
            validate_action_token_prefix=False,
            max_action_tokens=2,
            max_decoding_steps=2,
            chunk_size=2,
        )
        policy = PI0FastPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_pi0_fast_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization
    )

    return policy, preprocessor, postprocessor


def create_dummy_data(device: str = DEVICE) -> dict[str, Any]:
    """Create synthetic observations with float32 images in [0, 1]."""
    batch_size = 1
    prompt = "Pick up the red block and place it in the bin"

    # Create random RGB images in [0, 255] uint8 range (as PIL images would be)
    # Then convert to [0, 1] float32 range for LeRobot
    def fake_rgb(h: int, w: int) -> torch.Tensor:
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        return t.to(torch.float32) / 255.0

    batch = {
        f"{OBS_IMAGES}.base_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.left_wrist_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.right_wrist_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


# Pytest fixtures
@pytest.fixture(autouse=True)
def restore_determinism(restore_torch_backend_flags: None) -> Iterator[None]:
    """Keep the module's deterministic settings from leaking into other tests."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture(scope="module")
@require_cuda
@require_hf_token
def pi0_fast_components(
    request: pytest.FixtureRequest,
) -> tuple[
    PI0FastPolicy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Fixture to instantiate and provide all PI0Fast components for tests."""
    print(f"\nTesting with DEVICE='{DEVICE}'")
    print("\n[Setup] Instantiating LeRobot PI0Fast policy...")
    policy_obj, preprocessor_obj, postprocessor_obj = instantiate_lerobot_pi0_fast(from_pretrained=True)

    def release_cuda_model() -> None:
        # Failed assertions can retain the policy through traceback locals. Moving
        # its tensors to CPU frees GPU memory even while those references survive.
        policy_obj.cpu()
        policy_obj.reset()
        torch.cuda.empty_cache()

    request.addfinalizer(release_cuda_model)
    print("Model loaded successfully")
    return policy_obj, preprocessor_obj, postprocessor_obj


@pytest.fixture
@require_cuda
@require_hf_token
def policy(pi0_fast_components: tuple) -> PI0FastPolicy:
    """Fixture to provide the PI0Fast policy for tests."""
    policy_obj = pi0_fast_components[0]
    policy_obj.eval()
    policy_obj.reset()
    return policy_obj


@pytest.fixture(scope="module")
@require_cuda
@require_hf_token
def preprocessor(pi0_fast_components):
    """Fixture to provide the PI0Fast preprocessor for tests."""
    return pi0_fast_components[1]


@require_cuda
@require_hf_token
def test_pi0_fast_preprocessor_alignment(
    policy: PI0FastPolicy, preprocessor: PolicyProcessorPipeline
) -> None:
    """Require language inputs and valid image ranges through preprocessing."""
    set_seed_all(42)
    batch = create_dummy_data()
    observation = preprocessor(deepcopy(batch))

    for key in batch:
        if key.startswith(OBS_IMAGES + "."):
            image = observation[key]
            assert image.dtype == torch.float32
            assert torch.isfinite(image).all()
            assert image.min() >= 0 and image.max() <= 1

    # Use the actual constants: the old literal keys silently skipped these checks.
    tokens = observation[OBS_LANGUAGE_TOKENS]
    masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
    assert tokens.ndim == 2 and tokens.shape[0] == 1
    assert tokens.dtype == torch.long
    assert masks.shape == tokens.shape
    assert masks.dtype == torch.bool
    assert masks.any()

    images, _ = policy._preprocess_images(observation)
    assert images
    for image in images:
        assert torch.isfinite(image).all()
        assert image.min() >= -1 and image.max() <= 1


@require_cuda
@require_hf_token
def test_pi0_fast_action_generation(policy: PI0FastPolicy, preprocessor: PolicyProcessorPipeline) -> None:
    """Exercise generation and decoding without hardware-specific golden tokens."""
    set_seed_all(42)
    observation = preprocessor(create_dummy_data())

    with torch.no_grad():
        actions = policy.predict_action_chunk(observation)

    assert tuple(actions.shape) == EXPECTED_ACTIONS_SHAPE
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all()


@require_cuda
@require_hf_token
def test_pi0_fast_inference_reproducibility(
    policy: PI0FastPolicy, preprocessor: PolicyProcessorPipeline
) -> None:
    """Inference is repeatable for the same observations and seed on one stack."""
    set_seed_all(42)
    batch = create_dummy_data()

    set_seed_all(42)
    with torch.no_grad():
        actions_1 = policy.predict_action_chunk(preprocessor(deepcopy(batch)))

    set_seed_all(42)
    with torch.no_grad():
        actions_2 = policy.predict_action_chunk(preprocessor(deepcopy(batch)))

    torch.testing.assert_close(actions_1, actions_2, rtol=0, atol=1e-6)


@require_cuda
@require_hf_token
def test_pi0_fast_forward_pass_logits(policy: PI0FastPolicy, preprocessor: PolicyProcessorPipeline) -> None:
    """The training forward pass produces a finite, positive cross-entropy loss."""
    set_seed_all(42)
    observation = preprocessor(create_dummy_data())
    observation[ACTION_TOKENS] = torch.randint(
        0, 1000, (1, policy.config.max_action_tokens), dtype=torch.long, device=DEVICE
    )
    observation[ACTION_TOKEN_MASK] = torch.ones_like(observation[ACTION_TOKENS], dtype=torch.bool)

    policy.train()
    try:
        with torch.no_grad():
            loss, _ = policy.forward(observation)
        assert torch.isfinite(loss).all()
        assert loss.item() > 0
    finally:
        policy.eval()


@require_cuda
@require_hf_token
def test_pi0_fast_action_token_sampling(policy: PI0FastPolicy, preprocessor: PolicyProcessorPipeline) -> None:
    """Greedy decoding returns valid vocabulary IDs and is repeatable."""
    set_seed_all(42)
    observation = preprocessor(create_dummy_data())
    images, img_masks = policy._preprocess_images(observation)
    tokens = observation[OBS_LANGUAGE_TOKENS]
    masks = observation[OBS_LANGUAGE_ATTENTION_MASK]

    with torch.no_grad():
        action_tokens = policy.model.sample_actions_fast(
            images, img_masks, tokens, masks, max_decoding_steps=2, temperature=0.0
        )
        repeated_tokens = policy.model.sample_actions_fast(
            images, img_masks, tokens, masks, max_decoding_steps=2, temperature=0.0
        )

    assert tuple(action_tokens.shape) == EXPECTED_ACTION_TOKENS_SHAPE
    assert action_tokens.dtype == torch.long
    vocab_size = policy.model.paligemma_with_expert.paligemma.lm_head.out_features
    assert (action_tokens >= 0).all() and (action_tokens < vocab_size).all()
    assert torch.equal(action_tokens, repeated_tokens)


@require_cuda
@require_hf_token
def test_pi0_fast_detokenization(policy: PI0FastPolicy) -> None:
    """Decode fixed reference tokens independently of model/backend numerics."""
    with torch.no_grad():
        actions = policy.detokenize_actions(
            REFERENCE_ACTION_TOKENS.to(DEVICE), action_horizon=2, action_dim=32
        ).cpu()

    assert tuple(actions.shape) == EXPECTED_ACTIONS_SHAPE
    assert torch.isfinite(actions).all()
    torch.testing.assert_close(actions[0, 0, :5], EXPECTED_ACTIONS_FIRST_5, rtol=0, atol=1e-3)
