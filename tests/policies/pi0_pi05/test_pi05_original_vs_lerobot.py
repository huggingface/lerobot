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

"""Compare LeRobot PI0.5 against the vendored OpenPI PyTorch reference."""

import gc
import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.policies.pi0_pi05.openpi_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from tests.policies.pi0_pi05.utils.openpi_parity import (  # noqa: E402
    assert_processor_inputs_match_lerobot,
    clone_batch,
    deterministic_openpi_forward_preprocess,
    fix_reference_state_dict,
    fixed_flow_sampling,
    load_openpi_reference_state_dict,
    make_openpi_observation_from_raw,
    openpi_model_actions_from_raw,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="OpenPI parity and torch.compile checks are too slow for CI; run manually on GPU nodes",
)

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPILE_MODE = "default"
FORWARD_RTOL = 1e-4
FORWARD_ATOL = 1e-4
SAMPLE_RTOL = 1e-2
SAMPLE_ATOL = 5e-3

DUMMY_DATASET_STATS = {
    OBS_STATE: {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    ACTION: {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        "base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "left_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "right_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    },
}


@pytest.fixture(autouse=True)
def cleanup_cuda_after_test():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class PI05BaseOriginalConfig:
    action_dim: int = DUMMY_ACTION_DIM
    action_horizon: int = DUMMY_ACTION_HORIZON
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "float32"
    pi05: bool = True
    dtype: str = "float32"
    pytorch_compile_mode: str | None = None


def instantiate_lerobot_pi05(*, compile_model: bool = False, gradient_checkpointing: bool = False):
    config = PreTrainedConfig.from_pretrained("lerobot/pi05_base")
    config.device = str(DEVICE)
    config.dtype = "float32"
    config.compile_model = compile_model
    config.compile_mode = COMPILE_MODE
    config.gradient_checkpointing = gradient_checkpointing

    policy = PI05Policy.from_pretrained("lerobot/pi05_base", config=config, strict=True)
    policy.to(DEVICE)
    policy.config.device = str(DEVICE)
    preprocessor, _ = make_pi05_pre_post_processors(config=policy.config, dataset_stats=DUMMY_DATASET_STATS)
    return policy, preprocessor


def instantiate_original_pi05():
    policy = PI0Pytorch(PI05BaseOriginalConfig()).to(DEVICE)

    # NOTE: `lerobot/pi05_base` 的 LeRobot loader 和 PI0 一样会在 strict load 前做 key
    # 兼容转换，因此预期没有 missing_keys 或 unexpected_keys。vendored reference 则是裸
    # `nn.Module`，需要在测试侧补齐 checkpoint 与模块命名之间的最小差异。
    # NOTE: `lm_head.weight` 是 PaliGemma tied embedding 的保存名；LeRobot 的
    # from_pretrained 会把它映射到内部 `embed_tokens.weight`，而 reference 模型没有这层
    # loader，所以这里手动复用同一份 tensor，避免把权重别名差异误判成模型差异。
    state_dict = fix_reference_state_dict(load_openpi_reference_state_dict("lerobot/pi05_base"))
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    assert missing_keys == []
    assert unexpected_keys == []
    return policy


def create_dummy_data():
    batch_size = 2
    prompt = "Pick up the red block and place it in the bin"
    return {
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=DEVICE),
        ACTION: torch.randn(
            batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=DEVICE
        ),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=DEVICE
        ),
        "observation.images.left_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=DEVICE
        ),
        "observation.images.right_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=DEVICE
        ),
        "task": [prompt for _ in range(batch_size)],
    }


def prepare_parity_inputs(lerobot_pi05, lerobot_preprocessor):
    torch.manual_seed(0)
    raw_batch = create_dummy_data()
    lerobot_batch = lerobot_preprocessor(clone_batch(raw_batch))
    openpi_observation = make_openpi_observation_from_raw(
        raw_batch,
        action_dim=DUMMY_ACTION_DIM,
        max_token_len=DUMMY_MAX_TOKEN_LEN,
        dataset_stats=DUMMY_DATASET_STATS,
        pi05=True,
    )
    openpi_actions = openpi_model_actions_from_raw(
        raw_batch,
        action_dim=DUMMY_ACTION_DIM,
        dataset_stats=DUMMY_DATASET_STATS,
        pi05=True,
    )
    assert_processor_inputs_match_lerobot(
        lerobot_pi05,
        lerobot_batch,
        openpi_observation,
        compare_state=False,
    )
    batch_size = raw_batch[OBS_STATE].shape[0]
    noise = torch.randn(
        batch_size,
        DUMMY_ACTION_HORIZON,
        DUMMY_ACTION_DIM,
        dtype=torch.float32,
        device=DEVICE,
    )
    time = torch.linspace(0.2, 0.8, batch_size, dtype=torch.float32, device=DEVICE)
    return lerobot_batch, openpi_observation, openpi_actions, noise, time


def assert_forward_matches(*, compile_model: bool = False, gradient_checkpointing: bool = False):
    lerobot_pi05, lerobot_preprocessor = instantiate_lerobot_pi05(
        compile_model=compile_model,
        gradient_checkpointing=gradient_checkpointing,
    )
    original_pi05 = instantiate_original_pi05()
    lerobot_batch, openpi_observation, openpi_actions, noise, time = prepare_parity_inputs(
        lerobot_pi05,
        lerobot_preprocessor,
    )

    if gradient_checkpointing:
        lerobot_pi05.train()
    else:
        lerobot_pi05.eval()
    original_pi05.eval()

    with fixed_flow_sampling(lerobot_pi05.model, noise=noise, time=time):
        lerobot_loss, _ = lerobot_pi05(lerobot_batch, reduction="none")
    with deterministic_openpi_forward_preprocess(original_pi05):
        openpi_losses = original_pi05(openpi_observation, openpi_actions, noise=noise, time=time)
    openpi_loss = openpi_losses.mean(dim=(1, 2))

    torch.testing.assert_close(lerobot_loss, openpi_loss, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


def assert_sample_actions_match_openpi(*, compile_model: bool = False):
    lerobot_pi05, lerobot_preprocessor = instantiate_lerobot_pi05(compile_model=compile_model)
    original_pi05 = instantiate_original_pi05()
    lerobot_batch, openpi_observation, _openpi_actions, noise, _time = prepare_parity_inputs(
        lerobot_pi05,
        lerobot_preprocessor,
    )

    lerobot_pi05.eval()
    original_pi05.eval()
    with torch.no_grad():
        lerobot_actions = lerobot_pi05.predict_action_chunk(lerobot_batch, noise=noise, num_steps=10)
        openpi_actions = original_pi05.sample_actions(
            device=DEVICE,
            observation=openpi_observation,
            noise=noise,
            num_steps=10,
        )

    torch.testing.assert_close(lerobot_actions, openpi_actions, rtol=SAMPLE_RTOL, atol=SAMPLE_ATOL)


def test_pi05_forward_matches_openpi():
    assert_forward_matches()


def test_pi05_sample_actions_match_openpi():
    assert_sample_actions_match_openpi()


def test_pi05_gradient_checkpointing_forward_matches_openpi():
    assert_forward_matches(gradient_checkpointing=True)


def test_pi05_compile_forward_matches_openpi():
    assert_forward_matches(compile_model=True)


def test_pi05_compile_sample_actions_match_openpi():
    assert_sample_actions_match_openpi(compile_model=True)


def test_pi05_compile_gradient_checkpointing_forward_matches_openpi():
    assert_forward_matches(compile_model=True, gradient_checkpointing=True)
