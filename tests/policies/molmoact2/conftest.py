# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""Tiny offline MolmoAct2 components shared by the MolmoAct2 tests.

The policy builds the Molmo2-4B layout of `_get_molmo2_config`. The fixtures replace it with a random 2-layer Molmo2
that keeps the real vocabulary size (so the hardcoded token ids stay valid) and a small action expert, and provide
local tokenizer and FAST tokenizer files and a tiny image processor.
"""

import copy
import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2 import modeling_molmoact2, processor_molmoact2
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.utils.constants import ACTION, OBS_STATE

# Sizes of the tiny model and of its features, given to the tests by the `sizes` fixture.
SIZES = SimpleNamespace(
    cameras=("observation.images.top", "observation.images.wrist"),
    state_dim=8,
    action_dim=7,
    max_action_dim=32,
    chunk_size=4,
    # Size of the released base vocabulary; the 128 image tokens come after it.
    base_vocab_size=154624,
    # Patches per camera crop: a 28x28 image cut into 14x14 patches, pooled 2x2 into one `<im_patch>` token.
    patches_per_crop=4,
    pixels_per_patch=14 * 14 * 3,
    processor_config={
        "image_processor": {
            "crop_mode": "resize",
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
            "max_crops": 1,
            "patch_size": 14,
            "pooling_size": [2, 2],
            "resample": 2,
            "size": {"height": 28, "width": 28},
        },
        "image_use_col_tokens": True,
        "use_single_crop_col_tokens": False,
        "use_single_crop_start_token": True,
    },
    action_expert={
        "hidden_size": 16,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "ffn_multiple_of": 0,
        "timestep_embed_dim": 8,
    },
)
# Size of the tiny FAST BPE vocabulary, and number of `<action_k>` tokens of the tiny tokenizer.
FAST_VOCAB_SIZE = 256

# The MolmoAct2 special tokens, with 256 state tokens and an action token per FAST token.
SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|image|>",
    "<|video|>",
    "<im_patch>",
    "<im_col>",
    "<im_start>",
    "<low_res_im_start>",
    "<frame_start>",
    "<im_end>",
    "<frame_end>",
    "<im_low>",
    "<action_output>",
    "<action_start>",
    "<action_end>",
    "<state_start>",
    "<state_end>",
    "<setup_start>",
    "<setup_end>",
    "<control_start>",
    "<control_end>",
    *(f"<state_{index}>" for index in range(256)),
    *(f"<action_{index}>" for index in range(FAST_VOCAB_SIZE)),
]


@pytest.fixture(scope="session")
def sizes():
    """Sizes of the tiny model and of its features."""
    return SIZES


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory):
    """MolmoAct2 tokenizer files: a byte-level Qwen2 tokenizer with the MolmoAct2 special tokens."""
    pytest.importorskip("transformers", reason="transformers is required (install lerobot[molmoact2])")
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer

    path = tmp_path_factory.mktemp("molmoact2_tokenizer")
    tokenizer = Qwen2Tokenizer(
        vocab={symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))},
        merges=[],
        unk_token=None,
        bos_token="<|im_end|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def fast_dir(tmp_path_factory):
    """A FAST tokenizer fit on smooth random (4, 3) chunks, as `UniversalActionProcessor.fit` does upstream."""
    pytest.importorskip("transformers", reason="transformers is required (install lerobot[molmoact2])")
    pytest.importorskip("scipy", reason="scipy is required (install lerobot[molmoact2])")
    from scipy.fft import dct
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    rng = np.random.default_rng(0)
    chunks = np.clip(np.cumsum(rng.normal(0, 0.2, size=(200, SIZES.chunk_size, 3)), axis=1), -1, 1)
    coefficients = np.around(dct(chunks, axis=1, norm="ortho") * 10).reshape(len(chunks), -1).astype(int)
    min_token, max_token = int(coefficients.min()), int(coefficients.max())
    bpe = ByteLevelBPETokenizer()
    trainer = BpeTrainer(
        vocab_size=FAST_VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[],
        initial_alphabet=[chr(index) for index in range(max_token - min_token + 1)],
        max_token_length=10000,
    )
    bpe._tokenizer.train_from_iterator(("".join(map(chr, row - min_token)) for row in coefficients), trainer)
    path = tmp_path_factory.mktemp("molmoact2_fast")
    PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False).save_pretrained(path)
    (path / "processor_config.json").write_text(json.dumps({"scale": 10, "min_token": min_token}))
    return path


@pytest.fixture(scope="session")
def molmo2_config(tokenizer_dir):
    """A 2-layer Molmo2 with the real vocabulary layout; `<im_patch>` is the id of the tiny tokenizer."""
    from transformers import Molmo2Config, Qwen2Tokenizer

    image_patch_id = Qwen2Tokenizer.from_pretrained(tokenizer_dir).convert_tokens_to_ids("<im_patch>")
    attention = {"num_attention_heads": 2, "head_dim": 8, "intermediate_size": 32}
    return Molmo2Config(
        text_config={
            **attention,
            "hidden_size": 16,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "vocab_size": SIZES.base_vocab_size,
            "max_position_embeddings": 512,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        vision_config={
            **attention,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "image_size": [28, 28],
        },
        adapter_config={
            **attention,
            "hidden_size": 16,
            "num_key_value_heads": 2,
            "text_hidden_size": 16,
            "vision_feature_layer": [-1],
        },
        image_start_token_id=SIZES.base_vocab_size,
        image_end_token_id=SIZES.base_vocab_size + 1,
        image_patch_id=image_patch_id,
        low_res_image_start_token_id=SIZES.base_vocab_size + 4,
        frame_start_token_id=SIZES.base_vocab_size + 7,
        frame_end_token_id=SIZES.base_vocab_size + 8,
    )


@pytest.fixture
def tiny_molmoact2(monkeypatch, molmo2_config):
    """Make `MolmoAct2Model` build the tiny Molmo2 and action expert."""
    monkeypatch.setattr(modeling_molmoact2, "_get_molmo2_config", lambda: copy.deepcopy(molmo2_config))
    tiny_expert = partial(modeling_molmoact2.ActionExpert, **SIZES.action_expert)
    monkeypatch.setattr(modeling_molmoact2, "ActionExpert", tiny_expert)
    return molmo2_config


@pytest.fixture
def make_config(tokenizer_dir, monkeypatch):
    """Build a `MolmoAct2Config` of the tiny model: float32 on CPU, chunks of 4 steps, 2 flow timesteps.

    The features are two 32x32 cameras, an 8-D state and a 7-D action. Processors built from it resize the cameras
    to 28x28: 2x2 patches pooled into a single `<im_patch>` token per camera.
    """
    monkeypatch.setattr(processor_molmoact2, "_PROCESSOR_CONFIG", SIZES.processor_config)

    def make(**overrides):
        input_features = {
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)) for camera in SIZES.cameras
        }
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(SIZES.state_dim,))
        output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(SIZES.action_dim,))}
        kwargs = {
            "tokenizer_name": str(tokenizer_dir),
            "device": "cpu",
            "dtype": torch.float32,
            "chunk_size": SIZES.chunk_size,
            "n_action_steps": SIZES.chunk_size,
            "num_flow_timesteps": 2,
            "num_inference_steps": 3,
            "image_keys": list(SIZES.cameras),
            "input_features": input_features,
            "output_features": output_features,
            **overrides,
        }
        return MolmoAct2Config(**kwargs)

    return make


@pytest.fixture
def make_policy(tiny_molmoact2, make_config):
    """Build a tiny, randomly initialized `MolmoAct2Policy`; keyword arguments override the config.

    Every call builds the same weights. The action expert is moved away from its zero-initialized output layers, so
    that the velocity depends on every input.
    """

    def make(**overrides):
        torch.manual_seed(0)
        policy = MolmoAct2Policy(make_config(**overrides))
        with torch.no_grad():
            for param in policy.model.action_expert.parameters():
                param.add_(0.1 * torch.randn_like(param))
        return policy

    return make
