# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lerobot.policies.g05.action_codec_g05 import G05NativeActionCodec, _BinarySequenceCodec


def _tiny_codec_config() -> dict:
    return {
        "parts_meta": {
            "left_control": 3,
            "left_gripper": 1,
            "right_control": 3,
            "right_gripper": 1,
        },
        "rule_based_key_patterns": ["gripper"],
        "rule_based_min_block_len": 1,
        "rule_based_binarize_threshold": 0.0,
        "num_residuals": 2,
        "model_arch": {
            "horizon": 8,
            "horizon_patch_size": 2,
            "max_component_dim": 3,
            "conv_in_action_kernel": 2,
            "encoder_channels": 64,
            "c_mults": [1],
            "strides": [[1, 1]],
            "transformer_depths": [1],
            "latent_dim": 16,
            "num_heads": 1,
            "dim_heads": 64,
            "rope_base": 10_000,
            "ffn_mult": 2,
            "layer_scale_init": 0.01,
            "n_codebooks": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
            "use_block_dct": False,
        },
    }


def test_binary_sequence_codec_roundtrip_repairs_short_middle_runs() -> None:
    codec = _BinarySequenceCodec(sequence_length=8, min_block_length=1, vocab_size=16)
    values = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    tokens = codec.encode(values, threshold=0.5)
    decoded = codec.decode(tokens)

    assert tokens.shape == (1, codec.num_tokens)
    assert decoded.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]


def test_native_action_codec_language_roundtrip_and_absent_groups() -> None:
    codec = G05NativeActionCodec(_tiny_codec_config(), action_token_begin=100)
    actions = torch.linspace(-1, 1, 8 * 8).reshape(8, 8)

    for name, parameter in codec.module.named_parameters():
        if name.endswith(("ls1", "ls2")):
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.01))

    token_ids = codec.encode_for_language({"value": actions})
    decoded, absent = codec.decode_language_tokens(
        torch.tensor(token_ids),
        horizon=8,
        action_dim=8,
    )

    assert len(token_ids) == codec.action_token_length
    assert decoded.shape == (8, 8)
    assert torch.isfinite(decoded).all()
    assert absent == set()
    assert all(key.startswith("model.") for key in codec.module.state_dict())

    empty, absent = codec.decode_language_tokens(torch.empty(0, dtype=torch.long), horizon=8, action_dim=8)
    assert absent == set(codec.parts)
    assert torch.equal(empty, torch.zeros_like(empty))
