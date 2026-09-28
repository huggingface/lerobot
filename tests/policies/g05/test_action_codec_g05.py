# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

import pytest
import torch

pytest.importorskip("transformers", reason="g05 requires the `g05` extra (transformers)")

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


def test_native_action_codec_complete_training_objective() -> None:
    config = _tiny_codec_config()
    config.update(
        {
            "commitment_loss_weight": 0.25,
            "consistency_loss_weight": 0.0,
            "quantizer_dropout": 0.0,
            "reconstruction_loss_weight": 1.0,
            "threshold_ema_dead": 0.0,
        }
    )
    codec = G05NativeActionCodec(config, action_token_begin=100).train()
    components = {
        "left_control": torch.randn(2, 8, 3),
        "right_control": torch.randn(2, 8, 3),
    }

    output = codec.training_objective(components)
    loss_dict = output["loss_dict"]

    assert output["codes"]["left_control"].shape == (2, 2, codec.code_length)
    assert output["reconstructions"]["right_control"].shape == (2, 8, 3)
    torch.testing.assert_close(
        output["loss"],
        loss_dict["reconstruction_loss"] + 0.25 * loss_dict["commitment_loss"],
    )
    assert "codebook/perplexity_l0" in loss_dict
    assert "codebook/utilization_l1" in loss_dict
    assert all(quantizer.inited for quantizer in codec.model.rvq.quantizers)

    output["loss"].backward()
    assert codec.model.conv_in.weight.grad is not None
    assert torch.isfinite(codec.model.conv_in.weight.grad).all()


def test_native_action_codec_token_residual_consistency_objective() -> None:
    config = _tiny_codec_config()
    config.update(
        {
            "consistency_loss_type": "token_residual",
            "consistency_loss_weight": 0.5,
            "quantizer_dropout": 0.0,
            "threshold_ema_dead": 0.0,
        }
    )
    codec = G05NativeActionCodec(config, action_token_begin=100).train()
    components = {
        "left_control": torch.randn(2, 8, 3),
        "right_control": torch.randn(2, 8, 3),
    }
    positives = {name: values.roll(1, dims=1) for name, values in components.items()}

    output = codec.training_objective(
        components,
        x_pos_dict=positives,
        layer_weights=[1.0, 0.5],
    )

    assert "consist/loss" in output["loss_dict"]
    assert "consist/tcr_layer_0" in output["loss_dict"]
    assert torch.isfinite(output["loss"])


def test_native_action_codec_action_time_contrastive_objective() -> None:
    config = _tiny_codec_config()
    config.update(
        {
            "action_time_contrastive_bias_init": -10.0,
            "action_time_contrastive_mode": "siglip",
            "action_time_contrastive_temperature_init": 0.07,
            "consistency_loss_type": "action_time_contrastive",
            "consistency_loss_weight": 0.5,
            "quantizer_dropout": 0.0,
            "threshold_ema_dead": 0.0,
        }
    )
    codec = G05NativeActionCodec(config, action_token_begin=100).train()
    components = {
        "left_control": torch.randn(2, 8, 3),
        "right_control": torch.randn(2, 8, 3),
    }

    output = codec.training_objective(components)
    loss_dict = output["loss_dict"]

    assert "contrastive/loss" in loss_dict
    assert "contrastive/temperature" in loss_dict
    assert "model.action_time_contrastive_loss.logit_scale" in codec.module.state_dict()
    output["loss"].backward()
    assert codec.model.action_time_contrastive_loss is not None
    assert codec.model.action_time_contrastive_loss.logit_scale.grad is not None


def test_batched_language_encoding_matches_one_chunk_at_a_time() -> None:
    codec = G05NativeActionCodec(_tiny_codec_config(), action_token_begin=100)
    codec.module.eval()
    generator = torch.Generator().manual_seed(0)
    chunks = [torch.randn(8, 8, generator=generator) for _ in range(4)]

    batched = codec.encode_batch_for_language([{"value": chunk} for chunk in chunks])

    assert batched == [codec.encode_for_language({"value": chunk}) for chunk in chunks]


def test_padded_body_parts_are_left_out_of_the_action_tokens() -> None:
    config = {**_tiny_codec_config(), "dropout_noop_parts": True}
    codec = G05NativeActionCodec(config, action_token_begin=100)
    codec.module.eval()
    actions = torch.linspace(-1, 1, 8 * 8).reshape(8, 8)
    # A one-arm robot: only the right arm and gripper exist.
    is_pad = torch.tensor([True] * 4 + [False] * 4)

    full = codec.encode_for_language({"value": actions})
    one_arm = codec.encode_for_language({"value": actions, "action_dim_is_pad": is_pad})
    decoded, absent = codec.decode_language_tokens(torch.tensor(one_arm), horizon=8, action_dim=8)
    reference, _ = codec.decode_language_tokens(torch.tensor(full), horizon=8, action_dim=8)

    assert len(one_arm) == len(full) // 2
    assert absent == {"left_control", "left_gripper"}
    torch.testing.assert_close(decoded[:, 4:], reference[:, 4:])
    assert torch.equal(decoded[:, :4], torch.zeros(8, 4))
    # Without the released flag every part stays in the sequence.
    assert len(
        G05NativeActionCodec(_tiny_codec_config(), action_token_begin=100).encode_for_language(
            {"value": actions, "action_dim_is_pad": is_pad}
        )
    ) == len(full)
