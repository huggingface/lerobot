#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch
from conftest import (
    BATCH_SIZE,
    QWEN_HIDDEN_SIZE,
    _FakeVideoEncoder,
    make_config,
    make_inference_batch,
    make_train_batch,
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vla_jepa import modeling_vla_jepa
from lerobot.policies.vla_jepa.modeling_vla_jepa import (
    VLAJEPAPolicy,
    _action_token_windows,
    _encoded_video_validity,
    _horizon_validity,
    _masked_l1_components,
    _safe_ratio,
)
from lerobot.utils.constants import OBS_IMAGES


def test_encoded_video_validity_combines_views_tubelets_and_prefix_closes() -> None:
    video_is_pad = torch.tensor(
        [
            [[False, False, False, False], [False, False, False, False]],
            [[False, False, True, True], [False, False, False, False]],
            [[False, True, False, False], [False, False, False, False]],
        ]
    )
    expected = torch.tensor([[True, True], [True, False], [False, False]])
    assert torch.equal(_encoded_video_validity(video_is_pad, tubelet_size=2), expected)


def test_horizon_validity_requires_complete_span() -> None:
    encoded_valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    expected = torch.tensor([[True, False], [False, False]])
    assert torch.equal(_horizon_validity(encoded_valid, horizon=2), expected)


def test_masked_l1_components_use_scalar_latent_denominator() -> None:
    prediction = torch.ones(1, 2, 2, 3)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False]])
    numerator, denominator = _masked_l1_components(prediction, target, valid)
    assert numerator.item() == 6
    assert denominator.item() == 6
    assert _safe_ratio(numerator, denominator).item() == 1


def test_all_invalid_mask_returns_differentiable_zero() -> None:
    prediction = torch.full((2, 3, 2, 4), float("nan"), requires_grad=True)
    numerator, denominator = _masked_l1_components(
        prediction, torch.zeros_like(prediction), torch.zeros(2, 3, dtype=torch.bool)
    )
    loss = _safe_ratio(numerator, denominator)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_action_token_windows_preserve_origin_and_transition_order() -> None:
    action_tokens = torch.arange(1 * 3 * 2 * 1).reshape(1, 3, 2, 1)
    windows = _action_token_windows(action_tokens, horizon=2, num_origins=2)
    assert windows.tolist() == [[[[0], [1], [2], [3]], [[2], [3], [4], [5]]]]


def test_opt_in_multiview_merge_does_not_mix_batch_samples(
    monkeypatch: pytest.MonkeyPatch, patch_vla_jepa_external_models: None
) -> None:
    class TubeletVideoEncoder(_FakeVideoEncoder):
        def __init__(self) -> None:
            super().__init__(tubelet_size=2)

        def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
            batch_size, num_frames = pixel_values_videos.shape[:2]
            values = pixel_values_videos.float().mean(dim=(2, 3, 4))
            values = values.reshape(batch_size, num_frames // 2, 2).mean(dim=2)
            return values[:, :, None].expand(batch_size, num_frames // 2, self.config.hidden_size)

    monkeypatch.setattr(
        modeling_vla_jepa.AutoModel, "from_pretrained", lambda *args, **kwargs: TubeletVideoEncoder()
    )
    config = make_config(num_video_frames=6)
    config.input_features[f"{OBS_IMAGES}.second"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))
    config.jepa_tubelet_size = 2
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    captured_frames: list[torch.Tensor] = []

    def capture_predictor(frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        del action_tokens
        captured_frames.append(frame_tokens.detach().clone())
        return torch.zeros_like(frame_tokens)

    monkeypatch.setattr(policy.model.video_predictor, "forward", capture_predictor)
    videos = torch.empty(BATCH_SIZE, 2, 6, 3, 8, 8)
    videos[0, 0].fill_(1)
    videos[0, 1].fill_(2)
    videos[1, 0].fill_(10)
    videos[1, 1].fill_(20)
    action_tokens = torch.randn(BATCH_SIZE, 4, QWEN_HIDDEN_SIZE)
    video_is_pad = torch.zeros(BATCH_SIZE, 2, 6, dtype=torch.bool)

    policy.model._world_model_loss(videos, action_tokens, video_is_pad)

    horizon_one_frames = captured_frames[0]
    assert torch.equal(horizon_one_frames[0, 0, :8], torch.ones(8))
    assert torch.equal(horizon_one_frames[0, 0, 8:], torch.full((8,), 2.0))
    assert torch.equal(horizon_one_frames[1, 0, :8], torch.full((8,), 10.0))
    assert torch.equal(horizon_one_frames[1, 0, 8:], torch.full((8,), 20.0))


def test_direct_multihorizon_forward_contracts_and_gradients(
    monkeypatch: pytest.MonkeyPatch, patch_vla_jepa_external_models: None
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    original_forward = policy.model.video_predictor.forward

    def capture_forward(frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        output = original_forward(frame_tokens, action_tokens)
        calls.append((tuple(frame_tokens.shape), tuple(action_tokens.shape), tuple(output.shape)))
        return output

    monkeypatch.setattr(policy.model.video_predictor, "forward", capture_forward)
    loss, logs = policy.forward(make_train_batch(num_video_frames=4))
    loss.backward()

    assert calls == [
        ((BATCH_SIZE, 3, 8), (BATCH_SIZE, 6, QWEN_HIDDEN_SIZE), (BATCH_SIZE, 3, 8)),
        ((BATCH_SIZE, 2, 8), (BATCH_SIZE, 8, QWEN_HIDDEN_SIZE), (BATCH_SIZE, 2, 8)),
    ]
    assert set(logs) == {
        "action_loss",
        "wm_loss",
        "wm_loss_h1",
        "wm_valid_h1",
        "wm_loss_h2",
        "wm_valid_h2",
        "wm_direct_loss",
        "loss",
    }
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.model.video_predictor.parameters()
    )


def test_positive_consistency_adds_one_recursive_call_and_metric(
    monkeypatch: pytest.MonkeyPatch, patch_vla_jepa_external_models: None
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    config.temporal_consistency_weight = 0.5
    policy = VLAJEPAPolicy(config)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    outputs: list[torch.Tensor] = []
    original_forward = policy.model.video_predictor.forward

    def capture_forward(frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        output = original_forward(frame_tokens, action_tokens)
        calls.append((tuple(frame_tokens.shape), tuple(action_tokens.shape)))
        outputs.append(output)
        if output.requires_grad:
            output.retain_grad()
        return output

    monkeypatch.setattr(policy.model.video_predictor, "forward", capture_forward)
    loss, logs = policy.forward(make_train_batch(num_video_frames=4))
    loss.backward()

    assert calls == [
        ((BATCH_SIZE, 3, 8), (BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)),
        ((BATCH_SIZE, 2, 8), (BATCH_SIZE, 8, QWEN_HIDDEN_SIZE)),
        ((BATCH_SIZE, 2, 8), (BATCH_SIZE, 4, QWEN_HIDDEN_SIZE)),
    ]
    assert set(logs) == {
        "action_loss",
        "wm_loss",
        "wm_loss_h1",
        "wm_valid_h1",
        "wm_loss_h2",
        "wm_valid_h2",
        "wm_direct_loss",
        "wm_consistency_loss",
        "loss",
    }
    assert logs["wm_consistency_loss"] >= 0
    assert outputs[0].requires_grad and outputs[0].grad is not None
    assert outputs[1].requires_grad and outputs[1].grad is not None
    assert not outputs[2].requires_grad


def test_recursive_step_uses_next_action_token_group(
    monkeypatch: pytest.MonkeyPatch, patch_vla_jepa_external_models: None
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    config.temporal_consistency_weight = 0.5
    policy = VLAJEPAPolicy(config)
    captured_actions: list[torch.Tensor] = []

    def capture_predictor(frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        captured_actions.append(action_tokens.detach().clone())
        return frame_tokens

    monkeypatch.setattr(policy.model.video_predictor, "forward", capture_predictor)
    video_embeddings = torch.randn(BATCH_SIZE, 4, 8)
    flat_action_tokens = torch.arange(BATCH_SIZE * 3 * 2 * QWEN_HIDDEN_SIZE).reshape(
        BATCH_SIZE, 6, QWEN_HIDDEN_SIZE
    )
    video_is_pad = torch.zeros(BATCH_SIZE, 1, 4, dtype=torch.bool)

    policy.model._multi_horizon_world_model_loss(
        video_embeddings, flat_action_tokens, video_is_pad, tubelet_size=1
    )

    grouped_actions = flat_action_tokens.reshape(BATCH_SIZE, 3, 2, QWEN_HIDDEN_SIZE)
    expected_recursive_actions = grouped_actions[:, 1:3].flatten(1, 2)
    assert len(captured_actions) == 3
    assert torch.equal(captured_actions[2], expected_recursive_actions)


def test_all_invalid_consistency_loss_is_finite_zero(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    config.temporal_consistency_weight = 0.5
    policy = VLAJEPAPolicy(config)
    action_tokens = torch.randn(BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    video_is_pad = torch.ones(BATCH_SIZE, 1, 4, dtype=torch.bool)

    loss, metrics = policy.model._world_model_loss(videos, action_tokens, video_is_pad)
    loss.backward()

    assert loss.item() == 0
    assert metrics["wm_consistency_loss"].item() == 0
    assert torch.isfinite(loss)


def test_prepare_model_inputs_propagates_per_view_video_padding(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    batch = make_train_batch(num_video_frames=4)
    image_key = f"{OBS_IMAGES}.laptop"
    batch[f"{image_key}_is_pad"] = torch.tensor([[False, False, False, True], [False, False, True, True]])
    inputs = policy._prepare_model_inputs(batch)
    assert torch.equal(inputs["video_is_pad"], batch[f"{image_key}_is_pad"].unsqueeze(1))


def test_padding_and_missing_horizons_are_excluded(
    monkeypatch: pytest.MonkeyPatch,
    patch_vla_jepa_external_models: None,
) -> None:
    encoded_prefix_lengths: list[int] = []

    class BidirectionalVideoEncoder(_FakeVideoEncoder):
        def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
            batch_size, num_frames = pixel_values_videos.shape[:2]
            encoded_prefix_lengths.append(num_frames)
            frame_values = pixel_values_videos.float().mean(dim=(2, 3, 4))
            contextual_values = frame_values + frame_values.mean(dim=1, keepdim=True)
            return contextual_values[:, :, None].expand(batch_size, num_frames, self.config.hidden_size)

    monkeypatch.setattr(
        modeling_vla_jepa.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: BidirectionalVideoEncoder(),
    )
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    action_tokens = torch.randn(BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    video_is_pad = torch.tensor([[[False, False, False, True]], [[False, False, True, True]]])

    loss, metrics = policy.model._world_model_loss(videos, action_tokens, video_is_pad)
    changed_padding = videos.clone()
    changed_padding[:, :, -1] = 100
    changed_loss, changed_metrics = policy.model._world_model_loss(
        changed_padding, action_tokens, video_is_pad
    )

    assert torch.allclose(loss, changed_loss)
    assert metrics["wm_valid_h1"].item() == 3
    assert metrics["wm_valid_h2"].item() == 1
    assert metrics.keys() == changed_metrics.keys()
    assert encoded_prefix_lengths == [1, 2, 3, 4, 1, 2, 3, 4]


def test_horizon_with_no_valid_targets_does_not_dilute_direct_loss(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    action_tokens = torch.randn(BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    video_is_pad = torch.tensor([[[False, False, True, True]]] * BATCH_SIZE)

    loss, metrics = policy.model._world_model_loss(videos, action_tokens, video_is_pad)

    assert metrics["wm_valid_h1"].item() == BATCH_SIZE
    assert metrics["wm_valid_h2"].item() == 0
    assert metrics["wm_loss_h2"].item() == 0
    assert torch.allclose(loss, metrics["wm_loss_h1"])


def test_multihorizon_rejects_missing_qwen_action_tokens(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    too_few_action_tokens = torch.randn(BATCH_SIZE, 5, QWEN_HIDDEN_SIZE)

    with pytest.raises(ValueError, match="exactly 6 Qwen action tokens"):
        policy.model._world_model_loss(videos, too_few_action_tokens)


def test_all_invalid_direct_multihorizon_loss_is_zero(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)
    action_tokens = torch.randn(BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    video_is_pad = torch.ones(BATCH_SIZE, 1, 4, dtype=torch.bool)

    loss, metrics = policy.model._world_model_loss(videos, action_tokens, video_is_pad)
    loss.backward()

    assert loss.item() == 0
    assert metrics["wm_valid_h1"].item() == 0
    assert metrics["wm_valid_h2"].item() == 0


def test_multihorizon_opt_in_does_not_change_inference(
    monkeypatch: pytest.MonkeyPatch, patch_vla_jepa_external_models: None
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1, 2)
    policy = VLAJEPAPolicy(config)

    def unexpected_predictor_call(*args, **kwargs):
        raise AssertionError("World predictor must not run during inference")

    monkeypatch.setattr(policy.model.video_predictor, "forward", unexpected_predictor_call)
    actions = policy.predict_action_chunk(make_inference_batch())
    assert actions.shape == (BATCH_SIZE, config.chunk_size, config.action_dim)


def test_multihorizon_reuses_legacy_state_dict_keys(patch_vla_jepa_external_models: None) -> None:
    legacy = VLAJEPAPolicy(make_config(num_video_frames=4))
    opt_in_config = make_config(num_video_frames=4)
    opt_in_config.prediction_horizons = (1, 2)
    opt_in = VLAJEPAPolicy(opt_in_config)
    assert legacy.state_dict().keys() == opt_in.state_dict().keys()
    load_result = opt_in.load_state_dict(legacy.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys


def test_single_horizon_tuple_is_opt_in_not_legacy(
    patch_vla_jepa_external_models: None,
) -> None:
    config = make_config(num_video_frames=4)
    config.prediction_horizons = (1,)
    policy = VLAJEPAPolicy(config)

    _, logs = policy.forward(make_train_batch(num_video_frames=4))

    assert set(logs) == {
        "action_loss",
        "wm_loss",
        "wm_loss_h1",
        "wm_valid_h1",
        "wm_direct_loss",
        "loss",
    }


def test_legacy_world_model_keeps_one_full_clip_encoder_call(
    monkeypatch: pytest.MonkeyPatch,
    patch_vla_jepa_external_models: None,
) -> None:
    policy = VLAJEPAPolicy(make_config(num_video_frames=4))
    prefix_lengths: list[int] = []
    original_encode = policy.model.video_encoder.get_vision_features

    def capture_encode(pixel_values_videos: torch.Tensor) -> torch.Tensor:
        prefix_lengths.append(pixel_values_videos.shape[1])
        return original_encode(pixel_values_videos)

    monkeypatch.setattr(policy.model.video_encoder, "get_vision_features", capture_encode)
    videos = torch.rand(BATCH_SIZE, 1, 4, 3, 8, 8)
    action_tokens = torch.randn(BATCH_SIZE, 6, QWEN_HIDDEN_SIZE)

    loss = policy.model._world_model_loss(videos, action_tokens)

    assert torch.isfinite(loss)
    assert prefix_lengths == [4]


def test_legacy_input_path_does_not_add_video_padding_contract(
    patch_vla_jepa_external_models: None,
) -> None:
    policy = VLAJEPAPolicy(make_config(num_video_frames=4))
    assert "video_is_pad" not in policy._prepare_model_inputs(make_train_batch(num_video_frames=4))


@pytest.mark.parametrize(
    "video_is_pad,tubelet_size,match",
    [
        (torch.zeros(2, 4, dtype=torch.bool), 2, "shape"),
        (torch.zeros(1, 1, 3, dtype=torch.bool), 2, "divisible"),
        (torch.zeros(1, 1, 4, dtype=torch.bool), 0, "positive tubelet"),
    ],
)
def test_encoded_video_validity_rejects_invalid_contracts(
    video_is_pad: torch.Tensor, tubelet_size: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _encoded_video_validity(video_is_pad, tubelet_size)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_encoded_video_validity_coerces_numeric_masks(dtype: torch.dtype) -> None:
    video_is_pad = torch.tensor([[[0, 0, 1, 1]]], dtype=dtype)
    assert torch.equal(
        _encoded_video_validity(video_is_pad, tubelet_size=2),
        torch.tensor([[True, False]]),
    )
