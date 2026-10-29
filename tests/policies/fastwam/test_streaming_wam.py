# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

import json
from concurrent.futures import Executor, Future

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.convert_streaming_checkpoint import ARCHITECTURE, PHASE, map_checkpoint
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
from lerobot.policies.fastwam.streaming import Completion, StreamingController, prefix_target
from lerobot.policies.fastwam.wan.streaming import (
    ACStreamSlotEncoder,
    action_consistency_boundary,
    build_ac_stream_condition_mask,
    build_ac_stream_policy_mask,
    video_consistency_boundary,
)


class TinyVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    z_dim = 2

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            video = torch.stack(video)
        return torch.nn.functional.avg_pool3d(video[:, :2, ::4], (1, 16, 16))


def tiny_config(streaming=True):
    common = {
        "hidden_dim": 16,
        "ffn_dim": 32,
        "freq_dim": 8,
        "text_dim": 8,
        "num_heads": 2,
        "attn_head_dim": 8,
        "num_layers": 2,
        "eps": 1e-6,
        "fp32_attention": False,
    }
    return FastWAMConfig(
        streaming_wam=streaming,
        num_inference_steps=1,
        action_dim=3,
        proprio_dim=2,
        device="cpu",
        dtype=torch.float32,
        base_model_id=None,
        load_text_encoder=False,
        image_size=(16, 32),
        video_dit_config={
            **common,
            "in_dim": 2,
            "out_dim": 2,
            "patch_size": (1, 1, 1),
            "seperated_timestep": True,
            "video_attention_mask_mode": "first_frame_causal",
        },
        action_dit_config={**common, "action_dim": 3},
    )


@pytest.fixture
def tiny_components(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.fastwam.modeling_fastwam.load_pretrained_wan_vae", lambda **kwargs: TinyVAE()
    )
    monkeypatch.setattr(
        "lerobot.policies.fastwam.modeling_fastwam.build_wan_tokenizer", lambda **kwargs: None
    )


def observation():
    return {
        "input_image": torch.linspace(0, 1, 3 * 16 * 32).reshape(1, 3, 16, 32),
        "context": torch.ones(1, 4, 8),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "observation.state": torch.zeros(1, 2),
    }


@pytest.mark.parametrize("action_conditioned", [False, True])
def test_default_off_legacy_config_actions_loss_and_keys(tiny_components, tmp_path, action_conditioned):
    torch.manual_seed(9)
    settings = tiny_config(False)
    settings.video_dit_config.update(action_conditioned=action_conditioned, action_dim=3)
    policy = FastWAMPolicy(settings)
    policy.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config.pop("streaming_wam") is False
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = FastWAMPolicy.from_pretrained(tmp_path, strict=True)
    assert list(policy.state_dict()) == list(loaded.state_dict())
    assert all("slot_encoder" not in name for name in loaded.state_dict())
    assert torch.equal(policy.predict_action_chunk(observation()), loaded.predict_action_chunk(observation()))
    batch = {**observation(), "video": torch.rand(1, 3, 9, 16, 32), "action": torch.rand(1, 32, 3)}
    torch.manual_seed(21)
    first = policy(batch)[0]
    torch.manual_seed(21)
    second = loaded(batch)[0]
    assert torch.equal(first, second)


def test_tiny_cpu_optimizer_save_strict_reload(tiny_components, tmp_path):
    torch.manual_seed(5)
    policy = FastWAMPolicy(tiny_config())
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    initial = policy.model.action_expert.head.weight.detach().clone()
    batch = {**observation(), "video": torch.rand(1, 3, 9, 16, 32), "action": torch.rand(1, 32, 3)}
    policy.train()
    loss, metrics = policy(batch)
    assert torch.isfinite(loss) and metrics
    loss.backward()
    optimizer.step()
    assert not torch.equal(initial, policy.model.action_expert.head.weight)
    expected = policy.predict_action_chunk(observation())
    policy.save_pretrained(tmp_path)
    loaded = FastWAMPolicy.from_pretrained(tmp_path, strict=True, init_empty_weights=True)
    assert loaded.config.streaming_wam
    assert any("slot_encoder" in key for key in loaded.state_dict())
    assert torch.equal(expected, loaded.predict_action_chunk(observation()))
    assert not any(param.is_meta for param in loaded.parameters())


def test_streaming_reload_rejects_missing_weights_even_non_strict(tiny_components, tmp_path):
    from safetensors.torch import save_file

    policy = FastWAMPolicy(tiny_config())
    policy.save_pretrained(tmp_path)
    state = policy.state_dict()
    state.pop(next(iter(state)))
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="exact checkpoint keys"):
        FastWAMPolicy.from_pretrained(tmp_path, strict=False, init_empty_weights=True)


@pytest.mark.parametrize(
    "change",
    [
        {"num_inference_steps": 2},
        {"compile_action_infer": True},
        {"n_action_steps": 16},
        {"base_model_id": "lerobot/fastwam_base"},
        {"video_dit_config": {"action_conditioned": True}},
    ],
)
def test_reject_unsupported_streaming_config(change):
    with pytest.raises(ValueError, match="Streaming-WAM"):
        FastWAMConfig(
            streaming_wam=True,
            num_inference_steps=change.get("num_inference_steps", 1),
            base_model_id=change.get("base_model_id"),
            **{
                key: value
                for key, value in change.items()
                if key not in ("base_model_id", "num_inference_steps")
            },
        )


def test_masks_and_known_unknown_slots():
    policy = build_ac_stream_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )
    assert policy[:2, :2].all() and not policy[:2, 2:].any()
    assert policy[6:14, :2].all() and policy[6:14, 6:14].all()
    assert not policy[6:14, 2:6].any() and not policy[6:14, 14:].any()
    assert policy[14:].all()
    condition = build_ac_stream_condition_mask(
        video_seq_len=6, condition_seq_len=16, video_tokens_per_frame=2, device=torch.device("cpu")
    )
    assert condition[2:4, 6:].all()
    assert not condition[:2, 6:].any() and not condition[4:6, 6:].any()
    assert not condition[6:, 2:6].any()
    encoder = ACStreamSlotEncoder(4)
    tokens = torch.randn(1, 16, 4)
    known = torch.arange(16).unsqueeze(0) < 8
    output = encoder(tokens, known)
    assert torch.equal(output[:, 8:], encoder(torch.randn_like(tokens), known)[:, 8:])
    assert not torch.equal(output[:, :8], encoder(torch.randn_like(tokens), known)[:, :8])


def test_hard_prefix_and_known_queries_ignore_future_noise(tiny_components):
    policy = FastWAMPolicy(tiny_config())
    batch = observation()
    previous = torch.randn(1, 32, 3)
    kwargs = {
        "prompt": None,
        "input_image": batch["input_image"],
        "action_horizon": 32,
        "context": batch["context"],
        "context_mask": batch["context_mask"],
        "proprio": batch["observation.state"],
        "previous_action_chunk": previous,
    }
    first = policy.model.infer_action(**kwargs, seed=1)["action"]
    second = policy.model.infer_action(**kwargs, seed=2)["action"]
    assert torch.equal(first[:8], previous[0, :8]) and torch.equal(first[:8], second[:8])
    assert not torch.equal(first[8:], second[8:])
    video = torch.randn(1, 2, 3, 1, 2)
    action = torch.randn(1, 32, 3)
    velocity = policy.model.streaming_velocity(
        video, action, torch.tensor([1000.0]), batch["context"], batch["context_mask"], previous
    )[1]
    action[:, 8:] += 100
    video[:, :, 1:] += 100
    changed = policy.model.streaming_velocity(
        video, action, torch.tensor([1000.0]), batch["context"], batch["context_mask"], previous
    )[1]
    assert torch.equal(velocity[:, :8], changed[:, :8])


class ControlledExecutor(Executor):
    def __init__(self):
        self.calls = []
        self.closed = False

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        self.calls.append((future, lambda: fn(*args, **kwargs)))
        if len(self.calls) == 1:
            self.complete(0)
        return future

    def complete(self, index):
        future, fn = self.calls[index]
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)

    def shutdown(self, **kwargs):
        self.closed = True


def test_controller_alignment_snapshot_ack_and_on_time_install():
    executor = ControlledExecutor()
    inputs = []

    def predict(batch, previous):
        inputs.append((batch, previous))
        chunk = torch.arange(32.0).reshape(1, 32, 1) + 100 * (len(inputs) - 1)
        if previous is not None:
            chunk[:, :8] = previous[:, :8]
        return chunk

    clock = iter(range(100)).__next__
    controller = StreamingController(predict, executor=executor, clock=clock)
    obs = {"x": torch.tensor([1.0])}
    actions = []
    for index in range(16):
        actions.append(controller.next_action(obs).item())
        if index == 8:
            obs["x"].fill_(99)
            executor.complete(1)
        controller.mark_action_executed()
    assert actions == list(range(16))
    assert inputs[1][0]["x"].item() == 1
    assert inputs[1][1][0, :16, 0].tolist() == list(range(8, 24))
    assert controller.next_action(obs).item() == 108
    assert controller.deadline_misses == 0
    with pytest.raises(RuntimeError, match="mark_action_executed"):
        controller.next_action(obs)
    controller.mark_action_executed()
    controller.close()
    controller.close()
    assert executor.closed


def test_controller_miss_blocks_and_worker_exception_propagates():
    class CompletingFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                self.set_exception(RuntimeError("inference failed at boundary"))
            return super().result(timeout)

    class ExecutorWithLateFailure(ControlledExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if self.calls:
                return CompletingFuture()
            return super().submit(fn, *args, **kwargs)

    controller = StreamingController(
        lambda batch, prev: torch.zeros(1, 32, 1), executor=ExecutorWithLateFailure()
    )
    for _ in range(16):
        controller.next_action({})
        controller.mark_action_executed()
    with pytest.raises(RuntimeError, match="inference failed at boundary"):
        controller.next_action({})
    controller.reset()
    controller.close()


def test_reset_cancels_pending_and_rejects_stale_results():
    executor = ControlledExecutor()
    controller = StreamingController(lambda batch, prev: torch.zeros(1, 32, 1), executor=executor)
    for _ in range(9):
        controller.next_action({})
        controller.mark_action_executed()
    pending = executor.calls[1][0]
    controller.reset()
    assert pending.cancelled()
    stale = Future()
    stale.set_result(Completion(0, torch.zeros(1, 32, 1), 0, 1))
    with pytest.raises(RuntimeError, match="Stale"):
        controller._install(stale, 8)
    controller.close()
    with pytest.raises(RuntimeError, match="closed"):
        controller.next_action({})
    with pytest.raises(RuntimeError, match="No returned action"):
        controller.mark_action_executed()


def test_late_completion_blocks_then_installs_without_skipping():
    now = [0]

    class LateFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                now[0] += 1_000_000_000
                self.set_result(Completion(0, torch.arange(32.0).reshape(1, 32, 1), 0, now[0]))
            return super().result(timeout)

    class LateExecutor(ControlledExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if self.calls:
                return LateFuture()
            return super().submit(fn, *args, **kwargs)

    controller = StreamingController(
        lambda batch, prev: torch.zeros(1, 32, 1), executor=LateExecutor(), clock=lambda: now[0]
    )
    for _ in range(16):
        controller.next_action({})
        controller.mark_action_executed()
    assert controller.next_action({}).item() == 8
    assert controller.deadline_misses == 1
    assert controller.wait_ms == [1000.0]
    controller.mark_action_executed()
    controller.close()


def test_reset_drains_running_worker_and_clears_after_failure():
    executor = ControlledExecutor()
    controller = StreamingController(lambda batch, prev: torch.zeros(1, 32, 1), executor=executor)
    for _ in range(9):
        controller.next_action({})
        controller.mark_action_executed()
    pending = executor.calls[1][0]
    pending.set_running_or_notify_cancel()
    pending.set_exception(RuntimeError("running worker failed"))
    with pytest.raises(RuntimeError, match="running worker failed"):
        controller.reset()
    assert controller.count_queued_actions() == 0
    controller.close()
    assert executor.closed


def test_prefix_target_at_start_and_later_launch():
    chunk = torch.arange(32.0).reshape(1, 32, 1)
    assert prefix_target(chunk, 8)[0, :16, 0].tolist() == list(range(8, 24))
    assert prefix_target(chunk, 16)[0, :16, 0].tolist() == list(range(16, 32))
    assert not prefix_target(chunk, 32).any()


def test_checkpoint_mapping_rejects_metadata_keys_shapes_and_dtypes():
    payload = {
        "training_rtc_phase": PHASE,
        "training_rtc_architecture": ARCHITECTURE,
        "torch_dtype": "bfloat16",
        "step": 1,
        "proprio_encoder": {},
        "mot": {"mixtures.action.blocks.0.ffn.0.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
    }
    key = "model.mot.layers.0.blocks.action.ffn.0.weight"
    target = {key: torch.empty(2, 2, dtype=torch.bfloat16)}
    assert torch.equal(map_checkpoint(payload, target)[key], payload["mot"][next(iter(payload["mot"]))])
    with pytest.raises(ValueError, match="shape/dtype"):
        map_checkpoint(payload, {key: torch.empty(2, 2)})
    with pytest.raises(ValueError, match="key mismatch"):
        map_checkpoint(payload, {})
    with pytest.raises(ValueError, match="phase mismatch"):
        map_checkpoint({**payload, "training_rtc_phase": "wrong"}, target)


def test_streaming_processors_clamp_persist_and_handle_gripper_tie(tmp_path):
    from lerobot.configs import NormalizationMode
    from lerobot.policies import make_pre_post_processors

    config = tiny_config()
    config.normalization_mapping.update(STATE=NormalizationMode.MIN_MAX, ACTION=NormalizationMode.MIN_MAX)
    config.toggle_action_dimensions = [-1]
    stats = {
        "observation.state": {"min": torch.zeros(2), "max": torch.ones(2)},
        "action": {"min": torch.zeros(3), "max": torch.ones(3)},
    }
    pre, post = make_fastwam_pre_post_processors(config, stats)
    assert torch.equal(
        pre({"observation.state": torch.tensor([10.0, -10.0])})["observation.state"],
        torch.tensor([[5.0, -5.0]]),
    )
    expected = torch.tensor([[1.5, -0.5, 1.0]])
    assert torch.equal(post(torch.tensor([[2.0, -2.0, 0.0]])), expected)
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    _, restored = make_pre_post_processors(config, pretrained_path=tmp_path)
    assert torch.equal(restored(torch.tensor([[2.0, -2.0, 0.0]])), expected)


def test_consistency_boundaries_at_clean_and_unit_sigma():
    action = torch.randn(1, 32, 3)
    velocity = torch.randn_like(action)
    sigma = torch.ones(1, 32)
    sigma[:, :8] = 0
    result = action_consistency_boundary(action, velocity, sigma)
    assert torch.equal(result[:, :8], action[:, :8])
    assert torch.equal(result[:, 8:], (action - velocity)[:, 8:])
    video = torch.randn(1, 2, 3, 2, 2)
    result = video_consistency_boundary(video, torch.ones_like(video), torch.tensor([[0.0, 1.0, 1.0]]))
    assert torch.equal(result[:, :, :1], video[:, :, :1])
    expected = 0.2 * video[:, :, 1:] + (0.5 / 1.25**0.5) * (video[:, :, 1:] - 1)
    torch.testing.assert_close(result[:, :, 1:], expected)


def test_policy_streaming_ack_reset_and_close(tiny_components):
    policy = FastWAMPolicy(tiny_config())
    first = policy.select_action(observation())
    assert first.shape == (1, 3)
    assert policy.count_queued_actions() == 15
    with pytest.raises(RuntimeError, match="mark_action_executed"):
        policy.select_action(observation())
    policy.mark_action_executed()
    policy.drop_queued_actions()
    assert policy.count_queued_actions() == 0
    assert torch.equal(first, policy.select_action(observation()))
    policy.reset()
    assert policy.count_queued_actions() == 0
    policy.close()
