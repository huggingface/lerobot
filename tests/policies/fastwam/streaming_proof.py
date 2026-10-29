# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
"""Manual CPU proof, outside pytest collection. No downloads or GPU operations.

Run under an RSS/swap watchdog. Supply pinned reference files via --reference,
the converted native policy via --native, and the released model.pt via --release.
--tiny validates the reference bridge with a random tiny CPU policy instead.
The proof uses explicit shared latent/text fixtures and reports this limitation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from lerobot.policies.fastwam.convert_streaming_checkpoint import load_stats, native_schema, release_config
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy, _input_image_from_batch
from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
from lerobot.policies.fastwam.streaming import StreamingController, prefix_target
from lerobot.policies.fastwam.wan import ActionDiT, FastWAM, MoT, WanVideoDiT
from lerobot.policies.fastwam.wan.model import rope_params
from lerobot.policies.fastwam.wan.modular import MoTLayer
from lerobot.policies.fastwam.wan.streaming import StreamingWAM
from lerobot.policies.fastwam.wan.video_dit import precompute_freqs_cis
from tests.policies.fastwam.streaming_reference import load_reference, reference_model


class FrozenLatents(nn.Module):
    """Inject a shared encoder fixture into both compared model paths."""

    upsampling_factor = 16
    temporal_downsample_factor = 4

    def __init__(self, latents: Tensor) -> None:
        super().__init__()
        self.latents = latents
        self.z_dim = latents.shape[1]

    def encode(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.latents.clone()


def error(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_absolute": difference.max().item(),
        "max_relative": (difference / expected.float().abs().clamp_min(1e-6)).max().item(),
    }


def rollout_functions(directory: Path) -> dict[str, Any]:
    """Load the pinned raw-input and postprocessing definitions unchanged."""
    source = directory / "examples_libero_rollout.py"
    names = {
        "_resize_rgb",
        "_obs_to_images",
        "_obs_to_image",
        "_stat_tensor",
        "_normalize_state",
        "_denormalize_action",
    }
    parsed = ast.parse(source.read_text())
    selected = [node for node in parsed.body if getattr(node, "name", None) in names]
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    namespace = {"np": np, "torch": torch, "Image": Image}
    exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), namespace)
    return namespace


def preprocessing(args, config) -> dict[str, Any]:
    """Run the pinned raw LIBERO preprocessing functions on a fixed RGB fixture."""
    namespace = rollout_functions(args.reference)
    recipe = types.SimpleNamespace(
        data=types.SimpleNamespace(
            video_size=(224, 224),
            video_keys=list(config.image_features),
            concat_multi_camera="horizontal",
            state_norm_mode="minmax",
            action_norm_mode="minmax",
        )
    )
    rng = np.random.default_rng(23)
    raw = {
        key: rng.integers(0, 256, (240, 256, 3), dtype=np.uint8)
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    }
    reference_image, resized = namespace["_obs_to_image"](
        raw, recipe, checkpoint_format="fastwam", device=torch.device("cpu"), dtype=torch.bfloat16
    )
    stats = load_stats(args.release.parent / "dataset_stats.json")
    pre, post = make_fastwam_pre_post_processors(config, stats)
    state = torch.linspace(-2, 2, 8)
    batch = pre(
        {
            "observation.images.image": torch.from_numpy(resized["image"].copy()).permute(2, 0, 1).float()
            / 255,
            "observation.images.wrist_image": torch.from_numpy(resized["wrist_image"].copy())
            .permute(2, 0, 1)
            .float()
            / 255,
            "observation.state": state,
        }
    )
    native_image = _input_image_from_batch(batch, config)

    class RecordingVAE:
        def encode(self, videos, **kwargs):
            self.pixels = videos[0][:, 0].unsqueeze(0)
            return [torch.zeros(48, 1, 14, 28, dtype=torch.bfloat16)]

    recorder = RecordingVAE()
    fixture = types.SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.bfloat16, vae=recorder)
    StreamingWAM._encode_input_image_latents_tensor(fixture, native_image)
    pixels = recorder.pixels
    normalized = namespace["_normalize_state"](state.unsqueeze(0), recipe, stats["observation.state"])
    action = torch.linspace(-2, 2, 32 * 7).reshape(1, 32, 7)
    expected = torch.from_numpy(
        namespace["_denormalize_action"](action, recipe, stats["action"], checkpoint_format="fastwam")
    ).unsqueeze(0)
    actual = post(action)
    checks = {
        "pixels": error(pixels, reference_image),
        "state": error(batch["observation.state"], normalized),
        "denormalized_action": error(actual, expected),
    }
    assert all(value["max_absolute"] < 1e-5 for value in checks.values()), checks
    return {
        **checks,
        "camera_orientation": "caller rotates LIBERO raw RGB 180 degrees before PIL bilinear resize",
        "frozen_encoder_limit": "range/order tested separately; shared latent/text fixtures bypass actual encoders",
    }


@contextmanager
def capture_native(core: nn.Module, traces: dict[str, Tensor]):
    original = MoTLayer._apply_expert_post_block
    selected = {0, len(core.mot.layers) // 2, len(core.mot.layers) - 1}
    owners = {
        id(block): (index, name)
        for index, layer in enumerate(core.mot.layers)
        for name, block in layer.blocks.items()
        if index in selected
    }

    def post(block, *args, **kwargs):
        output = original(block, *args, **kwargs)
        if id(block) in owners:
            index, name = owners[id(block)]
            if name == "action" and output.shape[1] == 16:
                name = "condition"
            traces[f"layer{index}.{name}"] = output.detach().clone()
        return output

    MoTLayer._apply_expert_post_block = staticmethod(post)
    heads = []
    for name, expert in (("video", core.video_expert), ("action", core.action_expert)):
        original_post = expert.post_dit

        def head(*args, _original=original_post, _name=name, **kwargs):
            output = _original(*args, **kwargs)
            traces[f"{_name}_velocity"] = output.detach().clone()
            return output

        heads.append((expert, original_post))
        expert.post_dit = head
    try:
        yield
    finally:
        MoTLayer._apply_expert_post_block = staticmethod(original)
        for expert, original_post in heads:
            expert.post_dit = original_post


def capture_reference(model: nn.Module, traces: dict[str, Tensor]) -> None:
    for name, expert in model.mot.experts.items():
        selected = {0, len(expert.blocks) // 2, len(expert.blocks) - 1}
        for index in selected:
            block = expert.blocks[index]
            original = block.post_attention

            def post(*args, _original=original, _name=name, _index=index, **kwargs):
                output = _original(*args, **kwargs)
                stream = "condition" if _name == "action" and output.shape[1] == 16 else _name
                traces[f"layer{_index}.{stream}"] = output.detach().clone()
                return output

            block.post_attention = post
        original_post = expert.post_dit

        def head(*args, _original=original_post, _name=name, **kwargs):
            output = _original(*args, **kwargs)
            traces[f"{_name}_velocity"] = output.detach().clone()
            return output

        expert.post_dit = head


def native_core(config, state: dict[str, Tensor], first: Tensor) -> nn.Module:
    container = native_schema(config)
    container.load_state_dict(state, strict=True, assign=True)
    core = container.model
    core.device = torch.device("cpu")
    object.__setattr__(core, "vae", FrozenLatents(first))
    return core.eval()


def queue_parity(reference, reference_chunks: list[Tensor], native_chunks: list[Tensor]) -> dict[str, Any]:
    ref_index = iter(reference_chunks)

    def predict_ref(obs, previous, delay):
        actions = next(ref_index)[0]
        return reference.inference.ACStreamPrediction(actions.numpy(), actions, 0, 0)

    native_index = iter(native_chunks)
    ref = reference.inference.ACStreamController(predict_ref)
    native = StreamingController(lambda batch, previous: next(native_index))
    ref.start_episode({})
    reference_trace, native_trace = [], []
    try:
        for _ in range(40):
            reference_trace.append(torch.from_numpy(ref.next_action({})))
            native_trace.append(native.next_action({})[0])
            ref.mark_action_executed()
            native.mark_action_executed()
    finally:
        ref.close()
        native.close()
    return error(torch.stack(native_trace), torch.stack(reference_trace))


def parity(args) -> None:
    reference = load_reference(args.reference)
    config = release_config()
    if args.tiny:
        from tests.policies.fastwam.test_streaming_wam import tiny_config

        config = tiny_config()
        schema = native_schema(config)
        state = {
            key: torch.randn(value.shape, dtype=value.dtype) * 0.05
            for key, value in schema.state_dict().items()
        }
        payload = {"mot": {}, "proprio_encoder": {}}
        for key, value in state.items():
            if key.startswith("model.proprio_encoder."):
                payload["proprio_encoder"][key.removeprefix("model.proprio_encoder.")] = value
            elif key.startswith("model.mot.layers."):
                parts = key.split(".")
                payload["mot"][f"mixtures.{parts[5]}.blocks.{parts[3]}." + ".".join(parts[6:])] = value
            else:
                payload["mot"][key.removeprefix("model.mot.")] = value
    else:
        from safetensors.torch import load_file

        state = load_file(str(args.native / "model.safetensors"))
        payload = torch.load(args.release, map_location="cpu", weights_only=True, mmap=True)
    dtype = config.dtype
    height, width = config.image_size
    generator = torch.Generator().manual_seed(123)
    first = torch.randn(
        1,
        config.video_dit_config["in_dim"],
        1,
        height // 16,
        width // 16,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    image_bytes = torch.randint(0, 256, (1, 3, height, width), generator=generator, dtype=torch.uint8)
    image = image_bytes.float() / 255.0
    context = torch.randn(1, 128, config.video_dit_config["text_dim"], generator=generator).to(dtype)
    mask = torch.ones(1, 128, dtype=torch.bool)
    proprio = torch.linspace(-0.5, 0.5, config.proprio_dim).unsqueeze(0).to(dtype)
    explicit_noise = reference.consistency.sample_joint_consistency_noise(
        (1, first.shape[1], 3, first.shape[3], first.shape[4]),
        (1, 32, config.action_dim),
        seed=42,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    reference_model_instance = reference_model(reference, config, payload, first)
    traces: dict[str, Tensor] = {}
    capture_reference(reference_model_instance, traces)
    reference_outputs = []
    reference_chunks = []
    previous = None
    for stage in range(3):
        traces.clear()
        output = reference_model_instance.infer_action(
            input_image=image_bytes.to(dtype) * (2.0 / 255.0) - 1.0,
            context=context,
            context_mask=mask,
            action_horizon=32,
            num_video_frames=9,
            num_inference_steps=1,
            seed=42,
            proprio=proprio,
            ac_stream_prev_action_chunk=previous,
            ac_stream_inference_delay=0 if stage == 0 else 8,
        )
        reference_chunks.append(output.float().clone())
        reference_outputs.append({**traces, "action": output.float().clone()})
        previous = prefix_target(output, 8 if stage == 0 else 16)
        print(f"reference stage {stage} completed", flush=True)
    del reference_model_instance, traces, payload
    gc.collect()
    core = native_core(config, state, first)
    if not args.tiny:
        stats = load_stats(args.release.parent / "dataset_stats.json")
        _, postprocessor = make_fastwam_pre_post_processors(config, stats)
        denormalize = rollout_functions(args.reference)["_denormalize_action"]
        recipe = types.SimpleNamespace(data=types.SimpleNamespace(action_norm_mode="minmax"))
    native_chunks, metrics = [], []
    previous = None
    all_close = True
    for stage in range(3):
        traces = {}
        with capture_native(core, traces):
            output = core.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=32,
                context=context,
                context_mask=mask,
                proprio=proprio,
                seed=42,
                previous_action_chunk=previous,
                video_noise=explicit_noise[0],
                action_noise=explicit_noise[1],
            )
        traces["action"] = output["action"].unsqueeze(0)
        native_chunks.append(traces["action"].clone())
        stage_metrics = {}
        for key in traces:
            stage_metrics[key] = error(traces[key], reference_outputs[stage][key])
            tolerance = 0.02 if key == "action" else 0.125
            all_close &= torch.allclose(
                traces[key].float(), reference_outputs[stage][key].float(), atol=tolerance, rtol=0.02
            )
        if previous is not None:
            assert torch.equal(native_chunks[-1][:, :8], previous[:, :8].float())
        if not args.tiny:
            expected = torch.from_numpy(
                denormalize(reference_chunks[stage], recipe, stats["action"], checkpoint_format="fastwam")
            ).unsqueeze(0)
            actual = postprocessor(native_chunks[-1])
            stage_metrics["denormalized_action"] = error(actual, expected)
            all_close &= torch.allclose(actual, expected, atol=0.02, rtol=0.02)
        metrics.append(stage_metrics)
        previous = prefix_target(reference_chunks[stage].to(dtype), 8 if stage == 0 else 16)
        print(f"native stage {stage} completed: {stage_metrics['action']}", flush=True)
    queue_metrics = queue_parity(reference, reference_chunks, native_chunks)
    all_close &= queue_metrics["max_absolute"] <= 0.02
    report = {
        "label": "Dry run",
        "weights": "random tiny" if args.tiny else "released",
        "dtype": str(dtype),
        "sdpa_backend": "math",
        "threads": torch.get_num_threads(),
        "passed": all_close,
        "tolerances": {"action_atol": 0.02, "layer_atol": 0.125, "rtol": 0.02},
        "fixture": "shared synthetic RGB, normalized proprio, latent and cached text tensors; no encoder parity claim",
        "stages": metrics,
        "queue_trace": queue_metrics,
        "reference_sha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in args.reference.glob("*.py")
        },
    }
    if not args.tiny:
        report["preprocessing"] = preprocessing(args, config)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all_close:
        raise RuntimeError("Released-checkpoint parity exceeded the predetermined tolerances.")


def synchronous_core(config, state: dict[str, Tensor], first: Tensor) -> nn.Module:
    """Use the existing synchronous FastWAM implementation with identical weights."""
    with torch.device("meta"):
        video = WanVideoDiT(**config.video_dit_config)
        action = ActionDiT(**config.action_dit_config)
        mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
        with torch.device("cpu"):
            core = FastWAM(
                video,
                action,
                mot,
                nn.Identity(),
                text_dim=video.text_dim,
                proprio_dim=config.proprio_dim,
                device="meta",
                torch_dtype=config.dtype,
            )
        container = nn.Module()
        container.model = core
        container.to(dtype=config.dtype)
    filtered = {key: value for key, value in state.items() if "rtc_1step_selfatt_z1_slot_encoder" not in key}
    container.load_state_dict(filtered, strict=True, assign=True)
    dimension = video.hidden_dim // video.num_heads
    with torch.device("cpu"):
        video.freqs = torch.cat(
            [
                rope_params(1024, dimension - 4 * (dimension // 6)),
                rope_params(1024, 2 * (dimension // 6)),
                rope_params(1024, 2 * (dimension // 6)),
            ],
            dim=1,
        )
        action.freqs = precompute_freqs_cis(action.attn_head_dim, end=1024)
    core.device = torch.device("cpu")
    object.__setattr__(core, "vae", FrozenLatents(first))
    return core.eval()


def timing(args) -> None:
    """Short CPU control-loop comparison. Cadence is simulated, with real compute."""
    from safetensors.torch import load_file

    config = release_config()
    if args.tiny:
        from tests.policies.fastwam.test_streaming_wam import tiny_config

        config = tiny_config()
        state = {
            key: torch.randn(value.shape, dtype=value.dtype) * 0.05
            for key, value in native_schema(config).state_dict().items()
        }
    else:
        state = load_file(str(args.native / "model.safetensors"))
    generator = torch.Generator().manual_seed(123)
    dtype = config.dtype
    height, width = config.image_size
    first = torch.randn(
        1, config.video_dit_config["in_dim"], 1, height // 16, width // 16, generator=generator
    ).to(dtype)
    batch = {
        "input_image": torch.rand(1, 3, height, width, generator=generator),
        "context": torch.randn(1, 128, config.video_dit_config["text_dim"], generator=generator).to(dtype),
        "context_mask": torch.ones(1, 128, dtype=torch.bool),
        "observation.state": torch.linspace(-0.5, 0.5, config.proprio_dim).unsqueeze(0),
    }
    results = {}
    for mode in ("current_synchronous_fastwam", "synchronous_streaming_wam", "asynchronous_streaming_wam"):
        settings = copy.deepcopy(config)
        baseline = mode == "current_synchronous_fastwam"
        asynchronous = mode == "asynchronous_streaming_wam"
        settings.streaming_wam = asynchronous
        settings.num_inference_steps = 10 if baseline else 1
        if baseline:
            settings.video_dit_config["fp32_attention"] = True
            settings.action_dit_config["fp32_attention"] = True
            core = synchronous_core(settings, state, first)
        else:
            core = native_core(settings, state, first)
        with patch.object(FastWAMPolicy, "_build_core_model", return_value=core):
            policy = FastWAMPolicy(settings)
        inference_ms = []
        original = core.infer_action

        def measured(*values, _original=original, _baseline=baseline, _samples=inference_ms, **kwargs):
            # Current FastWAM uses FP32 time embeddings; CPU BF16 needs autocast.
            # Both Streaming-WAM modes use their explicit parameter-dtype convention.
            started = time.perf_counter()
            with (
                torch.inference_mode(),
                sdpa_kernel(SDPBackend.MATH),
                torch.autocast("cpu", dtype=torch.bfloat16, enabled=_baseline and dtype == torch.bfloat16),
            ):
                output = _original(*values, **kwargs)
            _samples.append((time.perf_counter() - started) * 1000)
            return output

        core.infer_action = measured
        selections = []
        periods = []
        count = 40 if asynchronous else 64
        started = time.perf_counter()
        try:
            for _ in range(count):
                before = time.perf_counter()
                action = policy.select_action(batch)
                assert action.shape == (1, config.action_dim) and torch.isfinite(action).all()
                selections.append((time.perf_counter() - before) * 1000)
                # Each acknowledgement represents a simulated action execution interval.
                time.sleep(max(0, args.period - (time.perf_counter() - before)))
                if asynchronous:
                    policy.mark_action_executed()
                periods.append((time.perf_counter() - before) * 1000)
            elapsed = time.perf_counter() - started
            controller = getattr(policy, "_streaming_controller", None)
            results[mode] = {
                "actions": count,
                "wall_seconds": elapsed,
                "actions_per_second": count / elapsed,
                "cold_inference_ms": inference_ms[0],
                "warm_inference_ms": inference_ms[1:],
                "cold_select_ms": selections[0],
                "warm_select_max_ms": max(selections[1:]),
                "selection_ms": selections,
                "stalls_over_period": sum(value > args.period * 1000 for value in selections),
                "action_period_ms": periods,
                "control_deadline_misses": sum(value > args.period * 1000 + 1 for value in periods),
                "deadline_timer_tolerance_ms": 1,
                "continuation_deadline_misses": controller.deadline_misses if controller else None,
                "continuation_wait_ms": list(controller.wait_ms) if controller else [],
                "inference_steps": settings.num_inference_steps,
            }
            print(
                f"{mode}: cold {inference_ms[0]:.3f} ms, warm {inference_ms[1:]}, "
                f"{count / elapsed:.3f} actions/s, misses {results[mode]['control_deadline_misses']}",
                flush=True,
            )
        finally:
            policy.close()
        del policy, core, original, measured
        gc.collect()
    args.output.write_text(
        json.dumps(
            {
                "label": "Dry run",
                "weights": "random tiny" if args.tiny else "released",
                "results": results,
                "threads": torch.get_num_threads(),
                "dtype": str(dtype),
                "sdpa_backend": "math",
                "action_period_seconds_simulated": args.period,
                "fixture": "same released weights, synthetic cached latents/text and fixed normalized proprio; excludes frozen encoders",
                "baseline": "current synchronous FastWAM select_action, ten steps, FP32 attention, CPU BF16 autocast",
                "limit": "short CPU samples; no deployment, arm, encoder, LIBERO-success or CUDA latency claim",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--preprocessing-only", action="store_true")
    parser.add_argument("--period", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(23)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        if args.preprocessing_only:
            args.output.write_text(json.dumps(preprocessing(args, release_config()), indent=2) + "\n")
        elif args.timing:
            timing(args)
        else:
            parity(args)


if __name__ == "__main__":
    main()
