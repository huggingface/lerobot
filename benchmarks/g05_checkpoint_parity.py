#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compare a converted G0.5 checkpoint with the pinned author implementation.

This is intentionally an opt-in checkpoint test: it requires an accepted gated
checkpoint, the pinned GalaxeaVLA source checkout, and its CUDA dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.g05.modeling_g05 import G05Policy
from lerobot.utils.constants import ACTION, OBS_STATE


def _tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    actual = actual.detach().float().cpu()
    if reference.shape != actual.shape:
        raise AssertionError(f"shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    error = (reference - actual).abs().flatten()
    return {
        "max": error.max().item() if error.numel() else 0.0,
        "p99": torch.quantile(error, 0.99).item() if error.numel() else 0.0,
        "mean": error.mean().item() if error.numel() else 0.0,
    }


def _assert_bool_equal(name: str, reference: torch.Tensor, actual: torch.Tensor) -> None:
    if not torch.equal(reference.detach().cpu().bool(), actual.detach().cpu().bool()):
        raise AssertionError(f"{name} differs")


def _move_to(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to(item, device) for item in value)
    return value


def _raw_sample(processor: Any, index: int, task: str) -> dict[str, Any]:
    images: dict[str, torch.Tensor] = {}
    for camera_index, meta in enumerate(processor.shape_meta["images"]):
        channels, height, width = meta["raw_shape"]
        values = torch.arange(channels * height * width, dtype=torch.int64)
        images[meta["key"]] = (
            (values.reshape(channels, height, width) + index * 37 + camera_index * 71)
            .remainder(256)
            .to(torch.uint8)
        )

    state = {}
    for part_index, meta in enumerate(processor.shape_meta["state"]):
        width = int(meta["raw_shape"])
        state[meta["key"]] = torch.linspace(
            -0.2 + 0.03 * index + 0.01 * part_index,
            0.2 + 0.03 * index + 0.01 * part_index,
            width,
        )

    action = {}
    horizon = int(processor.action_horizon)
    for part_index, meta in enumerate(processor.shape_meta["action"]):
        width = int(meta["raw_shape"])
        action[meta["key"]] = torch.linspace(
            -0.1 + 0.02 * index + 0.01 * part_index,
            0.1 + 0.02 * index + 0.01 * part_index,
            horizon * width,
        ).reshape(horizon, width)

    return {
        "images": {
            key: value.unsqueeze(0).expand(processor.num_obs_steps, -1, -1, -1)
            for key, value in images.items()
        },
        "state": {
            key: value.unsqueeze(0).expand(processor.num_obs_steps, -1) for key, value in state.items()
        },
        "action": action,
        "action_is_pad": torch.zeros(horizon, dtype=torch.bool),
        "state_is_pad": torch.zeros(processor.num_obs_steps, dtype=torch.bool),
        "image_is_pad": torch.zeros(processor.num_obs_steps, dtype=torch.bool),
        "task": task,
        "frequency": 15.0,
        "idx": index,
    }


def _lerobot_input(raw: dict[str, Any], processor: Any, config: G05Config) -> dict[str, Any]:
    image_pairs = zip(processor.shape_meta["images"], config.camera_order, strict=True)
    return {
        OBS_STATE: torch.cat(
            [raw["state"][meta["key"]][-1] for meta in processor.shape_meta["state"]], dim=-1
        ),
        ACTION: torch.cat([raw["action"][meta["key"]] for meta in processor.shape_meta["action"]], dim=-1),
        **{lerobot_key: raw["images"][meta["key"]][-1] for meta, lerobot_key in image_pairs},
        "action_is_pad": raw["action_is_pad"],
        "task": raw["task"],
    }


def _collate_lerobot(samples: list[dict[str, Any]], config: G05Config) -> dict[str, Any]:
    result: dict[str, Any] = {
        OBS_STATE: torch.cat([sample[OBS_STATE] for sample in samples], dim=0),
        ACTION: torch.stack([sample[ACTION] for sample in samples], dim=0),
        "task": [
            sample["task"][0] if isinstance(sample["task"], list) else sample["task"] for sample in samples
        ],
    }
    for key in (*config.camera_order, "proprio_dim_is_pad", "action_dim_is_pad"):
        result[key] = torch.cat([sample[key] for sample in samples], dim=0)
    action_pad = [sample["action_is_pad"] for sample in samples]
    result["action_is_pad"] = torch.stack(
        [value.squeeze(0) if value.ndim == 2 else value for value in action_pad], dim=0
    )
    return result


def _flatten_author_action(action: dict[str, torch.Tensor], processor: Any) -> torch.Tensor:
    return torch.cat([action[meta["key"]] for meta in processor.shape_meta["action"]], dim=-1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.author_source / "src"))
    from g05.utils.data.data_utils import collate_fn_pad_sequences
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.data.processor_utils import build_processors
    from omegaconf import OmegaConf

    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    if not OmegaConf.has_resolver("oc.load"):

        def _oc_load(path: str, key: str | None = None) -> Any:
            loaded = OmegaConf.load(args.author_source / path)
            return OmegaConf.select(loaded, key) if key is not None else loaded

        OmegaConf.register_new_resolver(
            "oc.load",
            _oc_load,
        )
    author_cfg = OmegaConf.load(checkpoint / "author_config.yaml")
    author_processors = build_processors(author_cfg)
    author_processors.set_normalizer_from_stats(
        load_dataset_stats_from_json(checkpoint / "g05_dataset_stats.json")
    )
    author_processors.eval()
    author_processor = author_processors.processors[args.embodiment]
    author_processor.action_horizon = int(author_cfg.data.action_size)

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, G05Config):
        raise TypeError(f"Expected G05Config, got {type(config).__name__}")
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint)

    tasks = [
        "  Pick café cup\nverbatim  ",
        "第二个 task — keep Unicode and whitespace\t",
    ][: args.batch_size]
    raw_samples = [_raw_sample(author_processor, index, task) for index, task in enumerate(tasks)]
    author_samples = [author_processor.preprocess(copy.deepcopy(sample)) for sample in raw_samples]
    author_batch = collate_fn_pad_sequences(copy.deepcopy(author_samples))
    lerobot_samples = [
        preprocessor(_lerobot_input(sample, author_processor, config)) for sample in raw_samples
    ]
    lerobot_batch = _collate_lerobot(lerobot_samples, config)

    policy = G05Policy.from_pretrained(
        checkpoint,
        local_files_only=True,
        strict=True,
    ).to(device)
    policy.eval()
    port_author_batch = policy._prepare_author_batch(lerobot_batch)

    report: dict[str, Any] = {
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(next(policy.parameters()).dtype),
        "prompt_exact": all(
            left["template"] == right["template"]
            and left["command"] == right["command"]
            and left["embodiment"] == right["embodiment"]
            for left, right in zip(author_batch["samples"], port_author_batch["samples"], strict=True)
        ),
    }
    if not report["prompt_exact"]:
        raise AssertionError("author and LeRobot prompt payloads differ")

    image_errors = {}
    for (author_key, author_images), (port_key, port_images) in zip(
        author_batch["pixel_values"].items(),
        port_author_batch["pixel_values"].items(),
        strict=True,
    ):
        image_errors[f"{author_key}->{port_key}"] = _tensor_error(author_images, port_images)
    report["images"] = image_errors

    author_proprio = torch.stack([sample["proprio"]["value"] for sample in author_batch["samples"]], dim=0)
    port_proprio = torch.stack([sample["proprio"]["value"] for sample in port_author_batch["samples"]], dim=0)
    report["proprio"] = _tensor_error(author_proprio, port_proprio)
    report["normalized_input_action"] = _tensor_error(author_batch[ACTION], lerobot_batch[ACTION])
    for index, (author_sample, port_sample) in enumerate(
        zip(author_batch["samples"], port_author_batch["samples"], strict=True)
    ):
        _assert_bool_equal(
            f"proprio_dim_is_pad[{index}]",
            author_sample["proprio"]["proprio_dim_is_pad"],
            port_sample["proprio"]["proprio_dim_is_pad"],
        )
    _assert_bool_equal(
        "action_dim_is_pad", author_batch["action_dim_is_pad"], port_author_batch["action_dim_is_pad"]
    )
    report["masks_exact"] = True

    author_ids, author_attention = policy.backend.processor.encode_inference(
        copy.deepcopy(author_batch["samples"]), device=device, mode="fm"
    )
    port_ids, port_attention = policy.backend.processor.encode_inference(
        copy.deepcopy(port_author_batch["samples"]), device=device, mode="fm"
    )
    _assert_bool_equal("input_ids", author_ids, port_ids)
    _assert_bool_equal("attention_mask", author_attention, port_attention)
    report["tokens_exact"] = True
    report["token_shape"] = list(author_ids.shape)

    author_cuda = _move_to(copy.deepcopy(author_batch), device)
    port_cuda = _move_to(lerobot_batch, device)
    torch.manual_seed(args.seed)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.model_weights_to_bf16 and device.type == "cuda",
        ),
    ):
        author_output = policy.backend.predict_action(author_cuda)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        port_action = policy.predict_action_chunk(port_cuda)
    report["normalized_action"] = _tensor_error(author_output[ACTION], port_action)

    author_post = author_processor.postprocess(
        {
            ACTION: author_output[ACTION].detach().cpu(),
            "proprio": author_cuda["proprio"].detach().cpu(),
            "action_dim_is_pad": author_cuda.get("action_dim_is_pad"),
            "proprio_dim_is_pad": author_cuda.get("proprio_dim_is_pad"),
        }
    )
    author_env_action = _flatten_author_action(author_post[ACTION], author_processor)
    # Isolate processor parity from small repeated BF16 sampling drift by feeding
    # both postprocessors the same normalized author action chunk.
    port_env_action = postprocessor(author_output[ACTION])
    report["environment_action"] = _tensor_error(author_env_action, port_env_action)

    if args.compare_training_loss:
        policy.train()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            author_loss, _ = policy.backend(_move_to(copy.deepcopy(author_batch), device))
        torch.manual_seed(args.seed)
        with torch.no_grad():
            port_loss, _ = policy(_move_to(lerobot_batch, device))
        report["training_loss"] = {
            "author": author_loss.detach().float().item(),
            "lerobot": port_loss.detach().float().item(),
            "absolute_error": abs(author_loss.detach().float().item() - port_loss.detach().float().item()),
        }

    numeric_sections = (
        *report["images"].values(),
        report["proprio"],
        report["normalized_input_action"],
        report["normalized_action"],
        report["environment_action"],
    )
    report["tolerance"] = args.atol
    report["passed"] = all(section["max"] <= args.atol for section in numeric_sections)
    if "training_loss" in report:
        report["passed"] &= report["training_loss"]["absolute_error"] <= args.atol
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", default="libero")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--compare-training-loss", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{payload}\n")
    print(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
