#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Deterministically package a user-authorized G0.5 checkpoint for LeRobot.

This command never downloads from the gated Hub. The user supplies a local checkpoint
after accepting Galaxea's license.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from lerobot.policies.g05.configuration_g05 import (
    G05_CAMERA_PROFILES,
    G05_HUB_REVISION,
    G05_SOURCE_REVISION,
    G05Config,
)
from lerobot.policies.g05.processor_g05 import make_g05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

_EXACT_RENAMES = {
    "model.embed_tokens.weight": "model.vlm.input_proj.weight",
}
_PREFIX_RENAMES = (
    ("model.joint_model.mixtures.vlm.", "model.vlm."),
    ("model.joint_model.mixtures.action.", "model.action_expert."),
    ("model.action_encoder.", "model.action_expert.input_proj."),
    ("model.action_decoder.", "model.action_expert.output_proj."),
)
_REQUIRED_PREFIXES = (
    "backend.model.vlm.",
    "backend.model.vision_tower.",
    "backend.model.action_expert.",
)


@dataclass
class ConversionReport:
    mapped: dict[str, str] = field(default_factory=dict)
    shared_aliases: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    duplicate: list[str] = field(default_factory=list)
    shape_mismatched: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def fail_if_invalid(self) -> None:
        if self.missing or self.unexpected or self.duplicate or self.shape_mismatched:
            raise ValueError(
                "G0.5 conversion failed strict validation: "
                f"missing={len(self.missing)}, unexpected={len(self.unexpected)}, "
                f"duplicate={len(self.duplicate)}, shape_mismatched={len(self.shape_mismatched)}"
            )


def _mapped_key(key: str) -> str:
    key = _EXACT_RENAMES.get(key, key)
    for old, new in _PREFIX_RENAMES:
        if key.startswith(old):
            key = new + key.removeprefix(old)
            break
    return key if key.startswith("backend.") else f"backend.{key}"


def convert_state_dict(
    source: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """Map every tensor exactly once and optionally validate a target state dict."""

    converted: dict[str, torch.Tensor] = {}
    report = ConversionReport()
    for old_key in sorted(source):
        value = source[old_key]
        if not isinstance(value, torch.Tensor):
            report.unexpected.append(old_key)
            continue
        new_key = _mapped_key(old_key)
        if new_key in converted:
            report.duplicate.append(new_key)
            continue
        converted[new_key] = value.detach().cpu().contiguous()
        report.mapped[old_key] = new_key

    if expected is not None:
        report.missing = sorted(set(expected) - set(converted))
        report.unexpected.extend(sorted(set(converted) - set(expected)))
        for key in sorted(set(expected) & set(converted)):
            if expected[key].shape != converted[key].shape:
                report.shape_mismatched[key] = {
                    "source": list(converted[key].shape),
                    "expected": list(expected[key].shape),
                }
    else:
        for prefix in _REQUIRED_PREFIXES:
            if not any(key.startswith(prefix) for key in converted):
                report.missing.append(f"{prefix}*")
    return converted, report


def save_converted_state_dict(state_dict: dict[str, torch.Tensor], path: Path) -> dict[str, str]:
    """Save exact tensor aliases once, matching safetensors' strict model loader."""

    aliases: dict[str, str] = {}
    unique: dict[str, torch.Tensor] = {}
    seen: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    for key in sorted(state_dict):
        tensor = state_dict[key]
        identity = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if identity in seen:
            aliases[key] = seen[identity]
        else:
            seen[identity] = key
            unique[key] = tensor
    save_file(unique, path, metadata=aliases or None)
    return aliases


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state-dict mapping in {path}.")
    return payload


def _profile_config(
    profile: str,
    hydra: dict[str, Any],
    embodiment: str | None = None,
    action_head: str | None = None,
) -> G05Config:
    model = hydra.get("model", {})
    arch = model.get("model_arch", {})
    data = hydra.get("data", {})
    fixed_embodiment = {
        "g05-libero": "libero",
        "g05-robotwin20": "robotwin20",
    }.get(profile)
    if fixed_embodiment is not None:
        if embodiment is not None and embodiment != fixed_embodiment:
            raise ValueError(f"{profile} requires embodiment={fixed_embodiment!r}.")
        embodiment = fixed_embodiment
    elif profile == "g05-base":
        available = sorted((data.get("processors") or {}).keys())
        if embodiment is None:
            raise ValueError(f"g05-base requires a concrete --embodiment; choose one of {available}.")
        if embodiment not in available:
            raise ValueError(f"g05-base embodiment must be one of {available}, got {embodiment!r}.")
    else:
        raise ValueError(f"Unsupported G0.5 profile {profile!r}.")
    assert embodiment is not None
    data_processor = (data.get("processors") or {}).get(embodiment, {})
    # Match the author's build_processors merge order: embodiment data first,
    # then task-level model.processor overrides.
    processor_metadata = {**data_processor, **model.get("processor", {})}
    horizon = int(data.get("action_size", arch.get("horizon_steps", 16)))
    predict_cot = bool(arch.get("predict_cot", False))
    discrete = bool(arch.get("discrete_action", True))
    continuous = bool(arch.get("continuous_action", False))
    if action_head is None:
        action_head = "flow" if continuous else "actioncodec"
    if action_head not in {"flow", "actioncodec"}:
        raise ValueError("action_head must be 'flow' or 'actioncodec'.")
    if action_head == "flow" and not continuous:
        raise ValueError("The selected checkpoint does not enable the continuous flow action head.")
    if action_head == "actioncodec" and not discrete:
        raise ValueError("The selected checkpoint does not enable the autoregressive ActionCodec head.")
    if profile != "g05-base" and action_head != "flow":
        raise ValueError(f"The released {profile} checkpoint supports only --action-head flow.")
    norm_name = str(processor_metadata.get("norm_default_mode", "")).lower()
    checkpoint_normalization = {
        "q01/q99": "q01_q99",
        "z-score": "z_score",
        "z-score-tail": "z_score_tail_mixed",
        "identity": "identity",
        "dummy": "identity",
    }.get(norm_name)
    arch = dict(arch)
    arch.pop("_target_", None)
    num_input_images = int(arch.get("num_input_images", len(G05_CAMERA_PROFILES[embodiment])))
    if profile == "g05-libero":
        return G05Config(
            checkpoint_profile=profile,
            embodiment="libero",
            action_head="flow",
            runtime_system="system1",
            predict_cot=predict_cot,
            discrete_action=discrete,
            continuous_action=continuous,
            return_continuous_action=True,
            chunk_size=horizon,
            normalization_mode="q01_q99",
            normalization_clip=(-5.0, 5.0),
            use_stepwise_action_norm=True,
            camera_order=G05_CAMERA_PROFILES["libero"],
            num_input_images=num_input_images,
            author_model_config=arch,
            processor_metadata=processor_metadata,
            action_codec_metadata=model.get("tokenizer", hydra.get("tokenizer", {})),
            author_source_revision=G05_SOURCE_REVISION,
            source_checkpoint_revision=G05_HUB_REVISION,
            license="other",
            tags=["g05", "robotics", "non-commercial"],
        )
    if profile == "g05-robotwin20":
        return G05Config(
            checkpoint_profile=profile,
            embodiment="robotwin20",
            action_head="flow",
            runtime_system="system1",
            predict_cot=predict_cot,
            discrete_action=discrete,
            continuous_action=continuous,
            return_continuous_action=True,
            raw_state_dim=14,
            raw_action_dim=14,
            chunk_size=horizon,
            normalization_mode="q01_q99",
            normalization_clip=(-5.0, 5.0),
            use_stepwise_action_norm=True,
            camera_order=G05_CAMERA_PROFILES["robotwin20"],
            num_input_images=num_input_images,
            author_model_config=arch,
            processor_metadata=processor_metadata,
            action_codec_metadata=model.get("tokenizer", hydra.get("tokenizer", {})),
            author_source_revision=G05_SOURCE_REVISION,
            source_checkpoint_revision=G05_HUB_REVISION,
            license="other",
            tags=["g05", "robotics", "non-commercial"],
        )
    if checkpoint_normalization != "z_score_tail_mixed":
        raise ValueError("The pinned g05-base R1 contract requires z-score-tail normalization.")
    exceptions = processor_metadata.get("norm_exception_mode") or {}
    exception_modes = {mode for category in exceptions.values() for mode in (category or {}).values()}
    if exception_modes - {"q01/q99"}:
        raise ValueError(f"Unsupported g05-base normalization exceptions: {exceptions}.")
    shape_meta = processor_metadata.get("shape_meta") or {}
    raw_state_dim = sum(int(item["shape"]) for item in shape_meta.get("state", []))
    raw_action_dim = sum(int(item["shape"]) for item in shape_meta.get("action", []))
    action_feature_names = tuple(
        f"{item['key']}.{index}"
        for item in shape_meta.get("action", [])
        for index in range(int(item["shape"]))
    )
    return G05Config(
        checkpoint_profile="g05-base",
        embodiment=embodiment,
        action_head=action_head,
        runtime_system="system2" if predict_cot else "system1",
        predict_cot=predict_cot,
        discrete_action=discrete,
        continuous_action=continuous,
        return_continuous_action=action_head == "flow",
        policy_action_dim=int(arch.get("action_dim", 27)),
        policy_state_dim=int(arch.get("proprio_dim", 27)),
        raw_state_dim=raw_state_dim,
        raw_action_dim=raw_action_dim,
        chunk_size=horizon,
        normalization_mode=checkpoint_normalization,
        normalization_clip=(-5.0, 5.0),
        use_relative_actions=True,
        relative_exclude_joints=("gripper",),
        action_feature_names=action_feature_names,
        use_stepwise_action_norm=bool(processor_metadata.get("use_stepwise_action_norm", False)),
        camera_order=G05_CAMERA_PROFILES[embodiment],
        n_obs_steps=int(processor_metadata.get("num_obs_steps", 1)),
        num_input_images=num_input_images,
        author_model_config=arch,
        processor_metadata=processor_metadata,
        action_codec_metadata=model.get("tokenizer", hydra.get("tokenizer", {})),
        author_source_revision=G05_SOURCE_REVISION,
        source_checkpoint_revision=G05_HUB_REVISION,
        license="other",
        tags=["g05", "robotics", "non-commercial"],
    )


def _camera_sizes(
    processor_metadata: dict[str, Any], camera_order: tuple[str, ...]
) -> dict[str, tuple[int, int]]:
    images = (processor_metadata.get("shape_meta") or {}).get("images") or []
    camera_size_config = processor_metadata.get("camera_size_config") or {}
    by_lerobot_key = {
        item.get("lerobot_key"): tuple(camera_size_config.get(item.get("camera_type"), item["shape"][-2:]))
        for item in images
        if item.get("lerobot_key") and len(item.get("shape") or ()) >= 3
    }
    if by_lerobot_key:
        missing = [key for key in camera_order if key not in by_lerobot_key]
        if missing:
            raise ValueError(f"Checkpoint processor shape_meta is missing cameras {missing}.")
        return {key: by_lerobot_key[key] for key in camera_order}
    return {}


def convert_dataset_stats(payload: dict[str, Any], config: G05Config) -> dict[str, dict[str, torch.Tensor]]:
    """Flatten author per-part stats into LeRobot raw state/action feature stats."""

    if config.embodiment in payload:
        payload = payload[config.embodiment]
    shape_meta = config.processor_metadata.get("shape_meta") or {}
    if config.normalization_mode == "z_score_tail_mixed":
        exceptions = config.processor_metadata.get("norm_exception_mode") or {}
        default_mode = str(config.processor_metadata.get("norm_default_mode"))
        result: dict[str, dict[str, torch.Tensor]] = {}
        for source_category, feature_name in (("state", OBS_STATE), ("action", ACTION)):
            category_stats = payload.get(source_category)
            component_meta = shape_meta.get(source_category)
            if not isinstance(category_stats, dict) or not isinstance(component_meta, list):
                raise ValueError(
                    f"dataset_stats.json and processor shape_meta must define {source_category!r} components."
                )
            prefix = (
                "stepwise" if source_category == "action" and config.use_stepwise_action_norm else "global"
            )
            collected: dict[str, list[torch.Tensor]] = {
                name: [] for name in ("mean", "std", "tail_q01", "tail_q99", "tail_mean", "tail_mask")
            }
            for component in component_meta:
                key = component["key"]
                stats = category_stats.get(key)
                if not isinstance(stats, dict):
                    raise ValueError(f"Missing checkpoint statistics for {source_category}.{key}.")
                mode = (exceptions.get(source_category) or {}).get(key, default_mode)
                q01 = torch.as_tensor(stats[f"{prefix}_q01"], dtype=torch.float32)
                q99 = torch.as_tensor(stats[f"{prefix}_q99"], dtype=torch.float32)
                mean = torch.as_tensor(stats[f"{prefix}_mean"], dtype=torch.float32)
                std = torch.as_tensor(stats[f"{prefix}_std"], dtype=torch.float32)
                if mode == "q01/q99":
                    collected["mean"].append((q01 + q99) / 2)
                    collected["std"].append((q99 - q01) / 2)
                    tail_mask = torch.zeros(q01.shape[-1], dtype=torch.bool)
                elif mode == "z-score-tail":
                    collected["mean"].append(mean)
                    collected["std"].append(std)
                    tail_mask = torch.ones(q01.shape[-1], dtype=torch.bool)
                else:
                    raise ValueError(
                        f"Unsupported checkpoint normalization mode {source_category}.{key}={mode!r}."
                    )
                collected["tail_q01"].append(q01)
                collected["tail_q99"].append(q99)
                collected["tail_mean"].append(mean)
                collected["tail_mask"].append(tail_mask)
            result[feature_name] = {name: torch.cat(parts, dim=-1) for name, parts in collected.items()}
        return result

    stat_names = (
        ("q01", "q99")
        if config.normalization_mode == "q01_q99"
        else ("mean", "std")
        if config.normalization_mode == "z_score"
        else ()
    )
    result: dict[str, dict[str, torch.Tensor]] = {}
    for source_category, feature_name in (("state", OBS_STATE), ("action", ACTION)):
        if not stat_names:
            continue
        category_stats = payload.get(source_category)
        component_meta = shape_meta.get(source_category)
        if not isinstance(category_stats, dict) or not isinstance(component_meta, list):
            raise ValueError(
                f"dataset_stats.json and processor shape_meta must define {source_category!r} components."
            )
        converted_feature: dict[str, torch.Tensor] = {}
        for short_name in stat_names:
            source_name = (
                f"stepwise_{short_name}"
                if source_category == "action" and config.use_stepwise_action_norm
                else f"global_{short_name}"
            )
            parts = []
            for component in component_meta:
                key = component["key"]
                if key not in category_stats or source_name not in category_stats[key]:
                    raise ValueError(f"Missing checkpoint statistic {source_category}.{key}.{source_name}.")
                parts.append(torch.as_tensor(category_stats[key][source_name], dtype=torch.float32))
            converted_feature[short_name] = torch.cat(parts, dim=-1)
        expected_width = config.raw_state_dim if feature_name == OBS_STATE else config.raw_action_dim
        if converted_feature[stat_names[0]].shape[-1] != expected_width:
            raise ValueError(
                f"Flattened {feature_name} statistics have width "
                f"{converted_feature[stat_names[0]].shape[-1]}, expected {expected_width}."
            )
        result[feature_name] = converted_feature
    return result


def convert_checkpoint(
    source_dir: Path,
    output_dir: Path,
    profile: str,
    *,
    license_file: Path,
    embodiment: str | None = None,
    action_head: str | None = None,
    expected_state: dict[str, torch.Tensor] | None = None,
) -> ConversionReport:
    hydra_path = source_dir / ".hydra" / "config.yaml"
    stats_path = source_dir / "dataset_stats.json"
    candidates = (
        source_dir / "model.pt",
        source_dir / "checkpoints" / "model_state_dict.pt",
    )
    checkpoint_path = next((path for path in candidates if path.is_file()), None)
    tokenizer_path = source_dir / "action_tokenizer.pt"
    processor_path = source_dir / "hf_processor"
    required = [hydra_path, stats_path, tokenizer_path, processor_path, license_file]
    missing_files = [str(path) for path in required if not path.exists()]
    if checkpoint_path is None:
        missing_files.append("model.pt or checkpoints/model_state_dict.pt")
    if missing_files:
        raise FileNotFoundError(f"Incomplete G0.5 checkpoint bundle: {missing_files}")

    hydra = yaml.safe_load(hydra_path.read_text())
    config = _profile_config(profile, hydra, embodiment=embodiment, action_head=action_head)
    camera_sizes = _camera_sizes(config.processor_metadata, config.camera_order)
    if camera_sizes:
        config.camera_sizes = camera_sizes
    stats_payload = json.loads(stats_path.read_text())
    lerobot_stats = convert_dataset_stats(stats_payload, config)
    converted, report = convert_state_dict(_load_checkpoint(checkpoint_path), expected_state)
    report.fail_if_invalid()

    output_dir.mkdir(parents=True, exist_ok=True)
    report.shared_aliases = save_converted_state_dict(converted, output_dir / "model.safetensors")
    config._save_pretrained(output_dir)
    preprocessor, postprocessor = make_g05_pre_post_processors(config, dataset_stats=lerobot_stats)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    shutil.copy2(stats_path, output_dir / "g05_dataset_stats.json")
    shutil.copy2(tokenizer_path, output_dir / "action_tokenizer.pt")
    shutil.copytree(processor_path, output_dir / "hf_processor", dirs_exist_ok=True)
    shutil.copy2(hydra_path, output_dir / "author_config.yaml")
    shutil.copy2(license_file, output_dir / "LICENSE-G0.5")
    (output_dir / "NOTICE").write_text(
        "G0.5 is licensed under the G0.5 Community License Agreement "
        "(Non-Commercial + Limited Patent License), not sold, Copyright © 2026 "
        "Galaxea. All rights reserved by Galaxea. “Galaxea” and related marks are "
        "trademarks of Galaxea or its affiliates.\n"
    )
    (output_dir / "conversion_report.json").write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("g05-base", "g05-libero", "g05-robotwin20"), required=True)
    parser.add_argument(
        "--embodiment",
        choices=("galaxea_r1lite", "galaxea_r1pro"),
        help="Required concrete embodiment for g05-base.",
    )
    parser.add_argument(
        "--action-head",
        choices=("flow", "actioncodec"),
        help="Select one enabled g05-base output head; benchmark checkpoints are flow-only.",
    )
    parser.add_argument("--license-file", type=Path, required=True)
    args = parser.parse_args()
    report = convert_checkpoint(
        args.source_dir,
        args.output_dir,
        args.profile,
        license_file=args.license_file,
        embodiment=args.embodiment,
        action_head=args.action_head,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
