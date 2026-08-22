# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Convert a G0.5 checkpoint into a self-contained LeRobot artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_tokenizer import G05ActionCodecConfig, G05ActionCodecModel
from .configuration_g05 import G05Config
from .modeling_g05 import G05Policy
from .processor_g05 import make_g05_pre_post_processors

_COT_PROMPTS_BY_BUILDER = {
    "SubtaskCoTBuilder": "predict subtask",
    "TaskAsSubtaskCoTBuilder": "predict subtask",
    "FutureSubtaskCoTBuilder": "predict future subtask",
    "BBoxCoTBuilder": "predict bbox",
    "BBoxSubtaskCoTBuilder": "predict bbox, subtask and action",
    "Trace2DCoTBuilder": "predict 2d trace of gripper",
    "SubtaskActionHintCoTBuilder": "predict subtask and action hint",
}

_SO100_LEGACY_JOINT_SIGNS = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
_SO100_LEGACY_JOINT_OFFSETS = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
_ACTION_CODEC_CONTRASTIVE_KEYS = {
    "action_time_contrastive_loss.logit_bias",
    "action_time_contrastive_loss.logit_scale",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("checkpoint conversion requires PyYAML") from error
    with path.open() as stream:
        return yaml.safe_load(stream)


def _checkpoint_path(input_dir: Path) -> Path:
    candidates = [input_dir / "model.pt", input_dir / "checkpoints" / "model_state_dict.pt"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one G0.5 model checkpoint under {input_dir}")
    return matches[0]


def _model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict) or not all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        raise ValueError("legacy checkpoint does not contain a tensor-only model_state_dict")
    return checkpoint


def _group_offsets(parts_meta: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for name, width in parts_meta.items():
        offsets[name] = offset
        offset += int(width)
    return offsets


def _layout_indices(
    shape_items: list[dict[str, Any]], merge_spec: dict[str, list[str]], parts_meta: dict[str, int]
) -> list[int]:
    offsets = _group_offsets(parts_meta)
    indices: list[int] = []
    for item in shape_items:
        source_name = item["key"]
        group = next((name for name, members in merge_spec.items() if source_name in members), None)
        if group is None:
            raise ValueError(f"shape key {source_name!r} is absent from the merger spec")
        width = int(item["shape"] if isinstance(item["shape"], int) else item["shape"][-1])
        if width > int(parts_meta[group]):
            raise ValueError(f"shape key {source_name!r} does not fit group {group!r}")
        indices.extend(range(offsets[group], offsets[group] + width))
    return indices


def _select_embodiment(hydra: dict[str, Any], stats: dict[str, Any], requested: str | None) -> str:
    embodiment = requested or hydra.get("eval_embodiment") or hydra.get("data", {}).get("embodiment")
    if embodiment is None and len(stats) == 1:
        embodiment = next(iter(stats))
    if embodiment is None:
        raise ValueError("this checkpoint contains multiple embodiments; pass --embodiment explicitly")
    if embodiment not in stats:
        raise ValueError(f"embodiment {embodiment!r} has no published statistics; available: {sorted(stats)}")
    return embodiment


def _canonical_shape_meta(shape_meta: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize supported shape metadata schemas."""

    canonical: dict[str, list[dict[str, Any]]] = {"action": [], "state": [], "images": []}
    for target_section, source_section in (("action", "action"), ("state", "state"), ("state", "proprio")):
        if canonical[target_section] or source_section not in shape_meta:
            continue
        source_offsets: dict[str, int] = {}
        for item in shape_meta[source_section]:
            if "shape" in item:
                canonical[target_section].append(dict(item))
                continue
            sources = item.get("sources") or []
            if len(sources) != 1:
                raise ValueError(f"G0.5 {source_section} part {item.get('key')!r} must have one source")
            source = sources[0]
            width = source.get("raw_shape")
            if not isinstance(width, int):
                raise ValueError(f"G0.5 {source_section} part {item.get('key')!r} has no scalar width")
            source_key = source["lerobot_key"]
            start_index = int(source.get("start_index", 0))
            expected_start = source_offsets.get(source_key, 0)
            if start_index != expected_start:
                raise ValueError(
                    f"G0.5 {source_section} source {source_key!r} has non-contiguous or "
                    f"out-of-order offset {start_index}; expected {expected_start}"
                )
            source_offsets[source_key] = start_index + width
            canonical[target_section].append({"key": item["key"], "lerobot_key": source_key, "shape": width})

    for item in shape_meta.get("images", []):
        if "lerobot_key" in item or item.get("dummy", False):
            canonical["images"].append(dict(item))
            continue
        sources = item.get("sources") or []
        if len(sources) != 1:
            raise ValueError(f"G0.5 image part {item.get('key')!r} must have one source")
        source_key = sources[0].get("lerobot_key")
        # Training exports can randomize physical cameras into anonymous slots.
        # Deployment uses the semantic part name recorded alongside that slot.
        if not source_key or ".random_slot_" in source_key:
            source_key = f"observation.images.{item['key']}"
        canonical["images"].append({"key": item["key"], "lerobot_key": source_key})

    if not all(canonical.values()):
        raise ValueError("G0.5 shape metadata must define action, state/proprio, and images")
    return canonical


def _processor_contract(processor: dict[str, Any]) -> dict[str, Any]:
    """Flatten the sequential processor format stored by newer training exports."""

    if "steps" not in processor:
        return processor
    contract: dict[str, Any] = {}
    for step in processor["steps"]:
        target = str(step.get("_target_", ""))
        if target.endswith("LinearNormalizer"):
            contract.update(
                norm_default_mode=step.get("default_mode"),
                norm_exception_mode=step.get("exception_mode") or {},
                use_stepwise_action_norm=step.get("use_stepwise_action_norm"),
            )
        elif target.endswith(("PaddingActionMerger", "ActionStateMerger")):
            contract["action_state_merger"] = {"merge_spec": step.get("merge_spec")}
        elif target.endswith("RelativeJointTransform"):
            contract.setdefault("action_state_transforms", []).append(step)
        elif target.endswith("G05ModelBoundaryTransform"):
            contract["image_keys"] = step.get("image_keys")
            contract["samples_builder"] = step.get("samples_builder")
    return contract


def _cot_prompt(processor: dict[str, Any]) -> str:
    """Resolve the inference-only prompt used before the source EOC boundary."""

    builder = processor.get("samples_builder")
    if not isinstance(builder, dict):
        raise ValueError("predict_cot=true but the checkpoint has no samples_builder metadata")
    target = str(builder.get("_target_", ""))
    if target.endswith("MixedSamplesBuilder"):
        builder = builder.get("eval_builder")
        if not isinstance(builder, dict):
            raise ValueError("the checkpoint's mixed samples builder has no eval_builder")
        target = str(builder.get("_target_", ""))
    builder_name = target.rsplit(".", 1)[-1]
    try:
        return _COT_PROMPTS_BY_BUILDER[builder_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported G0.5 inference CoT builder {builder_name!r}; "
            "the converter cannot safely reconstruct its pre-EOC prompt"
        ) from error


def _action_head_flags(architecture: dict[str, Any]) -> tuple[bool, bool]:
    """Select the deployment head from the checkpoint's own action flags.

    Returns ``(continuous_action, discrete_action)``. G05Policy serves inference
    through exactly one head, so a checkpoint that trained both objectives is
    collapsed here using its own ``return_continuous_action`` preference. Flip
    the flags on the CLI to resume the other objective for training.
    """

    continuous = bool(architecture.get("continuous_action", False))
    discrete = bool(architecture.get("discrete_action", False))
    if continuous and discrete:
        return (True, False) if bool(architecture.get("return_continuous_action", True)) else (False, True)
    if continuous:
        return True, False
    if discrete:
        return False, True
    raise ValueError("checkpoint enables neither continuous nor discrete action inference")


def _joint_frame_transform(
    embodiment: str, physical_state_dim: int, physical_action_dim: int
) -> tuple[list[float] | None, list[float] | None]:
    """Return the legacy SO100 training-frame transform used by its deploy client."""

    if embodiment not in {"so100", "so101"}:
        return None, None
    expected_width = len(_SO100_LEGACY_JOINT_SIGNS)
    if physical_state_dim != expected_width or physical_action_dim != expected_width:
        raise ValueError("the legacy SO100 joint-frame transform requires six state/action dimensions")
    return list(_SO100_LEGACY_JOINT_SIGNS), list(_SO100_LEGACY_JOINT_OFFSETS)


def _embodiment_metadata(hydra: dict[str, Any], embodiment: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = hydra["data"]
    if isinstance(data, dict):
        # Newer stripped training exports store one resolved embodiment directly.
        if data.get("shape_meta") is not None:
            if data.get("embodiment") not in {None, embodiment}:
                raise ValueError("the requested embodiment does not match the exported training bundle")
            return _canonical_shape_meta(data["shape_meta"]), _processor_contract(data.get("processor", {}))

        processor = data.get("processors", {}).get(embodiment, {})
        dataset = data.get("embodiment_datasets", {}).get(embodiment, {})
        shape_meta = processor.get("shape_meta") or dataset.get("shape_meta")
        if shape_meta is not None:
            return _canonical_shape_meta(shape_meta), _processor_contract(processor)
    raise ValueError(f"the published metadata has no shape schema for {embodiment!r}")


def _merge_spec(processor: dict[str, Any], parts_meta: dict[str, int]) -> dict[str, list[str]]:
    configured = processor.get("action_state_merger", {}).get("merge_spec")
    if isinstance(configured, dict):
        return configured
    # Some post-training sidecars retained an oc.load expression. The action
    # tokenizer's canonical group names make the intended mapping unambiguous.
    spec = {name: [name] for name in parts_meta}
    for side in ("left", "right"):
        control = f"{side}_control"
        if control in spec:
            spec[control].extend([f"{side}_arm", f"{side}_ee_pose"])
    return spec


def _relative_action_mask(
    processor: dict[str, Any], action_items: list[dict[str, Any]], state_items: list[dict[str, Any]]
) -> list[bool]:
    relative_keys: set[str] = set()
    for transform in processor.get("action_state_transforms", []):
        if str(transform.get("_target_", "")).endswith("RelativeJointTransform"):
            relative_keys.update(transform["keys"])
    if not relative_keys:
        return []
    action_layout = [(item["key"], int(item["shape"])) for item in action_items]
    state_layout = [(item["key"], int(item["shape"])) for item in state_items]
    if action_layout != state_layout:
        raise ValueError("relative actions require matching physical action and state layouts")
    unknown = relative_keys - {name for name, _ in action_layout}
    if unknown:
        raise ValueError(f"relative-action keys are absent from the shape schema: {sorted(unknown)}")
    return [name in relative_keys for name, width in action_layout for _ in range(width)]


def _normalization_specs(
    stats: dict[str, Any],
    shape_items: list[dict[str, Any]],
    *,
    stepwise: bool,
    default_mode: str,
    exception_modes: dict[str, str],
    section: str,
) -> list[dict[str, Any]]:
    prefix = "stepwise_" if stepwise else "global_"
    specs: list[dict[str, Any]] = []
    for item in shape_items:
        mode = exception_modes.get(item["key"], default_mode).removesuffix("-tail")
        if mode not in {"z-score", "q01/q99"}:
            raise ValueError(f"normalization mode {mode!r} for {item['key']!r} is not supported")

        width = int(item["shape"] if isinstance(item["shape"], int) else item["shape"][-1])
        item_stats = stats[item["key"]]
        names = ("mean", "std") if mode == "z-score" else ("q01", "q99")
        try:
            selected = {name: item_stats[prefix + name] for name in names}
        except KeyError as error:
            raise ValueError(
                f"published {section} statistics for {item['key']!r} do not support mode {mode!r}"
            ) from error
        specs.append({"mode": mode, "width": width, "stats": selected})
    return specs


def _normalization_config(
    model_processor: dict[str, Any], embodiment_processor: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the normalizer contract recorded by the original Hydra config.

    Post-training checkpoints can override the dataset processor at the model
    boundary. Such an override is atomic: exception
    modes must not leak in from the dataset's different normalizer contract.
    """
    model_processor = _processor_contract(model_processor)
    embodiment_processor = _processor_contract(embodiment_processor)
    source = model_processor if model_processor.get("norm_default_mode") is not None else embodiment_processor
    default_mode = source.get("norm_default_mode")
    if default_mode is None:
        raise ValueError(
            "the original checkpoint config does not define norm_default_mode for this embodiment"
        )

    stepwise = model_processor.get("use_stepwise_action_norm")
    if stepwise is None:
        stepwise = source.get("use_stepwise_action_norm")
    if stepwise is None:
        raise ValueError("the original checkpoint config does not define use_stepwise_action_norm")

    return {
        "default_mode": default_mode,
        "exception_modes": source.get("norm_exception_mode") or {},
        "use_stepwise_action_norm": bool(stepwise),
    }


def _camera_layout(
    image_items: list[dict[str, Any]], output_camera_count: int
) -> tuple[list[str], list[str], list[str]]:
    """Translate published camera slots into explicit LeRobot feature names."""
    so101_camera_keys = {
        "__so100_exterior__": "observation.images.exterior_rgb",
        "__so100_wrist_left__": "observation.images.left_wrist_rgb",
        "__so100_wrist_right__": "observation.images.right_wrist_rgb",
    }
    camera_keys: list[str] = []
    dummy_camera_keys: list[str] = []
    camera_order: list[str] = []
    for index, item in enumerate(image_items):
        published_key = item.get("lerobot_key")
        key = so101_camera_keys.get(published_key, published_key)
        is_dummy = bool(item.get("dummy", False))
        if key is None:
            key = f"observation.images.g05_dummy_{index}"
            is_dummy = True
        (dummy_camera_keys if is_dummy else camera_keys).append(key)
        camera_order.append(key)

    while len(camera_order) < output_camera_count:
        key = f"observation.images.g05_dummy_{len(camera_order)}"
        dummy_camera_keys.append(key)
        camera_order.append(key)
    if len(camera_order) != output_camera_count:
        raise ValueError("shape metadata contains more cameras than the model processor accepts")
    return camera_keys, dummy_camera_keys, camera_order


def _exported_action_tokens(input_dir: Path) -> list[str] | None:
    """Read the authoritative token order from a self-contained training export."""

    path = input_dir / "input_processor" / "input_processor_config.json"
    if not path.is_file():
        return None
    with path.open() as stream:
        metadata = json.load(stream)
    added = metadata.get("added_tokens")
    if not isinstance(added, dict) or not added:
        raise ValueError("exported input processor has no added-token mapping")
    ordered = sorted(added.items(), key=lambda item: item[1])
    ids = [int(index) for _, index in ordered]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("exported input processor token IDs are not contiguous")
    tokens = [token for token, _ in ordered]
    if tokens[-2:] != ["<EOV>", "<state>"]:
        raise ValueError("exported G0.5 vocabulary must end with EOV and state tokens")
    return tokens[:-2]


def _action_tokenizer_config(hydra: dict[str, Any], source: Path) -> dict[str, Any]:
    frontend = hydra["model"]["model_arch"]["AT_CONFIG"]
    if not isinstance(frontend, dict):
        frontend = hydra["tokenizer"]["vq_config"]
    if isinstance(frontend.get("model_arch"), dict):
        return frontend
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("the ActionCodec architecture is absent from Hydra; pass its HF export directory")
    with (source / "config.json").open() as stream:
        architecture = json.load(stream)
    return {**frontend, "model_arch": architecture}


def _action_tokens(model_config: dict[str, Any]) -> list[str]:
    tokenizer_config = model_config["AT_CONFIG"]
    architecture = tokenizer_config["model_arch"]
    codebook_size = int(architecture["codebook_size"])
    tokens = [f"<action{index:04d}>" for index in range(codebook_size)]
    if tokenizer_config.get("use_group_markers", False):
        parts = tokenizer_config["parts_meta"]
        rule_patterns = tuple(tokenizer_config.get("rule_based_key_patterns", []))
        rule = [name for name in parts if any(pattern in name for pattern in rule_patterns)]
        learned = [name for name in parts if name not in rule]
        residuals = int(architecture["n_codebooks"])
        tokens += [f"<{name}_{level}>" for level in range(residuals) for name in learned]
        tokens += [f"<{name}>" for name in rule]
    return tokens


def _save_action_tokenizer(
    checkpoint_path: Path,
    tokenizer_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Convert or validate an exported ActionCodec as a standalone HF model."""
    frontend_fields = {
        name: tokenizer_config[name]
        for name in (
            "parts_meta",
            "rule_based_key_patterns",
            "rule_based_min_block_len",
            "rule_based_binarize_threshold",
            "num_residuals",
            "use_group_markers",
            "absent_key_fill_value",
        )
        if name in tokenizer_config
    }
    # The Hydra mapping order is the checkpoint's flat action layout. Serialise it
    # explicitly: the saved JSON sorts object keys, so relying on parts_meta's own
    # order silently permutes the decoded action dimensions for any embodiment
    # whose canonical part order is not alphabetical.
    if "parts_meta" in frontend_fields:
        frontend_fields["parts_order"] = list(frontend_fields["parts_meta"])
    config = G05ActionCodecConfig(**tokenizer_config["model_arch"], **frontend_fields)

    if checkpoint_path.is_dir():
        weights_path = checkpoint_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing ActionCodec safetensors: {weights_path}")
        with torch.device("meta"):
            expected = G05ActionCodecModel(config).state_dict()
        # Ignore optional head tensors only when the configured model does not
        # instantiate that head.
        ignored_keys = _ACTION_CODEC_CONTRASTIVE_KEYS - set(expected)
        with safe_open(weights_path, framework="pt") as stream:
            serialized_keys = set(stream.keys())
            source_keys = serialized_keys - ignored_keys
            bad_shapes = {
                key: (tuple(stream.get_slice(key).get_shape()), tuple(expected[key].shape))
                for key in source_keys & set(expected)
                if tuple(stream.get_slice(key).get_shape()) != tuple(expected[key].shape)
            }
            tensors = (
                {key: stream.get_tensor(key) for key in source_keys}
                if serialized_keys != source_keys
                else None
            )
        missing = set(expected) - source_keys
        unexpected = source_keys - set(expected)
        if missing or unexpected or bad_shapes:
            raise ValueError(
                f"ActionCodec weight mapping failed: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, bad_shapes={bad_shapes}"
            )
        output_dir.mkdir(parents=True)
        config.save_pretrained(output_dir)
        if tensors is None:
            shutil.copy2(weights_path, output_dir / weights_path.name)
        else:
            save_torch_state_dict(tensors, output_dir, max_shard_size="5GB")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    source = checkpoint.get("model_state_dict", checkpoint)
    source = {key.removeprefix("model."): value.contiguous() for key, value in source.items()}
    with torch.device("meta"):
        expected = G05ActionCodecModel(config).state_dict()
    ignored_keys = _ACTION_CODEC_CONTRASTIVE_KEYS - set(expected)
    source = {key: value for key, value in source.items() if key not in ignored_keys}
    missing = set(expected) - set(source)
    unexpected = set(source) - set(expected)
    bad_shapes = {
        key: (tuple(source[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(source)
        if source[key].shape != expected[key].shape
    }
    if missing or unexpected or bad_shapes:
        raise ValueError(
            f"ActionCodec weight mapping failed: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, bad_shapes={bad_shapes}"
        )
    output_dir.mkdir(parents=True)
    config.save_pretrained(output_dir)
    save_torch_state_dict(source, output_dir, max_shard_size="5GB")


def _build_config(
    hydra: dict[str, Any],
    stats: dict[str, Any],
    state: dict[str, Tensor],
    embodiment: str | None = None,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    action_tokens: list[str] | None = None,
) -> tuple[G05Config, list[str]]:
    model = hydra["model"]
    architecture = model["model_arch"]
    processor = model["processor"]
    embodiment = _select_embodiment(hydra, stats, embodiment)
    shape_meta, embodiment_processor = _embodiment_metadata(hydra, embodiment)
    action_items, state_items, image_items = (
        shape_meta["action"],
        shape_meta["state"],
        shape_meta["images"],
    )
    if tokenizer_config is None:
        tokenizer_config = architecture["AT_CONFIG"]
        if not isinstance(tokenizer_config, dict):
            tokenizer_config = hydra["tokenizer"]["vq_config"]
    parts_meta = tokenizer_config["parts_meta"]
    merge_spec = _merge_spec(processor, parts_meta)
    action_indices = _layout_indices(action_items, merge_spec, parts_meta)
    state_indices = _layout_indices(state_items, merge_spec, parts_meta)
    physical_action_dim = len(action_indices)
    physical_state_dim = len(state_indices)
    joint_signs, joint_offsets = _joint_frame_transform(embodiment, physical_state_dim, physical_action_dim)
    embodiment_stats = dict(stats[embodiment])
    if "state" not in embodiment_stats and "proprio" in embodiment_stats:
        embodiment_stats["state"] = embodiment_stats["proprio"]
    normalization = _normalization_config(processor, embodiment_processor)
    action_normalization: list[dict[str, Any]] = []
    state_normalization: list[dict[str, Any]] = []
    if normalization["use_stepwise_action_norm"]:
        action_normalization = _normalization_specs(
            embodiment_stats["action"],
            action_items,
            stepwise=True,
            default_mode=normalization["default_mode"],
            exception_modes=normalization["exception_modes"].get("action", {}),
            section="action",
        )
        state_normalization = _normalization_specs(
            embodiment_stats["state"],
            state_items,
            stepwise=False,
            default_mode=normalization["default_mode"],
            exception_modes=normalization["exception_modes"].get("state", {}),
            section="state",
        )

    image_size = tuple(next(iter(processor["camera_size_config"].values())))
    output_camera_count = int(processor["num_output_cameras"])
    semantic_camera_order = {
        "exterior": "exterior_rgb",
        "wrist_left": "left_wrist_rgb",
        "wrist_right": "right_wrist_rgb",
    }
    image_by_key = {item["key"]: item for item in image_items}
    boundary_image_keys = embodiment_processor.get("image_keys")
    if isinstance(boundary_image_keys, list):
        missing_boundary_keys = set(boundary_image_keys) - set(image_by_key)
        if missing_boundary_keys:
            raise ValueError(
                f"model-boundary cameras are absent from the shape schema: {sorted(missing_boundary_keys)}"
            )
        image_items = [image_by_key[key] for key in boundary_image_keys]
    else:
        ordered_keys = [semantic_camera_order.get(name, name) for name in processor["camera_size_config"]]
        if set(ordered_keys) == set(image_by_key):
            image_items = [image_by_key[key] for key in ordered_keys]
    camera_keys, dummy_camera_keys, camera_order = _camera_layout(image_items, output_camera_count)
    chunk_size = architecture.get("horizon_steps")
    n_obs_steps = architecture.get("num_obs_steps")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("checkpoint metadata must define a positive integer horizon_steps")
    if not isinstance(n_obs_steps, int) or n_obs_steps <= 0:
        raise ValueError("checkpoint metadata must define a positive integer num_obs_steps")
    image_tokens = (
        (image_size[0] // int(architecture["vision"]["patch_size"]))
        * (image_size[1] // int(architecture["vision"]["patch_size"]))
        // int(architecture["vision"]["spatial_merge_size"]) ** 2
    )
    # max_chunk_token_length is a training truncation budget, not an inference
    # prompt limit. The base checkpoint's 18 image slots alone exceed 1024.
    max_prompt_length = max(
        int(architecture["max_chunk_token_length"]),
        output_camera_count * n_obs_steps * (image_tokens + 2) + 512,
    )
    vocab_size = int(state["model.vlm.input_proj.weight"].shape[0])
    action_tokens = action_tokens or _action_tokens({**architecture, "AT_CONFIG": tokenizer_config})
    base_tokenizer_size = vocab_size - len(action_tokens) - 2
    if base_tokenizer_size <= 0:
        raise ValueError("invalid checkpoint vocabulary layout")
    eov_token_id = base_tokenizer_size + len(action_tokens)
    state_token_id = eov_token_id + 1
    if state_token_id + 1 != vocab_size:
        raise ValueError("checkpoint vocabulary is not action tokens + EOV + state")

    vlm, expert, vision, fm = (
        architecture["vlm"],
        architecture["action_expert"],
        architecture["vision"],
        architecture["fm"],
    )
    predict_cot = bool(architecture["predict_cot"])
    continuous_action, discrete_action = _action_head_flags(architecture)
    optimizer_lr = float(model.get("learning_rate") or 1e-5)
    supported_fm_contract = {
        "time_convention": "pi_convention",
        "padding_action_weight": 0.0,
        "zero_pad_action_target": False,
        "action_causal": False,
        "final_action_clip_value": None,
    }
    for name, expected_value in supported_fm_contract.items():
        if fm.get(name) != expected_value:
            raise ValueError(
                f"unsupported G0.5 flow setting {name}={fm.get(name)!r}; expected {expected_value!r}"
            )
    if architecture.get("ae_vlm_condition_mode") != "cross_attn_only":
        raise ValueError("only cross_attn_only action conditioning is supported")
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(physical_state_dim,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_size)) for key in camera_keys},
    }
    config = G05Config(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(physical_action_dim,))},
        chunk_size=chunk_size,
        n_obs_steps=n_obs_steps,
        image_size=image_size,
        camera_keys=camera_keys,
        dummy_camera_keys=dummy_camera_keys,
        optional_camera_keys=(
            [key for key in camera_keys if "wrist" in key] if embodiment in {"so100", "so101"} else []
        ),
        camera_order=camera_order,
        internal_action_dim=int(architecture["action_dim"]),
        internal_state_dim=int(architecture["proprio_dim"]),
        action_indices=action_indices,
        state_indices=state_indices,
        action_normalization=action_normalization,
        state_normalization=state_normalization,
        normalization_strategy="g05_stepwise" if normalization["use_stepwise_action_norm"] else "lerobot",
        relative_action_mask=_relative_action_mask(embodiment_processor, action_items, state_items),
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
        embodiment=embodiment,
        vocab_size=vocab_size,
        pad_token_id=int(architecture["pad_token_id"]),
        eos_token_id=int(architecture["eos_token_id"]),
        image_token_id=int(architecture["image_token_index"]),
        state_token_id=state_token_id,
        eov_token_id=eov_token_id,
        max_prompt_length=max_prompt_length,
        text_hidden_size=int(vlm["hidden_size"]),
        text_intermediate_size=int(vlm["intermediate_size"]),
        text_num_layers=int(vlm["num_hidden_layers"]),
        text_num_heads=int(vlm["num_attention_heads"]),
        text_num_kv_heads=int(vlm["num_key_value_heads"]),
        text_head_dim=int(vlm["head_dim"]),
        text_layer_types=list(vlm["layer_types"]),
        rope_theta=float(vlm["rope_parameters"]["rope_theta"]),
        mrope_section=tuple(vlm["rope_parameters"]["mrope_section"]),
        vision_depth=int(vision["depth"]),
        vision_hidden_size=int(vision["hidden_size"]),
        vision_intermediate_size=int(vision["intermediate_size"]),
        vision_num_heads=int(vision["num_heads"]),
        vision_patch_size=int(vision["patch_size"]),
        vision_temporal_patch_size=int(vision["temporal_patch_size"]),
        vision_spatial_merge_size=int(vision["spatial_merge_size"]),
        expert_hidden_size=int(expert["hidden_size"]),
        expert_intermediate_size=int(expert["intermediate_size"]),
        expert_num_layers=int(expert["num_hidden_layers"]),
        expert_num_heads=int(expert["num_attention_heads"]),
        expert_num_kv_heads=int(expert["num_key_value_heads"]),
        expert_head_dim=int(expert["head_dim"]),
        num_inference_steps=int(fm["num_inference_steps"]),
        flow_sig_min=float(fm["flow_sig_min"]),
        flow_sampling=str(fm["flow_sampling"]),
        num_flow_samples=int(fm.get("num_flow_samples", 1)),
        flow_joint_training=bool(fm["joint_training"]),
        fm_loss_weight=float(fm["fm_weight"]),
        action_token_loss_weight=float(architecture["ar"]["ce_weight"]),
        action_token_start_id=base_tokenizer_size,
        action_token_end_id=base_tokenizer_size + len(action_tokens),
        max_cot_tokens=int(architecture["ar"].get("max_new_tokens", 300)),
        max_action_tokens=int(architecture["ar"].get("max_new_tokens", 300)),
        ar_do_sample=bool(architecture["ar"].get("do_sample", False)),
        ar_temperature=float(architecture["ar"].get("temperature", 0.7)),
        ar_top_k=int(architecture["ar"].get("top_k", 128)),
        ar_top_p=float(architecture["ar"].get("top_p", 0.95)),
        ar_repetition_penalty=float(architecture["ar"].get("repetition_penalty", 1.0)),
        ar_no_repeat_ngram_size=int(architecture["ar"].get("no_repeat_ngram_size", 0)),
        cot_prompt=_cot_prompt(embodiment_processor) if predict_cot else "",
        predict_cot=predict_cot,
        continuous_action=continuous_action,
        discrete_action=discrete_action,
        attn_implementation=str(architecture.get("attn_implementation", "eager")),
        action_feature_names=(
            [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ]
            if embodiment in {"so100", "so101"} and physical_action_dim == 6
            else []
        ),
        optimizer_lr=optimizer_lr,
        optimizer_betas=tuple(model.get("betas") or (0.9, 0.95)),
        optimizer_weight_decay=float(model.get("weight_decay") or 0.0),
        optimizer_grad_clip_norm=float(model.get("max_grad_norm") or 1.0),
        scheduler_warmup_steps=int(model.get("warmup_steps") or 0),
        scheduler_decay_steps=int(model.get("max_steps") or 100_000),
        scheduler_decay_lr=optimizer_lr * float(model.get("lr_min_ratio") or 0.1),
    )
    return config, action_tokens


def _remap_weights(source: dict[str, Tensor]) -> dict[str, Tensor]:
    remapped: dict[str, Tensor] = {}
    for key, value in source.items():
        target = key
        target = target.replace("model.vlm.input_proj.", "model.vlm.embed_tokens.")
        target = target.replace("model.vlm.output_proj.", "model.output_proj.")
        target = target.replace("model.proprio_embedder.mlp.", "model.proprio_embedder.")
        if target in remapped:
            raise ValueError(f"multiple source weights map to {target}")
        remapped[target] = value.contiguous()
    return remapped


def _dataset_stats_path(input_dir: Path) -> Path:
    canonical = input_dir / "dataset_stats.json"
    if canonical.is_file():
        return canonical
    matches = list(input_dir.glob("dataset_stats*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one dataset statistics file under {input_dir}")
    return matches[0]


def _action_tokenizer_source(input_dir: Path) -> Path:
    candidates = [
        input_dir / "action_tokenizer_hf",
        input_dir / "action_tokenizer.pt",
        input_dir.parent / "action_tokenizer_hf",
        input_dir.parent / "action_tokenizer.pt",
    ]
    source = next((path for path in candidates if path.is_dir() or path.is_file()), None)
    if source is None:
        raise FileNotFoundError("missing G0.5 ActionCodec checkpoint or HF export")
    return source


def _processor_source(input_dir: Path) -> Path:
    candidates = [
        input_dir / "hf_processor",
        input_dir / "processor",
        input_dir / "qwen3_5_2b_base_processor",
        input_dir.parent / "qwen3_5_2b_base_processor",
    ]
    source = next((path for path in candidates if (path / "tokenizer_config.json").is_file()), None)
    if source is None:
        raise FileNotFoundError("missing G0.5 Hugging Face processor")
    return source


def convert(args: argparse.Namespace) -> None:
    hydra = _load_yaml(args.input_dir / ".hydra" / "config.yaml")
    with _dataset_stats_path(args.input_dir).open() as stream:
        stats = json.load(stream)
    checkpoint_path = _checkpoint_path(args.input_dir)
    action_tokenizer_source = _action_tokenizer_source(args.input_dir)
    processor_source = _processor_source(args.input_dir)
    tokenizer_config = _action_tokenizer_config(hydra, action_tokenizer_source)
    exported_action_tokens = _exported_action_tokens(args.input_dir)
    source_state = _model_state(checkpoint_path)
    config, action_tokens = _build_config(
        hydra,
        stats,
        source_state,
        args.embodiment,
        tokenizer_config=tokenizer_config,
        action_tokens=exported_action_tokens,
    )

    base_tokenizer = AutoTokenizer.from_pretrained(processor_source, local_files_only=True)
    expected_base_size = config.vocab_size - len(action_tokens) - 2
    if len(base_tokenizer) != expected_base_size:
        raise ValueError(
            f"base tokenizer has {len(base_tokenizer)} tokens; checkpoint expects {expected_base_size}"
        )
    base_tokenizer.add_tokens(action_tokens + ["<EOV>", "<state>"], special_tokens=True)
    if len(base_tokenizer) != config.vocab_size:
        raise ValueError("reconstructed tokenizer vocabulary does not match model embeddings")

    remapped = _remap_weights(source_state)
    with torch.device("meta"):
        expected = G05Policy(config).state_dict()
    missing = set(expected) - set(remapped)
    unexpected = set(remapped) - set(expected)
    bad_shapes = {
        key: (tuple(remapped[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(remapped)
        if remapped[key].shape != expected[key].shape
    }
    # Tied language-model output weights are serialized once by LeRobot.
    missing.discard("model.output_proj.weight")
    if missing or unexpected or bad_shapes:
        raise ValueError(
            f"weight mapping failed: missing={sorted(missing)[:20]}, "
            f"unexpected={sorted(unexpected)[:20]}, bad_shapes={bad_shapes}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _save_action_tokenizer(
        action_tokenizer_source,
        tokenizer_config,
        args.output_dir / config.action_tokenizer_subdir,
    )
    processor_output = args.output_dir / config.tokenizer_subdir
    base_tokenizer.save_pretrained(processor_output)
    for filename in ("config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        source = processor_source / filename
        if source.is_file():
            shutil.copy2(source, processor_output / filename)
    config.save_pretrained(args.output_dir)
    preprocessor, postprocessor = make_g05_pre_post_processors(config, tokenizer_path=processor_output)
    preprocessor.save_pretrained(args.output_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(args.output_dir, config_filename="policy_postprocessor.json")
    save_torch_state_dict(
        remapped,
        args.output_dir,
        max_shard_size="20GB",
        shared_tensors_to_discard=["model.output_proj.weight"],
    )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    provenance = {
        "format": "lerobot-g05-v1",
        "embodiment": config.embodiment,
        "source_sha256": digest.hexdigest(),
        "source_filename": checkpoint_path.name,
        "action_tokenizer_filename": action_tokenizer_source.name,
        "self_contained": True,
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment")
    return parser.parse_args(argv)


def main() -> None:
    convert(_parse_args())


if __name__ == "__main__":
    main()
