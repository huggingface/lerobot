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
"""Policy export functionality for converting PyTorch policies to portable formats."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from .manifest import (
    ActionSpec,
    ExportMetadata,
    IOSpec,
    IterativeConfig,
    Manifest,
    NormalizationConfig,
    NormalizationType,
    PolicyInfo,
    PolicySource,
    TensorSpec,
    TwoPhaseConfig,
)
from .normalize import save_stats_safetensors

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def export_policy(
    policy: PreTrainedPolicy,
    output_dir: str | Path,
    *,
    backend: str = "onnx",
    example_batch: dict[str, Tensor] | None = None,
    opset_version: int = 17,
    include_normalization: bool = True,
) -> Path:
    """Export a policy to a PolicyPackage.

    Args:
        policy: Trained policy instance.
        output_dir: Directory to write the package.
        backend: Export backend ("onnx").
        example_batch: Example input for tracing (auto-generated if None).
        opset_version: ONNX opset version (if backend="onnx").
        include_normalization: Include normalization stats in package.

    Returns:
        Path to the created PolicyPackage directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    artifacts_dir = output_dir / "artifacts"
    assets_dir = output_dir / "assets"
    artifacts_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    policy_name = getattr(policy, "name", policy.__class__.__name__.lower())
    inference_type = _detect_inference_type(policy)

    if example_batch is None:
        example_batch = _generate_example_batch(policy)

    two_phase_config = None
    if inference_type == "two_phase":
        if backend == "onnx":
            artifacts, input_specs, output_specs, two_phase_config = _export_two_phase_onnx(
                policy, artifacts_dir, example_batch, opset_version
            )
        else:
            raise ValueError(f"Unsupported backend for two-phase export: {backend}")
    elif backend == "onnx":
        model_path = artifacts_dir / "model.onnx"
        input_specs, output_specs = _export_onnx(
            policy, model_path, example_batch, opset_version, inference_type
        )
        artifacts = {"onnx": "artifacts/model.onnx"}
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Export normalization stats if requested
    normalization_config = None
    if include_normalization:
        stats = _get_policy_stats(policy)
        if stats:
            stats_path = artifacts_dir / "stats.safetensors"
            save_stats_safetensors(stats, stats_path)
            normalization_config = NormalizationConfig(
                type=_get_normalization_type(policy),
                artifact="artifacts/stats.safetensors",
                input_features=_get_normalized_input_features(policy),
                output_features=["action"],
            )

    _save_policy_config(policy, assets_dir / "config.json")

    manifest = _build_manifest(
        policy=policy,
        policy_name=policy_name,
        inference_type=inference_type,
        artifacts=artifacts,
        input_specs=input_specs,
        output_specs=output_specs,
        normalization_config=normalization_config,
        two_phase_config=two_phase_config,
        backend=backend,
    )

    manifest.save(output_dir / "manifest.json")

    return output_dir


def _detect_inference_type(policy: PreTrainedPolicy) -> str:
    """Detect inference type: 'single_pass', 'iterative', or 'two_phase'."""
    policy_class_name = policy.__class__.__name__.lower()

    if "pi0" in policy_class_name or "smolvla" in policy_class_name:
        return "two_phase"

    iterative_patterns = ["diffusion", "flow"]
    for pattern in iterative_patterns:
        if pattern in policy_class_name:
            return "iterative"

    return "single_pass"


def _generate_example_batch(policy: PreTrainedPolicy) -> dict[str, Tensor]:
    """Generate an example batch for ONNX export tracing."""
    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device

    batch: dict[str, Tensor] = {}

    # Add state observation if configured
    if hasattr(config, "robot_state_feature") and config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]
        batch["observation.state"] = torch.randn(batch_size, state_dim, device=device)

    # Add environment state if configured
    if hasattr(config, "env_state_feature") and config.env_state_feature:
        env_dim = config.env_state_feature.shape[0]
        batch["observation.environment_state"] = torch.randn(batch_size, env_dim, device=device)

    # Add image observations if configured
    if hasattr(config, "image_features") and config.image_features:
        for img_key in config.image_features:
            # Get image shape from config
            img_shape = config.image_features[img_key].shape
            batch[img_key] = torch.randn(batch_size, *img_shape, device=device)

    return batch


def _export_onnx(
    policy: PreTrainedPolicy,
    output_path: Path,
    example_batch: dict[str, Tensor],
    opset_version: int,
    inference_type: str,
) -> tuple[list[TensorSpec], list[TensorSpec]]:
    """Export policy to ONNX format.

    Returns input and output tensor specifications.
    """
    from .protocols import is_iterative_exportable, is_single_phase_exportable

    policy.eval()

    if inference_type == "single_pass":
        if is_single_phase_exportable(policy):
            _ = policy.get_single_phase_export_config()
            wrapper = policy.get_forward_module()
            example_inputs, input_names, output_names = policy.prepare_forward_inputs(example_batch)
            export_batch = dict(zip(input_names, example_inputs, strict=True))
        else:
            wrapper, input_names, output_names, export_batch = _create_single_shot_wrapper(
                policy, example_batch
            )
            example_inputs = tuple(export_batch[name] for name in input_names if name in export_batch)
    else:
        if is_iterative_exportable(policy):
            _ = policy.get_iterative_export_config()
            wrapper = policy.get_denoise_module()
            example_inputs, input_names, output_names = policy.prepare_denoise_inputs(example_batch)
            export_batch = dict(zip(input_names, example_inputs, strict=True))
        else:
            wrapper, input_names, output_names, export_batch = _create_iterative_wrapper(
                policy, example_batch
            )
            example_inputs = tuple(export_batch[name] for name in input_names if name in export_batch)

    # Dynamic axes for batch dimension
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Build tensor specs from example batch
    input_specs = []
    for name in input_names:
        if name in example_batch:
            tensor = example_batch[name]
            shape = ["B"] + list(tensor.shape[1:])
            input_specs.append(TensorSpec(name=name, dtype="float32", shape=shape))

    output_specs = []
    # Run forward pass to get output shapes
    with torch.no_grad():
        outputs = wrapper(*example_inputs)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)

    for name, tensor in zip(output_names, outputs, strict=True):
        shape = ["B"] + list(tensor.shape[1:])
        output_specs.append(TensorSpec(name=name, dtype="float32", shape=shape))

    return input_specs, output_specs


def _create_single_shot_wrapper(
    policy: PreTrainedPolicy,
    example_batch: dict[str, Tensor],
) -> tuple[nn.Module, list[str], list[str], dict[str, Tensor]]:
    class SingleShotWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

        def forward(self, *args) -> Tensor:
            # Reconstruct batch dict from positional args
            batch = dict(zip(input_names, args, strict=True))

            # Handle image features for ACT
            if hasattr(self.policy.config, "image_features") and self.policy.config.image_features:
                from lerobot.utils.constants import OBS_IMAGES

                batch[OBS_IMAGES] = [batch[key] for key in self.policy.config.image_features if key in batch]

            # Call the underlying model directly to get action predictions
            if hasattr(self.policy, "model"):
                actions, _ = self.policy.model(batch)
            else:
                actions = self.policy.predict_action_chunk(batch)

            return actions

    input_names = list(example_batch.keys())
    output_names = ["action"]

    return SingleShotWrapper(policy), input_names, output_names, example_batch


def _create_iterative_wrapper(
    policy: PreTrainedPolicy,
    example_batch: dict[str, Tensor],
) -> tuple[nn.Module, list[str], list[str], dict[str, Tensor]]:
    from lerobot.utils.constants import OBS_IMAGES

    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device

    horizon = getattr(config, "horizon", None) or getattr(config, "chunk_size", None)
    if horizon is None:
        raise ValueError(
            f"Policy config must have 'horizon' or 'chunk_size' attribute for iterative export. "
            f"Found attributes: {list(config.__dict__.keys())}"
        )

    if hasattr(config, "action_feature") and config.action_feature is not None:
        action_dim = config.action_feature.shape[0]
    else:
        raise ValueError(
            f"Policy config must have 'action_feature' with shape for iterative export. "
            f"Found: action_feature={getattr(config, 'action_feature', 'MISSING')}"
        )

    extended_batch = dict(example_batch)
    extended_batch["x_t"] = torch.randn(batch_size, horizon, action_dim, device=device)
    extended_batch["timestep"] = torch.tensor([1.0], dtype=torch.float32, device=device)

    is_diffusion = hasattr(policy, "diffusion") and hasattr(policy.diffusion, "unet")
    image_feature_keys = (
        list(config.image_features.keys())
        if hasattr(config, "image_features") and config.image_features
        else []
    )

    class DiffusionIterativeWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy, image_keys: list[str]):
            super().__init__()
            self.diffusion = policy.diffusion
            self.image_keys = image_keys
            if hasattr(self.diffusion, "rgb_encoder"):
                encoder = self.diffusion.rgb_encoder
                if isinstance(encoder, nn.ModuleList):
                    for enc in encoder:
                        enc.do_crop = False
                else:
                    encoder.do_crop = False

        def forward(self, *args) -> Tensor:
            batch = dict(zip(input_names, args, strict=True))

            x_t = batch.pop("x_t")
            timestep = batch.pop("timestep")

            if self.image_keys:
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.image_keys], dim=-4)

            global_cond = self.diffusion._prepare_global_conditioning(batch)
            timestep_long = timestep.long()
            return self.diffusion.unet(x_t, timestep_long, global_cond=global_cond)

    class GenericIterativeWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

        def forward(self, *args) -> Tensor:
            batch = dict(zip(input_names, args, strict=True))

            x_t = batch.pop("x_t")
            timestep = batch.pop("timestep")

            if hasattr(self.policy, "denoise_step"):
                return self.policy.denoise_step(batch, x_t, timestep)
            elif hasattr(self.policy, "model") and hasattr(self.policy.model, "denoise_step"):
                return self.policy.model.denoise_step(batch, x_t, timestep)
            else:
                raise NotImplementedError(
                    f"Policy {type(self.policy).__name__} does not have a denoise_step method"
                )

    input_names = list(extended_batch.keys())
    output_names = ["v_t"]

    if is_diffusion:
        return (
            DiffusionIterativeWrapper(policy, image_feature_keys),
            input_names,
            output_names,
            extended_batch,
        )
    else:
        return GenericIterativeWrapper(policy), input_names, output_names, extended_batch


def _fix_onnx_gather_indices(onnx_path: Path) -> None:
    """Fix ONNX Gather nodes that have float indices from ScatterND outputs.

    The transformers SmolVLM vision encoder produces position_ids through ScatterND which
    inherits float dtype from ConstantOfShape. But Gather requires int indices.
    This inserts Cast nodes to convert float indices to int64.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))

    nodes_to_insert: list[tuple[int, onnx.NodeProto]] = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Gather" and "position_embedding" in node.name:
            indices_input = node.input[1]
            if "ScatterND" in indices_input:
                cast_output = indices_input + "_cast_to_int64"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[indices_input],
                    outputs=[cast_output],
                    name=indices_input + "/Cast_to_int64",
                    to=TensorProto.INT64,
                )
                nodes_to_insert.append((idx, cast_node))
                node.input[1] = cast_output

    for idx, cast_node in reversed(nodes_to_insert):
        model.graph.node.insert(idx, cast_node)

    if nodes_to_insert:
        onnx.save(model, str(onnx_path))


def _fix_onnx_double_to_float(onnx_path: Path) -> None:
    """Fix ONNX nodes that use double precision where float32 is sufficient.

    onnxruntime CPUExecutionProvider doesn't implement Cos/Sin for double.
    This converts double constants and casts to float32.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(onnx_path))
    modified = False

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.DOUBLE:
                    data = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(data))
                    modified = True

        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    modified = True

    if modified:
        onnx.save(model, str(onnx_path))


def _export_two_phase_onnx(
    policy: PreTrainedPolicy,
    artifacts_dir: Path,
    example_batch: dict[str, Tensor],
    opset_version: int,
) -> tuple[dict[str, str], list[TensorSpec], list[TensorSpec], TwoPhaseConfig]:
    """Export a two-phase VLA policy (PI0, SmolVLA) to ONNX.

    Two-phase export creates two ONNX models:
    1. encoder.onnx: Encodes images/language/state → KV cache
    2. denoise.onnx: Single denoise step using cached KV values
    """

    from .protocols import is_two_phase_exportable

    policy.eval()
    device = next(policy.parameters()).device

    is_pi0 = "pi0" in policy.__class__.__name__.lower()

    if not is_pi0:
        policy = policy.float()

    if not is_two_phase_exportable(policy):
        raise ValueError(
            f"Two-phase VLA policy {policy.__class__.__name__} must implement the ExportableTwoPhase protocol. "
            f"Add the required methods: get_two_phase_export_config(), get_encoder_module(), "
            f"get_denoise_module(), prepare_encoder_inputs(), prepare_denoise_inputs(). "
            f"See PI0Policy or SmolVLAPolicy for reference implementations."
        )

    export_config = policy.get_two_phase_export_config()
    num_layers = export_config.num_layers
    num_kv_heads = export_config.num_kv_heads
    head_dim = export_config.head_dim
    num_steps = export_config.num_steps
    chunk_size = export_config.chunk_size
    action_dim = export_config.action_dim

    encoder_inputs, encoder_input_names, num_images, input_mapping = policy.prepare_encoder_inputs(
        example_batch
    )
    encoder_wrapper = policy.get_encoder_module(num_images=num_images)

    with torch.no_grad():
        encoder_outputs = encoder_wrapper(*encoder_inputs)
        prefix_len = encoder_outputs[0].shape[1]

    denoise_inputs, denoise_input_names = policy.prepare_denoise_inputs(prefix_len, device)
    denoise_wrapper = policy.get_denoise_module()

    # Build encoder output names
    encoder_output_names = ["prefix_pad_mask"]
    for layer_idx in range(num_layers):
        encoder_output_names.append(f"past_key_{layer_idx}")
        encoder_output_names.append(f"past_value_{layer_idx}")

    # Dynamic axes for encoder
    encoder_dynamic_axes = {}
    for name in encoder_input_names:
        encoder_dynamic_axes[name] = {0: "batch_size"}
    for name in encoder_output_names:
        encoder_dynamic_axes[name] = {0: "batch_size"}

    # Export encoder
    encoder_path = artifacts_dir / "encoder.onnx"
    torch.onnx.export(
        encoder_wrapper,
        encoder_inputs,
        str(encoder_path),
        input_names=encoder_input_names,
        output_names=encoder_output_names,
        dynamic_axes=encoder_dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    _fix_onnx_gather_indices(encoder_path)

    denoise_output_names = ["v_t"]

    # Dynamic axes for denoise
    denoise_dynamic_axes = {}
    for name in denoise_input_names:
        denoise_dynamic_axes[name] = {0: "batch_size"}
    for name in denoise_output_names:
        denoise_dynamic_axes[name] = {0: "batch_size"}

    # Export denoise step
    denoise_path = artifacts_dir / "denoise.onnx"
    torch.onnx.export(
        denoise_wrapper,
        denoise_inputs,
        str(denoise_path),
        input_names=denoise_input_names,
        output_names=denoise_output_names,
        dynamic_axes=denoise_dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    _fix_onnx_double_to_float(denoise_path)

    artifacts = {
        "onnx_encoder": "artifacts/encoder.onnx",
        "onnx_denoise": "artifacts/denoise.onnx",
    }

    # Build input specs from encoder inputs
    input_specs = []
    for name, tensor in zip(encoder_input_names, encoder_inputs, strict=True):
        shape = ["B"] + list(tensor.shape[1:])
        if tensor.dtype == torch.long:
            dtype = "int64"
        elif tensor.dtype == torch.bool:
            dtype = "bool"
        else:
            dtype = "float32"
        input_specs.append(TensorSpec(name=name, dtype=dtype, shape=shape))

    output_specs = [TensorSpec(name="action", dtype="float32", shape=["B", chunk_size, action_dim])]

    two_phase_config = TwoPhaseConfig(
        num_steps=num_steps,
        encoder_artifact="onnx_encoder",
        denoise_artifact="onnx_denoise",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        input_mapping=input_mapping,
    )

    return artifacts, input_specs, output_specs, two_phase_config


def _get_policy_stats(policy: PreTrainedPolicy) -> dict[str, dict[str, Any]] | None:
    """Extract normalization statistics from policy."""
    # Try to get stats from policy processor
    if hasattr(policy, "policy_processor"):
        processor = policy.policy_processor
        for step in getattr(processor, "steps", []):
            if hasattr(step, "stats") and step.stats:
                return step.stats

    # Try to get stats from config
    if hasattr(policy, "config") and hasattr(policy.config, "stats"):
        return policy.config.stats

    return None


def _get_normalization_type(policy: PreTrainedPolicy) -> NormalizationType:
    """Detect normalization type from policy."""
    # Default to standard normalization
    return NormalizationType.STANDARD


def _get_normalized_input_features(policy: PreTrainedPolicy) -> list[str]:
    """Get list of input features that should be normalized."""
    features = []
    if hasattr(policy.config, "robot_state_feature") and policy.config.robot_state_feature:
        features.append("observation.state")
    return features


def _save_policy_config(policy: PreTrainedPolicy, path: Path) -> None:
    """Save policy configuration as JSON reference."""
    try:
        config_dict = policy.config.__dict__.copy()
        # Remove non-serializable items
        config_dict = {k: v for k, v in config_dict.items() if _is_json_serializable(v)}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    except Exception as e:
        # Config export is optional but log for debugging
        logger.debug("Could not save policy config to %s: %s", path, e)


def _is_json_serializable(value: Any) -> bool:
    """Check if a value is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _build_iterative_config(policy: PreTrainedPolicy) -> IterativeConfig:
    config = policy.config
    num_steps = getattr(config, "num_inference_steps", 10)

    is_diffusion = hasattr(policy, "diffusion") and hasattr(policy.diffusion, "noise_scheduler")

    if is_diffusion:
        scheduler_type = config.noise_scheduler_type.lower()

        return IterativeConfig(
            num_steps=num_steps,
            scheduler=scheduler_type,
            timestep_spacing="leading",
            timestep_range=[config.num_train_timesteps - 1, 0],
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )
    else:
        return IterativeConfig(
            num_steps=num_steps,
            scheduler="euler",
            timestep_spacing="linear",
            timestep_range=[1.0, 0.0],
        )


def _build_manifest(
    policy: PreTrainedPolicy,
    policy_name: str,
    inference_type: str,
    artifacts: dict[str, str],
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    normalization_config: NormalizationConfig | None,
    two_phase_config: TwoPhaseConfig | None,
    backend: str,
) -> Manifest:
    """Build the manifest for the exported policy."""
    config = policy.config

    action_dim = getattr(config, "max_action_dim", None)
    if action_dim is None:
        action_dim = config.action_feature.shape[0] if hasattr(config, "action_feature") else 14
    chunk_size = getattr(config, "chunk_size", None) or getattr(config, "horizon", 100)
    n_action_steps = getattr(config, "n_action_steps", chunk_size)

    policy_info = PolicyInfo(
        name=policy_name,
        source=PolicySource(
            repo_id=getattr(config, "repo_id", None),
            revision=getattr(config, "revision", None),
        ),
    )

    io_spec = IOSpec(inputs=input_specs, outputs=output_specs)

    action_spec = ActionSpec(
        dim=action_dim,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        representation="absolute",
    )

    # Build inference config based on type
    inference_config = None
    if inference_type == "iterative":
        inference_config = _build_iterative_config(policy)
    elif inference_type == "two_phase":
        inference_config = two_phase_config
    # single_pass: inference_config stays None

    try:
        import lerobot

        lerobot_version = getattr(lerobot, "__version__", None)
    except (ImportError, AttributeError):
        lerobot_version = None

    device = str(next(policy.parameters()).device)
    metadata = ExportMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by="lerobot.export",
        lerobot_version=lerobot_version,
        export_device=device,
        export_dtype="float32",
    )

    return Manifest(
        policy=policy_info,
        artifacts=artifacts,
        io=io_spec,
        action=action_spec,
        inference=inference_config,
        normalization=normalization_config,
        metadata=metadata,
    )
