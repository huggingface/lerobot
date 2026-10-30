# Copyright 2026 Shaohua Pan.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the repository root LICENSE for the full license.
# Adapted from Streaming-WAM, Xuyao Huang et al.
# Source revision: ff36d6e66514afd65bf957a06a0a2410f182b0e3.

"""Convert the pinned public AC-Stream checkpoint without materializing FP32 weights.

Run with ``python -m lerobot.policies.fastwam.convert_streaming_checkpoint
--checkpoint /path/ac-stream/model.pt --stats /path/ac-stream/dataset_stats.json
--output /path/native``. Run under an external RSS/swap guard on constrained hosts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_fastwam import FastWAMConfig
from .modeling_fastwam import FastWAMPolicy
from .processor_fastwam import make_fastwam_pre_post_processors
from .wan.streaming import StreamingActionDiT, StreamingMoT, StreamingVideoDiT, StreamingWAM

REFERENCE_REVISION = "ff36d6e66514afd65bf957a06a0a2410f182b0e3"
HUB_REVISION = "8dcc67ce14c72a5cce0c04c87a518b4ccfec217e"
HUB_REPO = "SJTU-DENG-Lab/Streaming-WAM"
PHASE = "stage2_1step_selfatt_z1"
ARCHITECTURE = "stage1_16slot_z1_only_fixed_residual_plus_d0d8_policy_stream_v1"


def release_config() -> FastWAMConfig:
    """Return the released LIBERO recipe, with explicit camera order and minmax stats."""
    return FastWAMConfig(
        streaming_wam=True,
        num_inference_steps=1,
        base_model_id=None,
        device="cpu",
        dtype=torch.bfloat16,
        fp32_attention=False,
        load_text_encoder=False,
        toggle_action_dimensions=[-1],
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
        },
    )


def native_schema(config: FastWAMConfig) -> nn.Module:
    """Construct only the native parameter schema on meta, without frozen assets."""
    if config.video_dit_config is None or config.action_dit_config is None:
        raise ValueError("Expert configurations must be resolved.")
    with torch.device("meta"):
        video = StreamingVideoDiT(
            **config.video_dit_config, stream_action_hidden_dim=config.action_dit_config["hidden_dim"]
        )
        action = StreamingActionDiT(**config.action_dit_config)
        mot = StreamingMoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
        with torch.device("cpu"):
            core = StreamingWAM(
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
        return container.to(dtype=config.dtype)


def map_checkpoint(payload: dict[str, Any], target: dict[str, Tensor]) -> dict[str, Tensor]:
    """Validate metadata and map released expert keys to the native single owners."""
    for name, expected in (("training_rtc_phase", PHASE), ("training_rtc_architecture", ARCHITECTURE)):
        if payload.get(name) != expected:
            raise ValueError(f"Streaming-WAM {name} mismatch: {payload.get(name)!r}.")
    if str(payload.get("torch_dtype")) not in ("bfloat16", "torch.bfloat16") or "step" not in payload:
        raise ValueError("Streaming-WAM requires BF16 release metadata and a step.")
    mapped: dict[str, Tensor] = {}
    for key, tensor in payload["mot"].items():
        match = re.fullmatch(r"mixtures\.(video|action)\.blocks\.(\d+)\.(.+)", key)
        if match:
            expert, index, suffix = match.groups()
            destination = f"model.mot.layers.{index}.blocks.{expert}.{suffix}"
        elif key.startswith(("mixtures.video.", "mixtures.action.")):
            destination = f"model.mot.{key}"
        else:
            raise ValueError(f"Unsupported checkpoint key: {key}.")
        if destination in mapped:
            raise ValueError(f"Duplicate mapped key: {destination}.")
        mapped[destination] = tensor
    for key, tensor in payload["proprio_encoder"].items():
        mapped[f"model.proprio_encoder.{key}"] = tensor
    if set(mapped) != set(target):
        raise ValueError(
            f"Checkpoint key mismatch: missing={sorted(set(target) - set(mapped))[:20]}, "
            f"unexpected={sorted(set(mapped) - set(target))[:20]}."
        )
    for key, tensor in mapped.items():
        if (
            not isinstance(tensor, Tensor)
            or tensor.shape != target[key].shape
            or tensor.dtype != target[key].dtype
        ):
            raise ValueError(f"Checkpoint shape/dtype mismatch for {key}.")
    return mapped


def load_stats(path: Path) -> dict[str, dict[str, Tensor]]:
    raw = json.loads(path.read_text())
    stats = {}
    for source, name, dim in (("action", ACTION, 7), ("state", OBS_STATE, 8)):
        values = raw[source]["default"]
        stats[name] = {
            key: torch.tensor(values[f"global_{key}"], dtype=torch.float32)
            for key in ("min", "max", "mean", "std")
        }
        if any(value.shape != (dim,) or not torch.isfinite(value).all() for value in stats[name].values()):
            raise ValueError(f"Invalid released {source} statistics.")
        if ((stats[name]["max"] - stats[name]["min"]) < 1e-6).any():
            raise ValueError(f"Released {source} minmax ranges must be at least 1e-6.")
    return stats


def convert(checkpoint: Path, stats_path: Path, output: Path) -> dict[str, Any]:
    """Save native artifacts and require a strict public policy reload."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    config = release_config()
    schema = native_schema(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = map_checkpoint(payload, schema.state_dict())
    schema.load_state_dict(state, strict=True, assign=True)
    stats = load_stats(stats_path)
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        output / "model.safetensors",
        metadata={"format": "pt", "source_revision": REFERENCE_REVISION, "hub_revision": HUB_REVISION},
    )
    config.save_pretrained(output)
    pre, post = make_fastwam_pre_post_processors(config, stats)
    pre.save_pretrained(output, config_filename="policy_preprocessor.json")
    post.save_pretrained(output, config_filename="policy_postprocessor.json")
    del state, payload, schema
    policy = FastWAMPolicy.from_pretrained(
        output, strict=True, local_files_only=True, init_empty_weights=True
    )
    report = {
        "source_revision": REFERENCE_REVISION,
        "hub_revision": HUB_REVISION,
        "strict_reload": True,
        "tensors": len(policy.state_dict()),
        "parameter_bytes": sum(t.numel() * t.element_size() for t in policy.state_dict().values()),
    }
    policy.close()
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.stats, args.output), indent=2))


if __name__ == "__main__":
    main()
