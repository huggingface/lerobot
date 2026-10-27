#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Convert a released Zero-WAM checkpoint (diffusers layout) into a LeRobot ``zero_wam`` policy.

Only the transformer is converted; the frozen VAE, UMT5 text encoder and tokenizer keep loading
from ``--policy.frozen_components_path``. The robot layout (cameras, resolution, temporal stride and
action channels) is part of the policy config, so convert once per robot:

    # RoboTwin (the layout of the released checkpoints)
    python examples/zero_wam/convert_checkpoint.py \
        --source robbyant-research/zero-wam-pretrain --output outputs/zero_wam_robotwin --for-training

    # Bimanual joint-space robot (e.g. ReBot: 2 x (6 joints + gripper), 30 fps, 3 cameras)
    python examples/zero_wam/convert_checkpoint.py \
        --source robbyant-research/zero-wam-pretrain --output outputs/zero_wam_rebot --for-training \
        --cameras observation.images.base observation.images.left_wrist observation.images.right_wrist \
        --video-frame-stride 2 --action-channels 14 15 16 17 18 19 28 21 22 23 24 25 26 29
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.zero_wam.configuration_zero_wam import ROBOTWIN_ACTION_CHANNELS, ZeroWAMConfig
from lerobot.policies.zero_wam.modeling_zero_wam import ZeroWAMPolicy
from lerobot.policies.zero_wam.processor_zero_wam import make_zero_wam_pre_post_processors
from lerobot.utils.constants import ACTION

ROBOTWIN_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def load_transformer_state_dict(
    source: str, revision: str | None = None
) -> tuple[dict[str, torch.Tensor], dict, Path]:
    root = Path(source)
    if not root.is_dir():
        root = Path(snapshot_download(source, revision=revision, allow_patterns=["transformer/*"]))
    transformer = root / "transformer"
    config = json.loads((transformer / "config.json").read_text())
    state_dict: dict[str, torch.Tensor] = {}
    for shard in sorted(transformer.glob("*.safetensors")):
        tensors = load_file(shard)
        duplicates = state_dict.keys() & tensors.keys()
        if duplicates:
            raise ValueError(f"Duplicate checkpoint keys in {shard.name}: {sorted(duplicates)}")
        state_dict.update(tensors)
    if not state_dict:
        raise ValueError(f"No safetensors found in {transformer}")
    return state_dict, config, root


def load_action_stats(path: Path, channels: list[int]) -> dict[str, dict[str, torch.Tensor]]:
    """Read LeRobot q01/q99, or the official RoboTwin stats in model-channel order."""
    data = json.loads(path.read_text())
    if "norm_stats" in data:
        if data.get("method") != "abs":
            raise ValueError("Official RoboTwin stats must declare method=abs")
        hand = data["norm_stats"]["action.hand.position"]
        gripper = data["norm_stats"]["action.effector.position"]
        if set(channels) - set(ROBOTWIN_ACTION_CHANNELS):
            raise ValueError("RoboTwin EEF statistics cannot normalize joint-space channels")
        stats = {k: torch.tensor(hand[k] + [0.0] * 14 + gripper[k])[channels] for k in ("q01", "q99")}
    else:
        stats = {k: torch.tensor(data[ACTION][k], dtype=torch.float32) for k in ("q01", "q99")}
    for value in stats.values():
        if value.shape != (len(channels),) or not torch.isfinite(value).all():
            raise ValueError("Action quantiles must be finite vectors in dataset action order")
    if (stats["q99"] < stats["q01"]).any():
        raise ValueError("q99 must be greater than or equal to q01")
    return {ACTION: stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="robbyant-research/zero-wam-pretrain")
    parser.add_argument("--revision", help="Pin the Hugging Face source checkpoint revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=ROBOTWIN_CAMERAS)
    parser.add_argument("--camera-shape", nargs=3, type=int, default=[3, 480, 640], metavar=("C", "H", "W"))
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--video-frame-stride", type=int, default=4)
    parser.add_argument("--action-channels", nargs="+", type=int, default=ROBOTWIN_ACTION_CHANNELS)
    parser.add_argument(
        "--no-mcp", action="store_true", help="Drop the training-only MCP heads (~1.4B params)"
    )
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    stats_group = parser.add_mutually_exclusive_group(required=True)
    stats_group.add_argument(
        "--action-stats", type=Path, help="LeRobot stats.json or official RoboTwin q01/q99 JSON"
    )
    stats_group.add_argument(
        "--for-training",
        action="store_true",
        help="Omit stats; fit them from the fine-tuning dataset before inference",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory must be empty: {args.output}")

    state_dict, upstream, source_root = load_transformer_state_dict(args.source, args.revision)
    if upstream.get("_class_name") != "WanICLTransformer3DModel":
        raise ValueError(f"Expected a WanICLTransformer3DModel checkpoint, got {upstream.get('_class_name')}")

    architecture = {
        key: upstream[key]
        for key in (
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "action_dim",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "eps",
            "rope_max_seq_len",
            "action_inner_dim",
            "action_ffn_dim",
            "attn_window",
            "num_mcp_modules",
            "mcp_blocks_per_group",
        )
    }
    config = ZeroWAMConfig(
        patch_size=tuple(upstream["patch_size"]),
        mcp_hidden_collect_layers=tuple(upstream["mcp_hidden_collect_layers"]),
        enable_mcp=bool(upstream.get("enable_mcp", True)) and not args.no_mcp,
        obs_cam_keys=list(args.cameras),
        height=args.height,
        width=args.width,
        video_frame_stride=args.video_frame_stride,
        used_action_channel_ids=list(args.action_channels),
        dtype=getattr(torch, args.dtype),
        frozen_components_path=str(source_root) if Path(args.source).is_dir() else args.source,
        frozen_components_revision=source_root.name if not Path(args.source).is_dir() else None,
        device="cpu",
        input_features={
            key: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(args.camera_shape))
            for key in args.cameras
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(args.action_channels),))},
        **architecture,
    )

    with torch.device("meta"):
        policy = ZeroWAMPolicy(config)
    dropped = sorted(k for k in state_dict if k.startswith("mcp_") and not config.enable_mcp)
    if dropped:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mcp_")}
    dtype = getattr(torch, args.dtype)
    prefixed = {f"transformer.{k}": v.to(dtype) for k, v in state_dict.items()}
    expected = policy.state_dict()
    report = {
        "source": args.source,
        "revision": config.frozen_components_revision,
        "mapped": sorted(prefixed.keys() & expected.keys()),
        "missing": sorted(expected.keys() - prefixed.keys()),
        "unexpected": sorted(prefixed.keys() - expected.keys()),
        "shape_mismatches": {
            k: {"source": list(v.shape), "target": list(expected[k].shape)}
            for k, v in prefixed.items()
            if k in expected and v.shape != expected[k].shape
        },
        "dropped_mcp": dropped,
        "action_stats": str(args.action_stats) if args.action_stats else None,
        "for_training": args.for_training,
    }
    stats = load_action_stats(args.action_stats, args.action_channels) if args.action_stats else None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["missing"] or report["unexpected"] or report["shape_mismatches"]:
        raise RuntimeError(f"Checkpoint mismatch; see {args.output / 'conversion_report.json'}")
    policy.load_state_dict(prefixed, strict=True, assign=True)
    policy.transformer.tie_cross_attention_aliases()
    policy.config.device = "cpu"
    policy.save_pretrained(args.output)
    # Processors carry no statistics yet: lerobot-train fills them from the training dataset.
    for processor in make_zero_wam_pre_post_processors(policy.config, stats):
        processor.save_pretrained(args.output)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Saved {n_params / 1e9:.2f}B parameters to {args.output}")


if __name__ == "__main__":
    main()
