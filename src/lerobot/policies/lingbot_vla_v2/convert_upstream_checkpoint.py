# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Convert the raw upstream LingBot-VLA 2.0 checkpoint to the LeRobot format.

The released upstream checkpoint (``robbyant/lingbot-vla-v2-6b``) is a sharded
safetensors checkpoint whose ``config.json`` only carries ``{"vlm_family": "qwen3_vl"}``.
This one-shot tool repackages it as a standard LeRobot checkpoint:

- ``config.json`` with ``type: lingbot_vla_v2`` and the full architecture overrides,
- a single ``model.safetensors`` written by ``PreTrainedPolicy.save_pretrained``,
- ``policy_preprocessor.json`` / ``policy_postprocessor.json`` with the robot config
  and normalization stats embedded, so the result is portable across machines.

Example:
    python -m lerobot.policies.lingbot_vla_v2.convert_upstream_checkpoint \
        --input robbyant/lingbot-vla-v2-6b \
        --output ./lingbot-vla-v2-6b-lerobot \
        --robot-config-path ./omx_multicubes_robot_config.yaml \
        --norm-stats-path ./omx_multicubes_norm_stats.json

    # optionally push straight to the Hub:
    python -m lerobot.policies.lingbot_vla_v2.convert_upstream_checkpoint \
        --input robbyant/lingbot-vla-v2-6b \
        --output ./lingbot-vla-v2-6b-lerobot \
        --robot-config-path ./omx_multicubes_robot_config.yaml \
        --push-to-hub <user>/lingbot-vla-v2-6b-lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config
from .processor_lingbot_vla_v2 import make_lingbot_vla_v2_pre_post_processors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPSTREAM_REPO_ID = "robbyant/lingbot-vla-v2-6b"
SAFE_WEIGHTS_INDEX = "model.safetensors.index.json"

# Architecture values that the released 6B weights were trained with. Applied on top
# of the dataclass defaults defensively so the converted config always matches them.
UPSTREAM_CONFIG_OVERRIDES = {
    "use_moe": True,
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "token_moe_intermediate_size": 512,
    "token_shared_intermediate_size": 704,
    "expert_hidden_size": 768,
    "router_activation": "sigmoid",
    "routed_scaling_factor": 4.0,
    "use_shared_expert_gate": False,
    "moe_implementation": "fused",
    "use_depth": False,
    "max_action_dim": 55,
    "max_state_dim": 55,
}

# Upstream-only tensors that the action path does not consume when use_depth=False:
# the depth / video predictive-distillation branches.
ALLOWED_SKIPPED_PREFIXES = (
    "model.current_video_align_",
    "model.future_video_align_",
    "model.depth_align_",
    "model.future_depth_align_",
    "model.current_shared_task_proj.",
    "model.future_shared_task_proj.",
)


def _resolve_upstream_checkpoint(input_path: str, revision: str | None = None) -> str:
    if Path(input_path).is_dir():
        return input_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=input_path,
        revision=revision,
        allow_patterns=[
            "config.json",
            "configuration.json",
            SAFE_WEIGHTS_INDEX,
            "model-*.safetensors",
            "tokenizer*",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
        ],
    )


def split_upstream_loading_keys(
    model_keys: set[str], checkpoint_keys: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split load-state keys into (missing, allowed-skipped, hard-unexpected).

    Missing keys and hard-unexpected keys mean the upstream checkpoint does not match
    the LingBot-VLA 2.0 architecture and the conversion must fail; allowed-skipped
    keys are the upstream-only depth / video distillation tensors (use_depth=False).
    """
    missing = sorted(model_keys - checkpoint_keys)
    skipped = sorted(k for k in checkpoint_keys - model_keys if k.startswith(ALLOWED_SKIPPED_PREFIXES))
    unexpected = sorted(k for k in checkpoint_keys - model_keys if not k.startswith(ALLOWED_SKIPPED_PREFIXES))
    return missing, skipped, unexpected


def _load_upstream_weights(policy, checkpoint_dir: str) -> tuple[int, int]:
    """Load the sharded upstream weights onto the (CPU) policy, validating coverage."""
    from safetensors.torch import load_file

    index_path = Path(checkpoint_dir) / SAFE_WEIGHTS_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"{SAFE_WEIGHTS_INDEX} not found in {checkpoint_dir}")

    with index_path.open() as f:
        weight_map = json.load(f).get("weight_map", {})

    model_keys = set(policy.state_dict().keys())
    missing_keys, skipped_keys, unexpected_keys = split_upstream_loading_keys(
        model_keys, set(weight_map.keys())
    )
    if missing_keys or unexpected_keys:
        parts = []
        if missing_keys:
            parts.append(f"missing required keys ({len(missing_keys)}): {', '.join(missing_keys[:5])}")
        if unexpected_keys:
            parts.append(
                f"unexpected non-whitelisted keys ({len(unexpected_keys)}): {', '.join(unexpected_keys[:5])}"
            )
        raise RuntimeError("Upstream checkpoint load failed: " + "; ".join(parts))

    # Shards load on CPU: this is a one-shot conversion tool and must not grab a GPU.
    loaded_keys = 0
    for shard_name in sorted(set(weight_map.values())):
        shard_state = load_file(str(Path(checkpoint_dir) / shard_name), device="cpu")
        kept = {key: tensor for key, tensor in shard_state.items() if key in model_keys}
        policy.load_state_dict(kept, strict=False)
        loaded_keys += len(kept)
    return loaded_keys, len(skipped_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", default=UPSTREAM_REPO_ID, help="Upstream repo id or local checkpoint directory."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the converted LeRobot checkpoint."
    )
    parser.add_argument("--revision", default=None, help="Hub revision of the upstream checkpoint.")
    parser.add_argument("--robot-config-path", required=True, help="Per-embodiment robot config YAML.")
    parser.add_argument(
        "--norm-stats-path",
        default=None,
        help="Normalization stats JSON (defaults to the path inside the robot config).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local source to load the Qwen3-VL tokenizer/processor from (e.g. the upstream "
        "checkpoint directory itself). The saved configs still record the portable Hub id.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="Optionally push the converted checkpoint to this Hub repo.",
    )
    args = parser.parse_args()

    checkpoint_dir = _resolve_upstream_checkpoint(args.input, args.revision)
    logger.info("Converting upstream checkpoint from %s", checkpoint_dir)

    # Declare the dataset-side camera keys this robot config reads from (its images
    # origin_keys), so the saved config.json lines up with the dataset's visual
    # features. The canonical slot mapping itself happens inside the feature transform.
    import yaml

    with open(args.robot_config_path) as f:
        _robot_config = yaml.safe_load(f)
    camera_keys = []
    for entry in _robot_config.get("images", []):
        if isinstance(entry, str):
            camera_keys.append(entry)
            continue
        for convert_info in entry.values():
            origin_keys = convert_info.get("origin_keys") if isinstance(convert_info, dict) else None
            if isinstance(origin_keys, str):
                camera_keys.append(origin_keys)
            elif isinstance(origin_keys, list):
                for item in origin_keys:
                    if isinstance(item, dict):
                        camera_keys.extend(item.keys())

    config = LingbotVLAV2Config(
        robot_config_path=args.robot_config_path,
        norm_stats_path=args.norm_stats_path,
        tokenizer_path=args.tokenizer_path or LingbotVLAV2Config().tokenizer_path,
        # Canonical features; training re-infers them from the dataset anyway.
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(55,)),
            **{cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for cam in camera_keys},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(55,))},
    )
    for key, value in UPSTREAM_CONFIG_OVERRIDES.items():
        setattr(config, key, value)
    config._moe_implementation = config.moe_implementation

    from .modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    logger.info("Building the 6B policy on CPU (this takes a while and ~40 GB RAM)...")
    policy = LingbotVLAV2Policy(config)

    loaded, skipped = _load_upstream_weights(policy, checkpoint_dir)
    logger.info(
        "Loaded %d tensors, %d upstream-only (depth/video distillation) tensors skipped.", loaded, skipped
    )

    # Building the processors resolves + embeds the robot config / norm stats into the
    # config, so the saved checkpoint no longer references machine-specific paths.
    preprocessor, postprocessor = make_lingbot_vla_v2_pre_post_processors(config)

    # Restore the portable Hub tokenizer/processor ids before saving: the conversion may
    # have loaded them from a local directory, but the checkpoint must not serialize it.
    if args.tokenizer_path:
        config.tokenizer_path = LingbotVLAV2Config().tokenizer_path
        config.processor_path = None
        for step in preprocessor.steps:
            if getattr(step, "processor_path", None):
                step.processor_path = config.tokenizer_path

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    logger.info("Converted checkpoint written to %s", output_dir)

    if args.push_to_hub:
        policy.save_pretrained(output_dir, repo_id=args.push_to_hub, push_to_hub=True)
        preprocessor.save_pretrained(output_dir, repo_id=args.push_to_hub, push_to_hub=True)
        postprocessor.save_pretrained(output_dir, repo_id=args.push_to_hub, push_to_hub=True)
        logger.info("Pushed to https://huggingface.co/%s", args.push_to_hub)


if __name__ == "__main__":
    main()
