#!/usr/bin/env python
"""Convert the released BimanualYAM model and tagged processors for lerobot-rollout.

No robot is instantiated, CAN opened, or motion commanded by this program.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import YAM_FEATURE_NAMES
from lerobot.utils.constants import ACTION, OBS_STATE

CHECKPOINT = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
IMAGE_KEYS = [f"observation.images.{name}" for name in ("top", "left", "right")]


def validate_metadata(metadata: dict) -> None:
    if metadata.get("control_mode") != "absolute joint pose":
        raise ValueError(
            "This YAM adapter accepts absolute joints, not end-effector poses; IK is required for EE"
        )
    if metadata.get("camera_keys") != IMAGE_KEYS:
        raise ValueError("Expected camera order top, left, right")
    if metadata.get("normalize_gripper") is not False:
        raise ValueError("Expected unnormalized continuous grippers: 0 closed, 1 open")
    expected_mask = [True] * 6 + [False] + [True] * 6 + [False]
    for key in ("action_stats", "state_stats"):
        stats = metadata[key]
        if stats.get("names") != list(YAM_FEATURE_NAMES) or stats.get("mask") != expected_mask:
            raise ValueError(f"Unexpected joint order or gripper mask in {key}")
        if any(len(stats.get(name, [])) != 14 for name in ("q01", "q99")):
            raise ValueError(f"Missing 14-dimensional quantile stats in {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    revision = args.revision or HfApi().model_info(CHECKPOINT).sha
    stats_path = hf_hub_download(CHECKPOINT, "norm_stats.json", revision=revision)
    metadata = json.loads(Path(stats_path).read_text())["metadata_by_tag"][NORM_TAG]
    validate_metadata(metadata)
    source_config = json.loads(
        Path(hf_hub_download(CHECKPOINT, "config.json", revision=revision)).read_text()
    )
    if source_config.get("action_mode") != "both":
        raise ValueError("Revalidate the conversion for this checkpoint's training action mode")
    config = MolmoAct2Config(
        checkpoint_path=CHECKPOINT,
        checkpoint_revision=revision,
        norm_tag=NORM_TAG,
        train_mode_vlm="fft",
        # Preserve native BOTH-mode attention masking when reloading the saved
        # checkpoint; inference still exclusively uses the continuous expert.
        action_mode="both",
        inference_action_mode="continuous",
        device="cuda",
        dtype=args.dtype,
        compile_model=False,
        chunk_size=30,
        n_action_steps=30,
        num_inference_steps=10,
        image_keys=IMAGE_KEYS,
        normalize_gripper=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for key in IMAGE_KEYS},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        dataset_feature_names={OBS_STATE: list(YAM_FEATURE_NAMES), ACTION: list(YAM_FEATURE_NAMES)},
    )
    print(f"Loading {CHECKPOINT}@{revision}", flush=True)
    policy = MolmoAct2Policy(config).to("cuda").eval()
    # These are inference-only processors: no discrete action labels/tokenizer
    # are needed. Keep the policy's BOTH metadata unchanged for attention masks.
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(
        replace(config, action_mode="continuous")
    )
    # Save the original quantile stats and gripper masks with the checkpoint.
    # Rollout must load these saved processors, not estimate stats from live data.
    policy.save_pretrained(args.output)
    preprocessor.save_pretrained(args.output)
    postprocessor.save_pretrained(args.output)
    (args.output / "yam_source.json").write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "revision": revision,
                "norm_tag": NORM_TAG,
                "camera_keys": IMAGE_KEYS,
                "joint_names": list(YAM_FEATURE_NAMES),
                "control_mode": metadata["control_mode"],
                "dtype": args.dtype,
            },
            indent=2,
        )
    )
    print(f"Saved rollout checkpoint and processors: {args.output}", flush=True)
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
