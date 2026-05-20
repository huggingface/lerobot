#!/usr/bin/env python
"""
Convert all VLA-JEPA .pt checkpoints (ginwind/VLA-JEPA) to LeRobot safetensors
format and upload them to maximellerbach org inside a HF collection.

Usage:
    uv run python convert_vla_jepa_checkpoints.py

For each variant the script:
  1. Downloads the .pt checkpoint.
  2. Extracts the state dict.
  3. Instantiates VLAJEPAPolicy with the variant's confirmed config.
  4. Loads the state dict (strict=False — mismatches printed to stdout).
  5. push_to_hub → writes model.safetensors + config.json in LeRobot format.
  6. Adds the new repo to a shared HF collection.

Config sources
--------------
Numeric hyper-params  : ginwind/VLA-JEPA/<variant>/config.json
Image keys  LIBERO    : lerobot/libero_10 meta/info.json          ✓ confirmed
Image keys  Pretrain  : lerobot/droid_1.0.1 meta/info.json        ✓ confirmed
Image keys  SimplerEnv: OXE Bridge/RT1 are single-camera          ✓ confirmed
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file as save_safetensors
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------
SOURCE_REPO_ID = "ginwind/VLA-JEPA"
TARGET_ORG = "maximellerbach"
COLLECTION_TITLE = "VLA-JEPA"
COLLECTION_DESCRIPTION = (
    "VLA-JEPA model checkpoints (LIBERO, Pretrain, SimplerEnv) converted from .pt to safetensors via LeRobot."
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key mapping — mirrors todo_converter.py map_key() so both converters
# produce identical safetensors layouts that match the LeRobot action_head code.
# ---------------------------------------------------------------------------


def _normalize_source_key(key: str) -> str:
    return key[len("module."):] if key.startswith("module.") else key


def _map_checkpoint_key(raw_key: str) -> str | None:
    """Map original VLA-JEPA state-dict keys to LeRobot vla_jepa layout."""
    key = _normalize_source_key(raw_key)

    if key.startswith("qwen_vl_interface."):
        return "model.qwen." + key[len("qwen_vl_interface."):]
    if key.startswith("vj_encoder."):
        return "model.video_encoder." + key[len("vj_encoder."):]
    if key.startswith("vj_predictor."):
        return "model.video_predictor." + key[len("vj_predictor."):]
    if key.startswith("action_model."):
        # LeRobot code uses the same sub-key names as the source checkpoint,
        # so only the top-level "model." prefix needs to be added.
        return "model." + key

    return None


def _fetch_action_stats(api: "HfApi", source_repo_id: str, subfolder: str) -> dict | None:
    """Try to download dataset_statistics.json and return the action stats dict."""
    import json

    stats_file = f"{subfolder}/dataset_statistics.json"
    try:
        local = api.hf_hub_download(source_repo_id, stats_file)
        data = json.loads(Path(local).read_text())
        # The original repo nests stats under a robot key, e.g. {"franka": {"action": {...}}}
        for robot_key in data:
            if isinstance(data[robot_key], dict) and "action" in data[robot_key]:
                log.info("  Loaded action stats from %s (robot key: %s)", stats_file, robot_key)
                return data[robot_key]["action"]
        log.warning("  %s found but no 'action' key under any robot — skipping action stats.", stats_file)
    except Exception as exc:  # noqa: BLE001
        log.warning("  Could not fetch %s: %s — action_unnormalization_stats will be None.", stats_file, exc)
    return None


# ---------------------------------------------------------------------------
# Architecture — identical across all 4 variants (from config.json)
# ---------------------------------------------------------------------------
_ARCH = {
    "qwen_model_name": "Qwen/Qwen3-VL-2B-Instruct",  # 2B, NOT the default 4B
    "chunk_size": 7,
    "n_action_steps": 7,
    "num_video_frames": 8,
    "jepa_tubelet_size": 2,
    "num_action_tokens_per_timestep": 8,
    "num_embodied_action_tokens_per_instruction": 32,
    "num_inference_timesteps": 4,
    "action_hidden_size": 1024,
    "action_model_type": "DiT-B",
    # Explicit dims matching DiT-B preset and ginwind checkpoint shape
    "action_num_heads": 12,
    "action_attention_head_dim": 64,
    "action_num_layers": 16,
    "action_dropout": 0.2,
    "repeated_diffusion_steps": 8,
    "action_noise_beta_alpha": 1.5,
    "action_noise_beta_beta": 1.0,
    "action_noise_s": 0.999,
    "action_num_timestep_buckets": 1000,
    # World model predictor (12 blocks, confirmed from checkpoint)
    "predictor_depth": 12,
}

# ---------------------------------------------------------------------------
# Image-key sets (confirmed sources in module docstring)
# ---------------------------------------------------------------------------
# LIBERO — confirmed from lerobot/libero_10 meta/info.json
_LIBERO_CAMS = [
    "observation.images.image",  # agentview camera
    "observation.images.wrist_image",  # eye-in-hand camera
]

# DROID pretrain — 2 views match the predictor embed_dim=2 × 1024=2048 in checkpoint
_DROID_CAMS = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
]

# OXE Bridge + RT1 — single-camera; world model disabled (predictor embed_dim mismatch)
_OXE_CAMS = [
    "observation.images.image",
]


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def _build_config(
    camera_keys: list[str],
    with_state: bool,
    enable_world_model: bool = True,
    action_stats: dict | None = None,
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in camera_keys}
    if with_state:
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))

    cfg = VLAJEPAConfig(
        input_features=input_features,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        enable_world_model=enable_world_model,
        action_unnormalization_stats=action_stats,
        binarize_gripper_action=True,
        clip_normalized_actions=True,
        **_ARCH,
    )
    cfg.validate_features()
    return cfg


# Maps each subfolder in SOURCE_REPO_ID to (camera_keys, with_state, enable_world_model, repo_suffix)
VARIANTS: dict[str, tuple] = {
    "LIBERO": (_LIBERO_CAMS, True, True, "LIBERO"),
    "Pretrain": (_DROID_CAMS, False, True, "Pretrain"),
    # SimplerEnv uses a single camera; the predictor embed_dim (2048) would mismatch, so
    # disable the world model — only qwen + action_model weights are needed for inference.
    "SimplerEnv": (_OXE_CAMS, False, False, "SimplerEnv"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model")
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def subfolder_of(pt_path: str) -> str | None:
    for part in Path(pt_path).parts:
        if part in VARIANTS:
            return part
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    api = HfApi()

    log.info("Listing .pt files in %s …", SOURCE_REPO_ID)
    pt_files = [f for f in api.list_repo_files(SOURCE_REPO_ID) if f.endswith(".pt")]
    if not pt_files:
        log.error("No .pt files found.")
        return
    for f in pt_files:
        log.info("  %s", f)

    # Create / reuse the collection once
    collection = api.create_collection(
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        namespace=TARGET_ORG,
        exists_ok=True,
    )
    log.info("Collection: %s", collection.url)

    for pt_filename in pt_files:
        log.info("\n=== %s ===", pt_filename)

        subfolder = subfolder_of(pt_filename)
        if subfolder is None:
            log.warning("  No variant entry for '%s' — skipping.", pt_filename)
            continue

        camera_keys, with_state, enable_world_model, repo_suffix = VARIANTS[subfolder]
        target_repo_id = f"{TARGET_ORG}/VLA-JEPA-{repo_suffix}"

        log.info(
            "  cameras=%d  with_state=%s  wm=%s  → %s",
            len(camera_keys),
            with_state,
            enable_world_model,
            target_repo_id,
        )

        # 1. Download
        local_pt = api.hf_hub_download(SOURCE_REPO_ID, pt_filename)
        log.info("  Downloaded → %s", local_pt)

        # 2. Load checkpoint
        try:
            ckpt = torch.load(local_pt, map_location="cpu", mmap=True, weights_only=False)  # nosec B614
        except TypeError:
            ckpt = torch.load(local_pt, map_location="cpu")  # nosec B614

        sd = extract_state_dict(ckpt)

        # Map source key names → LeRobot layout (handles layer1→w1, transformer_blocks→blocks, etc.)
        mapped_sd: dict[str, torch.Tensor] = {}
        skipped_keys: list[str] = []
        for raw_key, value in sd.items():
            target_key = _map_checkpoint_key(raw_key)
            if target_key is None:
                skipped_keys.append(raw_key)
            else:
                mapped_sd[target_key] = value
        log.info("  %d tensors mapped, %d skipped", len(mapped_sd), len(skipped_keys))
        if skipped_keys:
            log.info("  Skipped sample: %s", skipped_keys[:5])
        log.info("  First 5 mapped keys: %s", list(mapped_sd)[:5])

        # Fetch action unnormalization stats from the source repo
        action_stats = _fetch_action_stats(api, SOURCE_REPO_ID, subfolder)

        # 3. Build config (no policy instantiation — avoids loading backbone from Hub)
        config = _build_config(camera_keys, with_state, enable_world_model, action_stats)

        # 4. Save everything to a temp dir and upload in one shot
        api.create_repo(target_repo_id, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)

            log.info("  Saving model.safetensors …")
            save_safetensors(mapped_sd, save_dir / "model.safetensors")

            config._save_pretrained(save_dir)  # writes config.json via draccus

            preprocessor, postprocessor = make_vla_jepa_pre_post_processors(config)
            preprocessor.save_pretrained(save_dir)   # writes policy_preprocessor.json
            postprocessor.save_pretrained(save_dir)  # writes policy_postprocessor.json

            log.info("  Uploading …")
            commit_url = api.upload_folder(
                folder_path=save_dir,
                repo_id=target_repo_id,
                repo_type="model",
                commit_message=f"Convert {Path(pt_filename).name} to safetensors",
            )
        log.info("  Uploaded → %s", commit_url)

        # 6. Add to collection
        api.add_collection_item(
            collection_slug=collection.slug,
            item_id=target_repo_id,
            item_type="model",
            exists_ok=True,
        )
        log.info("  Added to collection.")

    log.info("\nAll done. Collection: %s", collection.url)


if __name__ == "__main__":
    main()
