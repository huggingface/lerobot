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
from pathlib import Path

import torch
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------
SOURCE_REPO_ID = "ginwind/VLA-JEPA"
TARGET_ORG = "lerobot"
COLLECTION_TITLE = "VLA-JEPA"
COLLECTION_DESCRIPTION = (
    "VLA-JEPA model checkpoints (LIBERO, Pretrain, SimplerEnv) converted from .pt to safetensors via LeRobot."
)

# Remap state-dict key prefixes before loading into the LeRobot policy.
# E.g. {"": "model."} prepends "model." to every key.
# Leave empty if keys already match — the first run's log will tell you.
KEY_PREFIX_REMAP: dict[str, str] = {
    # Specific rules must come before the "" catch-all (dict order is preserved).
    "qwen_vl_interface.": "model.qwen.",
    "vj_encoder.": "model.video_encoder.",
    "vj_predictor.": "model.video_predictor.",
    # Everything else (action_model.*) just needs the "model." wrapper.
    "": "model.",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture — identical across all 4 variants (from config.json)
# ---------------------------------------------------------------------------
_ARCH = {
    "qwen_model_name": "Qwen/Qwen3-VL-2B-Instruct",  # 2B, NOT the default 4B
    "chunk_size": 7,
    "n_action_steps": 7,
    "future_action_window_size": 6,
    "num_video_frames": 8,
    "jepa_tubelet_size": 2,
    "num_action_tokens_per_timestep": 8,
    "num_embodied_action_tokens_per_instruction": 32,
    "num_inference_timesteps": 4,
    "action_hidden_size": 1024,
    "action_model_type": "DiT-B",
    "action_num_layers": 16,
    "action_dropout": 0.2,
    "repeated_diffusion_steps": 8,
    "action_noise_beta_alpha": 1.5,
    "action_noise_beta_beta": 1.0,
    "action_noise_s": 0.999,
    "action_num_timestep_buckets": 1000,
    # Action head embedding params (from original config.json)
    "num_target_vision_tokens": 32,
    "action_max_seq_len": 1024,
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


def _build_config(camera_keys: list[str], with_state: bool, enable_world_model: bool = True):
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


def remap_keys(sd: dict[str, torch.Tensor], remap: dict[str, str]) -> dict[str, torch.Tensor]:
    if not remap:
        return sd
    out = {}
    for k, v in sd.items():
        new_k = k
        for old, new in remap.items():
            if k.startswith(old):
                new_k = new + k[len(old) :]
                break
        out[new_k] = v
    return out


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
        sd = remap_keys(sd, KEY_PREFIX_REMAP)
        log.info("  %d tensors extracted", len(sd))
        log.info("  First 5 keys: %s", list(sd)[:5])

        # 3. Build policy
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        config = _build_config(camera_keys, with_state, enable_world_model)
        policy = VLAJEPAPolicy(config)

        # 4. Load weights
        missing, unexpected = policy.load_state_dict(sd, strict=False)

        def _prefix_summary(keys: list[str]) -> dict[str, int]:
            from collections import Counter

            return dict(Counter(".".join(k.split(".")[:3]) for k in keys).most_common())

        if missing:
            log.warning("  Missing    (%d) by prefix: %s", len(missing), _prefix_summary(missing))
        if unexpected:
            log.warning("  Unexpected (%d) by prefix: %s", len(unexpected), _prefix_summary(unexpected))
        if not missing and not unexpected:
            log.info("  State dict loaded cleanly.")

        # 5. Push to hub (writes model.safetensors + config.json)
        api.create_repo(target_repo_id, repo_type="model", exist_ok=True)
        commit_url = policy.push_to_hub(
            repo_id=target_repo_id,
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
