# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Verify the provisioned public EgoHand/SOMA models and calibrated camera runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import CHECKPOINT_SHA256, HandBackendConfig


def file_checksum(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def verify_assets(config: HandBackendConfig) -> dict:
    manifest = Path(config.assets_manifest)
    if file_checksum(manifest)[0] != config.assets_sha256:
        raise ValueError("The native asset manifest differs from the pinned release")
    assets = json.loads(manifest.read_text())
    listed = {}
    for entry in assets["files"]:
        path = (manifest.parent / entry["path"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("The release asset escaped its workspace")
        if "mano" in path.name.lower():
            raise ValueError("The EgoHand/SOMA runtime must not contain MANO assets")
        if file_checksum(path) != (entry["sha256"], entry["size"]):
            raise ValueError(f"Native runtime asset checksum mismatch: {path.name}")
        listed[str(path.resolve())] = entry
    for name in ("checkpoint", "detector", "droid_moge_checkpoint"):
        if str(Path(getattr(config, name)).resolve()) not in listed:
            raise ValueError(f"The release manifest must pin {name}")
    if listed[str(Path(config.checkpoint).resolve())]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Use the verified inference-only EgoHand checkpoint; no model conversion at launch")
    if assets.get("identity_backend") != "mhr" or assets.get("gt_scale_alignment") is not False:
        raise ValueError("The native release requires SOMA/MHR and no GT scale alignment")
    return assets


def download_release(profile: dict, token: str | None, root: Path) -> dict:
    """Use embedded or public immutable assets without forwarding the Job token."""
    from huggingface_hub import hf_hub_download, snapshot_download

    # Keep the token argument for existing callers. Asset distribution is public;
    # authentication belongs to Jobs and the user's datasets, never model downloads.
    options = {
        "repo_type": profile.get("repo_type", "model"),
        "revision": profile["revision"],
        "token": False,
    }
    embedded = profile.get("embedded") is True
    manifest = (
        root / "assets.json" if embedded else Path(hf_hub_download(profile["repo"], "assets.json", **options))
    )
    if file_checksum(manifest)[0] != profile["manifest_sha256"]:
        raise ValueError("The hand asset release manifest changed")
    document = json.loads(manifest.read_text())
    root.mkdir(parents=True, exist_ok=True)
    for entry in document["files"]:
        name = Path(entry["path"])
        if name.is_absolute() or ".." in name.parts or "mano" in name.name.lower():
            raise ValueError("Unexpected file in the native hand release")
    if not embedded:
        snapshot_download(
            profile["repo"],
            **options,
            local_dir=root,
            allow_patterns=[entry["path"] for entry in document["files"]],
            max_workers=16,
        )
        (root / "assets.json").write_bytes(manifest.read_bytes())
    return {
        "release_revision": profile["revision"],
        "manifest_sha256": profile["manifest_sha256"],
        "files": len(document["files"]),
        "bytes": sum(entry["size"] for entry in document["files"]),
    }
