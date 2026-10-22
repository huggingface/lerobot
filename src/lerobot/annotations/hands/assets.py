# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""User-controlled MANO assets. No public MANO checkpoint fallback."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError


def file_checksum(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest(), path.stat().st_size


MANO_FILES = {"left": "MANO_LEFT.safetensors", "right": "MANO_RIGHT.safetensors"}


def resolve_mano(
    repo: str, token: str, acknowledged: bool, *, repo_type: str | None = None, api_factory=HfApi
) -> dict:
    if not acknowledged:
        raise ValueError("Confirm your MANO and model rights for non-commercial research first")
    api = api_factory(token=token)
    owner = api.whoami()["name"]
    if not repo or repo.split("/")[0] != owner:
        raise ValueError("Choose a MANO repository in your personal Hugging Face account")
    if repo_type not in {None, "model", "dataset"}:
        raise ValueError("MANO assets require a model or dataset repository")
    kind = repo_type or "model"
    try:
        info = api.repo_info(repo, repo_type=kind)
    except RepositoryNotFoundError:
        if repo_type is not None:
            raise
        kind = "dataset"
        info = api.repo_info(repo, repo_type=kind)
    if not info.private:
        raise ValueError(
            "MANO must come from your private licensed asset repository; public mirrors are disabled"
        )
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("MANO assets require an immutable Hub revision")
    files = set(api.list_repo_files(repo, repo_type=kind, revision=info.sha))
    if not set(MANO_FILES.values()) <= files:
        raise ValueError(
            "The private repository must contain MANO_LEFT.safetensors and MANO_RIGHT.safetensors"
        )
    return {
        "repo_id": repo,
        "repo_type": kind,
        "revision": info.sha,
        "files": MANO_FILES,
        "research_rights_acknowledged": True,
    }


def download_mano(
    source: dict, token: str, root: Path, *, api_factory=HfApi, download=hf_hub_download
) -> dict:
    # Recheck privacy, ownership and immutable revision in the user's Job before GPU setup.
    pinned = resolve_mano(
        source["repo_id"],
        token,
        source.get("research_rights_acknowledged", False),
        repo_type=source.get("repo_type", "model"),
        api_factory=api_factory,
    )
    if pinned["revision"] != source["revision"] or source["files"] != MANO_FILES:
        raise ValueError("MANO assets changed after the estimate; create a fresh estimate")
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for side, filename in MANO_FILES.items():
        path = Path(
            download(
                source["repo_id"],
                filename,
                repo_type=pinned["repo_type"],
                revision=source["revision"],
                token=token,
            )
        )
        # Safetensors is validated by the model loader; never unpickle uploaded model files.
        target = root / filename
        shutil.copyfile(path, target)
        digest, size = file_checksum(target)
        receipts[side] = {"path": str(target), "sha256": digest, "size": size}
    (root / "receipt.json").write_text(json.dumps({"source": source, "files": receipts}))
    return receipts


def checkpoint_guard(directory: Path, receipts: dict) -> Path:
    """Apply to every native subprocess via PYTHONPATH/sitecustomize.

    Only the two exact MANO filenames may be resolved, and only to the user's
    checksum-verified files. All other model downloads obey the offline cache.
    """
    for side in MANO_FILES:
        entry = receipts[side]
        if file_checksum(Path(entry["path"])) != (entry["sha256"], entry["size"]):
            raise ValueError("Private MANO asset checksum mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {filename: receipts[side]["path"] for side, filename in MANO_FILES.items()}
    script = (
        "import huggingface_hub\n"
        f"_mano_paths = {paths!r}\n"
        "_download = huggingface_hub.hf_hub_download\n"
        "def _licensed_download(repo_id, filename, *args, **kwargs):\n"
        "    if 'mano' in filename.lower():\n"
        "        if filename not in _mano_paths:\n"
        "            raise RuntimeError('Unapproved MANO checkpoint request')\n"
        "        return _mano_paths[filename]\n"
        "    return _download(repo_id, filename, *args, **kwargs)\n"
        "huggingface_hub.hf_hub_download = _licensed_download\n"
    )
    (directory / "sitecustomize.py").write_text(script)
    return directory
