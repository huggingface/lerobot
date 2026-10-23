# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Assemble the frozen model package from anonymous upstream downloads at image build time."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def checksum(path: Path) -> str:
    """Hash a large checkpoint without copying its tensors into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(root: Path, name: str) -> Path:
    """Confine every provisioned file to the model package."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or "mano" in path.name.lower():
        raise ValueError("Unexpected path in the hand model package")
    return path


def verify(root: Path, sources: dict) -> None:
    """Require exactly the same model bytes as the accepted native release."""
    manifest = root / "assets.json"
    if checksum(manifest) != sources["manifest_sha256"]:
        raise ValueError("The frozen model manifest changed")
    document = json.loads(manifest.read_text())
    for entry in document["files"]:
        path = asset_path(root, entry["path"])
        if path.stat().st_size != entry["size"] or checksum(path) != entry["sha256"]:
            raise ValueError(f"Frozen model asset differs: {entry['path']}")
    print(f"Verified {len(document['files'])} frozen hand/camera asset files", flush=True)


def assemble(root: Path, sources: dict) -> None:
    """Fetch public weights once, strip training state, and validate the complete package."""
    import torch
    from huggingface_hub import hf_hub_download

    if checksum(root / "assets.json") != sources["manifest_sha256"]:
        raise ValueError("The frozen model manifest changed")
    for source in sources["downloads"]:
        output = asset_path(root, source["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading public model {source['repo']} at {source['revision']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="hand-model-build-") as temporary:
            downloaded = Path(
                hf_hub_download(
                    source["repo"],
                    source["file"],
                    revision=source["revision"],
                    token=False,
                    local_dir=temporary,
                )
            )
            if checksum(downloaded) != source["sha256"]:
                raise ValueError(f"Upstream model changed: {source['repo']}")
            if "inference_tensors" in source:
                # This is the pinned public NVIDIA checkpoint, verified above before unpickling.
                # mmap avoids materializing its optimizer state. No tensor conversion is applied.
                checkpoint = torch.load(downloaded, map_location="cpu", mmap=True, weights_only=False)  # nosec B614
                state = checkpoint["state_dict"]
                if len(state) != source["inference_tensors"]:
                    raise ValueError("The upstream EgoHand tensor catalog changed")
                torch.save({"state_dict": state}, output)
                del state, checkpoint
            else:
                shutil.move(str(downloaded), output)
    verify(root, sources)


def main() -> None:
    """Build or check the model package without credentials, CUDA or inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    if args.verify_only:
        verify(args.root, sources)
    else:
        assemble(args.root, sources)


if __name__ == "__main__":
    main()
