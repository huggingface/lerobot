# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Build frozen model assets from public sources using bounded-memory copies."""

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote


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


def create_template(checkpoint: Path, destination: Path) -> None:
    """Store accepted checkpoint metadata and tensor hashes, without tensor values."""
    segments = []
    cursor = 0
    with (
        checkpoint.open("rb") as raw,
        zipfile.ZipFile(checkpoint) as archive,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as template,
    ):
        prefix = archive.namelist()[0].split("/")[0] + "/"
        for entry in archive.infolist():
            name = entry.filename.removeprefix(prefix)
            if not name.startswith("data/"):
                continue
            if entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Expected uncompressed PyTorch tensor records")
            raw.seek(entry.header_offset)
            header = struct.unpack("<4s5H3I2H", raw.read(30))
            start = entry.header_offset + 30 + header[-2] + header[-1]
            raw.seek(cursor)
            gap = raw.read(start - cursor)
            key = f"layout/{len(segments)}"
            template.writestr(key, gap)
            segments.append({"layout": key})
            digest = hashlib.sha256()
            with archive.open(entry) as tensor:
                for block in iter(lambda: tensor.read(8 * 1024**2), b""):
                    digest.update(block)
            segments.append({"tensor": name, "size": entry.file_size, "sha256": digest.hexdigest()})
            cursor = start + entry.file_size
        raw.seek(cursor)
        template.writestr("layout/tail", raw.read())
        segments.append({"layout": "layout/tail"})
        template.writestr("segments.json", json.dumps(segments, separators=(",", ":")))


def rebuild_checkpoint(source: Path, output: Path, template_path: Path) -> None:
    """Stream public tensor records into the accepted container; never unpickle weights."""
    with (
        zipfile.ZipFile(source) as archive,
        zipfile.ZipFile(template_path) as template,
        output.open("wb") as target,
    ):
        prefix = archive.namelist()[0].split("/")[0] + "/"
        for segment in json.loads(template.read("segments.json")):
            if "layout" in segment:
                target.write(template.read(segment["layout"]))
                continue
            entry = archive.getinfo(prefix + segment["tensor"])
            if entry.file_size != segment["size"]:
                raise ValueError("The upstream tensor catalog changed")
            digest = hashlib.sha256()
            with archive.open(entry) as tensor:
                for block in iter(lambda: tensor.read(8 * 1024**2), b""):
                    digest.update(block)
                    target.write(block)
            if digest.hexdigest() != segment["sha256"]:
                raise ValueError(f"Upstream tensor differs: {segment['tensor']}")


def download_public(source: dict, target: Path) -> None:
    """Stream an HTTPS upstream download without tokens or parallel Xet buffers."""
    url = f"https://huggingface.co/{quote(source['repo'], safe='/')}/resolve/{quote(source['revision'], safe='')}/{quote(source['file'], safe='/')}"
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as stream:  # nosec B310 - fixed HTTPS Hugging Face origin
        shutil.copyfileobj(response, stream, length=8 * 1024**2)


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


def assemble(root: Path, sources: dict, template_root: Path) -> None:
    """Fetch public weights once and build an offline, byte-identical model package."""
    if checksum(root / "assets.json") != sources["manifest_sha256"]:
        raise ValueError("The frozen model manifest changed")
    for source in sources["downloads"]:
        output = asset_path(root, source["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading public model {source['repo']} at {source['revision']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="hand-model-build-") as temporary:
            downloaded = Path(temporary) / "upstream.pt"
            download_public(source, downloaded)
            if checksum(downloaded) != source["sha256"]:
                raise ValueError(f"Upstream model changed: {source['repo']}")
            if "template" in source:
                template = asset_path(template_root, source["template"])
                if checksum(template) != source["template_sha256"]:
                    raise ValueError("The accepted checkpoint layout changed")
                rebuild_checkpoint(downloaded, output, template)
            else:
                shutil.move(str(downloaded), output)
    verify(root, sources)


def main() -> None:
    """Build or check assets without credentials, model imports or inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    if args.verify_only:
        verify(args.root, sources)
    else:
        assemble(args.root, sources, args.sources.parent)


if __name__ == "__main__":
    main()
