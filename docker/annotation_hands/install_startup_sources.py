"""Restore additive frozen Python vendors without replacing native source or operators."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

PREFIX = PurePosixPath("release-195fps-verified/DROID-W/thirdparty")
PACKAGES = {"gaussian_splatting", "depth_anything_v2"}


def checksum(path):
    """Hash the small source bundle or manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(archive, manifest, version, destination):
    """Verify every source/notices byte before adding missing vendor files."""
    archive, manifest, version, destination = map(Path, (archive, manifest, version, destination))
    identity = json.loads(version.read_text())["source_files"]
    document = json.loads(manifest.read_text())
    if checksum(manifest) != identity[manifest.name] or checksum(archive) != identity[archive.name]:
        raise ValueError("Startup source checksum differs from the pinned image")
    if document.get("python_sources_only") is not True or any(
        document.get(key) is not False for key in ("compiled_binaries", "model_weights")
    ):
        raise ValueError("Startup vendors must contain only Python source and notices")
    if checksum(archive) != document["archive_sha256"]:
        raise ValueError("Startup archive and source manifest differ")
    entries = {entry["path"]: entry for entry in document["files"]}
    if not entries or len(entries) != len(document["files"]):
        raise ValueError("Startup source paths must be nonempty and unique")
    verified = []
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) != len(entries) or {member.name for member in members} != set(entries):
            raise ValueError("Startup archive contains unlisted sources")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or not name.is_relative_to(PREFIX)
                or ".." in name.parts
                or len(name.parts) < 5
                or name.parts[3] not in PACKAGES
                or not (
                    name.suffix == ".py"
                    or name.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "AUTHORS"))
                )
            ):
                raise ValueError("Startup vendors must be additive Python source and notices")
            payload = source.extractfile(member).read()
            entry = entries[member.name]
            if len(payload) != entry["size"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("Startup vendor file checksum mismatch")
            raw_target = destination / str(name)
            if raw_target.exists() or raw_target.is_symlink():
                raise FileExistsError("Refuse to replace frozen native source or an operator")
            target = raw_target.resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Startup vendor escaped its installation root")
            verified.append((target, payload))
    for package in {PurePosixPath(name).parts[3] for name in entries}:
        if not any(
            PurePosixPath(name).parts[3] == package
            and PurePosixPath(name).name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
            for name in entries
        ):
            raise ValueError("Each startup vendor must retain its license notice")
    for target, payload in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(
        f"Verified {len(verified)} additive frozen Python sources/notices; no operators or weights",
        flush=True,
    )


if __name__ == "__main__":
    install(*sys.argv[1:])
