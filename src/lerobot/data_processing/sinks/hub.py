# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""One guarded, bounded release commit; never infer a licence or move an old tag."""

import json
import re
import tempfile
from pathlib import Path

import draccus
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from ..bundles import check_credentials
from ..types import canonical_json, checked_name

LINEAGE = "meta/processing_release.json"


def publish_annotation(root, cfg, *, changed_paths):
    """Publish after validation. Large conversion releases need a staging workflow.

    An existing processed repo requires an expected parent and matching remote
    lineage. A new target gets a complete dataset; updates upload only supplied
    changed Parquet/metadata, README and lineage, not unchanged videos.
    """
    if not cfg.new_repo_id:
        raise ValueError("Publication requires an explicit --new_repo_id processed target")
    if not cfg.repo_id or not re.fullmatch(r"[0-9a-f]{40}", cfg.revision or ""):
        raise ValueError("Publication requires a pinned upstream repo_id/revision for provenance")
    if not cfg.release_tag:
        raise ValueError("Publication requires an immutable release_tag")
    checked_name(cfg.release_tag)
    if cfg.release_tag in {"main", "master"} or re.fullmatch(r"v[0-9]+\.[0-9]+", cfg.release_tag):
        raise ValueError("Use a processing release tag distinct from format compatibility tags")
    # Permission is an explicit human assertion, not a runtime's legal opinion
    # that e.g. a non-commercial licence permits public redistribution.
    if not cfg.redistribution_permission:
        raise ValueError("Record explicit redistribution_permission before publishing")
    root = Path(root).resolve()
    card_path = root / "README.md"
    card = DatasetCard(card_path.read_text() if card_path.exists() else "---\n{}\n---\n# Processed dataset\n")
    # Preserve missing licences as missing; permission notes never become Apache.
    api = HfApi()
    target = cfg.new_repo_id
    try:
        current = api.dataset_info(target).sha
    except RepositoryNotFoundError:
        current = None
    if current is not None:
        if cfg.expected_target_revision != current:
            raise ValueError("Target changed or expected_target_revision is missing; refusing overwrite")
        previous = json.loads(
            Path(hf_hub_download(target, LINEAGE, repo_type="dataset", revision=current)).read_text()
        )
        if previous.get("processed_repo_id") != target:
            raise ValueError("Target is not a processed dataset owned by this workflow")
        if target != cfg.repo_id or cfg.revision != current:
            raise ValueError("Annotation must start from the expected processed target revision")
    elif target == cfg.repo_id:
        raise ValueError("Publisher inputs are not implicit publication targets")
    elif cfg.expected_target_revision:
        raise ValueError("Expected target revision was provided for an absent repository")
    # Do not delete or repoint an immutable release after a rerun.
    if current is not None:
        tags = {tag.name for tag in api.list_repo_refs(target, repo_type="dataset").tags}
        if cfg.release_tag in tags:
            raise ValueError("Release tag already exists; choose a new release tag")
    if current is None:
        files = {
            f"{name}/{path.relative_to((root / name).resolve()).as_posix()}": path
            for name in ("data", "meta", "videos")
            for path in (root / name).resolve().rglob("*")
            if path.is_file()
        }
        files.update(
            {
                name: root / name
                for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE")
                if (root / name).is_file()
            }
        )
    else:
        files = {}
        for path in changed_paths:
            path = Path(path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Changed publication path escapes its dataset root")
            relative = path.relative_to(root).as_posix()
            if relative.startswith("videos/") or not relative.startswith(("data/", "meta/")):
                raise ValueError("Annotation updates only publish changed data/metadata, not videos")
            files[relative] = path
    lineage = {
        "version": 1,
        "processed_repo_id": target,
        "source": {"repo_id": cfg.repo_id, "revision": cfg.revision},
        "parent_commit": current,
        "release_tag": cfg.release_tag,
        "redistribution_permission": cfg.redistribution_permission,
        "processing": draccus.encode(cfg),
    }
    if cfg.expected_target_revision:
        lineage["upstream_source"] = previous.get("upstream_source", previous.get("source"))
    check_credentials(lineage)
    card.data.tags = list(dict.fromkeys([*(card.data.tags or []), "lerobot", "finerobotics"]))
    card.text += (
        f"\n\n## Processing release: {cfg.release_tag}\n\n"
        f"This processed dataset is based on [{cfg.repo_id}](https://huggingface.co/datasets/{cfg.repo_id}/tree/{cfg.revision}). "
        "The original remains with its publisher. Native signals and source labels are retained; "
        "enabled LeRobot language stages add canonical Parquet annotations. "
        f"Exact settings and permission notes: `{LINEAGE}`.\n\n"
        "```python\nfrom lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"dataset = LeRobotDataset({target!r}, revision={cfg.release_tag!r})\n```\n"
    )
    with tempfile.TemporaryDirectory(prefix="lerobot-publish-") as directory:
        staging = Path(directory)
        (staging / "README.md").write_text(str(card))
        (staging / "lineage.json").write_bytes(canonical_json(lineage))
        files["README.md"] = staging / "README.md"
        files[LINEAGE] = staging / "lineage.json"
        if (
            len(files) > cfg.max_publish_files
            or sum(path.stat().st_size for path in files.values()) > cfg.max_publish_bytes
        ):
            raise ValueError(
                "Release exceeds single-commit limits; use an explicit staged large-release workflow"
            )
        if current is None:
            api.create_repo(repo_id=target, repo_type="dataset", private=cfg.push_private, exist_ok=False)
            current = api.dataset_info(target).sha
            lineage["parent_commit"] = current
            (staging / "lineage.json").write_bytes(canonical_json(lineage))
        commit = api.create_commit(
            repo_id=target,
            repo_type="dataset",
            parent_commit=current,
            commit_message=cfg.push_commit_message or f"Processing release {cfg.release_tag}",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in sorted(files.items())
            ],
        )
    api.create_tag(repo_id=target, repo_type="dataset", tag=cfg.release_tag, revision=commit.oid)
    info = json.loads((root / "meta/info.json").read_text())
    version = info["codebase_version"]
    tags = {tag.name for tag in api.list_repo_refs(target, repo_type="dataset").tags}
    if version not in tags:
        api.create_tag(repo_id=target, repo_type="dataset", tag=version, revision=commit.oid)
    # A tag failure is an error, never swallowed as publication success.
    return commit
