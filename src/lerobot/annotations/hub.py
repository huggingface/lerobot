# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Guarded releases with immutable release tags and a current format-version alias."""

import json
import re
import tempfile
from pathlib import Path

import draccus
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from .runtime.bundles import check_credentials
from .runtime.types import canonical_json, checked_name

LINEAGE = "meta/processing_release.json"


def _publication_context(root, cfg):
    """Read-only target, permission and lineage checks, repeated at final publish."""
    if not cfg.new_repo_id:
        raise ValueError("Publication requires an explicit --new_repo_id processed target")
    if cfg.repo_id and not re.fullmatch(r"[0-9a-f]{40}", cfg.revision or ""):
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
    api = HfApi()
    target = cfg.new_repo_id
    previous = {}
    target_info = None
    try:
        target_info = api.dataset_info(target)
        current = target_info.sha
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
    source_info = None
    if cfg.repo_id:
        source_info = (
            target_info if cfg.repo_id == target else api.dataset_info(cfg.repo_id, revision=cfg.revision)
        )
    for info in (source_info, target_info):
        if info is not None and not isinstance(getattr(info, "private", None), bool):
            raise ValueError("Cannot determine source/target visibility; refusing publication")
    private = cfg.push_private
    if target_info is not None:
        private = target_info.private
        if cfg.push_private is not None and cfg.push_private != private:
            raise ValueError(
                "Existing target visibility differs from push_private; it is not changed automatically"
            )
    elif private is None:
        private = source_info.private if source_info is not None else True
    if source_info is not None and source_info.private and not private and not cfg.allow_public_from_private:
        raise ValueError("Public publication of a private source requires allow_public_from_private=true")
    # Do not delete or repoint an immutable release after a rerun.
    if current is not None:
        tags = {tag.name for tag in api.list_repo_refs(target, repo_type="dataset").tags}
        if cfg.release_tag in tags:
            raise ValueError("Release tag already exists; choose a new release tag")
    check_credentials(draccus.encode(cfg))
    return api, current, previous, private


def _source_files(root, human_video_output="human_videos"):
    files = {
        f"{name}/{path.relative_to((root / name).resolve()).as_posix()}": path
        for name in ("data", "meta", "videos", human_video_output)
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
    return files


def _check_limits(files, cfg, *, extra_files=0):
    if (
        len(files) + extra_files > cfg.max_publish_files
        or sum(path.stat().st_size for path in files.values()) > cfg.max_publish_bytes
    ):
        raise ValueError(
            "Release exceeds single-commit limits; use an explicit staged large-release workflow"
        )


def preflight_publication(root, cfg, *, planned_paths=()):
    """Fail before inference on known publication blockers; final sizes are rechecked."""
    root = Path(root).resolve()
    _, current, _, _ = _publication_context(root, cfg)
    planned = set(planned_paths)
    paths = set(_source_files(root, cfg.human_video.output_dir)) if current is None else set()
    paths.update(planned | {"README.md", LINEAGE})
    # Count all planned outputs on both create and update, with no double counting.
    # Only unchanged source bytes are a known lower bound; generated sizes may shrink.
    unchanged = {
        name: root / name for name in paths - planned - {"README.md", LINEAGE} if (root / name).is_file()
    }
    _check_limits(unchanged, cfg, extra_files=len(paths) - len(unchanged))


def publish_annotation(root, cfg, *, changed_paths):
    """One guarded release; updates never re-upload unchanged videos."""
    root = Path(root).resolve()
    api, current, previous, private = _publication_context(root, cfg)
    target = cfg.new_repo_id
    card_path = root / "README.md"
    card = DatasetCard(card_path.read_text() if card_path.exists() else "---\n{}\n---\n# Processed dataset\n")
    if current is None:
        files = _source_files(root, cfg.human_video.output_dir)
    else:
        files = {}
        for path in changed_paths:
            path = Path(path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Changed publication path escapes its dataset root")
            relative = path.relative_to(root).as_posix()
            prefixes = ("data/", "meta/")
            if cfg.human_video.enabled:
                prefixes += (cfg.human_video.output_dir + "/",)
            if relative.startswith("videos/") or not relative.startswith(prefixes):
                raise ValueError("Annotation updates only publish changed data/metadata, not videos")
            files[relative] = path
    if cfg.repo_id:
        source = {"repo_id": cfg.repo_id, "revision": cfg.revision}
        attribution = f"[{cfg.repo_id}](https://huggingface.co/datasets/{cfg.repo_id}/tree/{cfg.revision})"
    else:
        from .processing_source import LeRobotEpisodeSource

        source = {
            "type": "local",
            "native_content_sha256": LeRobotEpisodeSource(
                root, cache_dir=cfg.resolved_staging_dir(root)
            ).dataset_ref.revision,
        }
        attribution = f"a local dataset (native content SHA256 `{source['native_content_sha256']}`)"
    # Reuse the result-affecting cache contract, not the operational constructor
    # config. Import here to keep publication checks independent of model modules.
    from .processing import LanguageModule

    processing = {
        phase: LanguageModule(root, phase, draccus.encode(cfg)).scientific_config()
        for phase in ("plan", "interjections", "vqa")
        if getattr(cfg, phase).enabled
    }
    if cfg.human_video.enabled:
        from .human_video_processing import HumanVideoStage

        processing["human_video"] = HumanVideoStage(root, draccus.encode(cfg)).scientific_config()
    images = (cfg.job.image, cfg.runtime.hf_jobs.cpu_image, cfg.runtime.hf_jobs.gpu_image)
    code_revision = cfg.runtime.hf_jobs.code_revision or cfg.job.lerobot_ref
    lineage = {
        "version": 2,
        "processed_repo_id": target,
        "source": source,
        "parent_commit": current,
        "release_tag": cfg.release_tag,
        "private": private,
        "redistribution_permission": cfg.redistribution_permission,
        "processing": processing,
        "only_episodes": cfg.only_episodes,
        "validation_enabled": not cfg.skip_validation,
        "pins": {
            "code_revision": code_revision if re.fullmatch(r"[0-9a-f]{40}", code_revision or "") else None,
            "image_sha256": sorted(
                {
                    image.rsplit("@sha256:", 1)[1]
                    for image in images
                    if image and re.search(r"@sha256:[0-9a-f]{64}$", image)
                }
            ),
        },
    }
    if cfg.expected_target_revision:
        lineage["upstream_source"] = previous.get("upstream_source", previous.get("source"))
    check_credentials(lineage)
    card.data.tags = list(dict.fromkeys([*(getattr(card.data, "tags", None) or []), "lerobot"]))
    card.text += (
        f"\n\n## Processing release: {cfg.release_tag}\n\n"
        f"This processed dataset is based on {attribution}. "
        "The original remains with its publisher. Native signals and source labels are retained; "
        "enabled LeRobot language stages add canonical Parquet annotations. "
        f"Scientific settings, available pins and permission notes: `{LINEAGE}`. "
        "Default loads follow the latest format-version tag; pin a release tag or commit for reproducibility.\n\n"
        "```python\nfrom lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"dataset = LeRobotDataset({target!r}, revision={cfg.release_tag!r})\n```\n"
    )
    with tempfile.TemporaryDirectory(prefix="lerobot-publish-") as directory:
        staging = Path(directory)
        (staging / "README.md").write_text(str(card))
        (staging / "lineage.json").write_bytes(canonical_json(lineage))
        files["README.md"] = staging / "README.md"
        files[LINEAGE] = staging / "lineage.json"
        _check_limits(files, cfg)
        if current is None:
            api.create_repo(repo_id=target, repo_type="dataset", private=private, exist_ok=False)
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
    # Match LeRobotDataset.push_to_hub: the format tag is a mutable compatibility
    # alias used by default loads, not an immutable processing release.
    if version in tags:
        api.delete_tag(repo_id=target, repo_type="dataset", tag=version)
    api.create_tag(repo_id=target, repo_type="dataset", tag=version, revision=commit.oid)
    # A tag failure is an error, never swallowed as publication success.
    return commit
