# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
from lerobot.annotations import hub  # noqa: E402
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig  # noqa: E402

SHA = "a" * 40
COMMIT = "b" * 40


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for relative, data in [
        ("data/chunk-000/file-000.parquet", b"parquet"),
        ("meta/info.json", b'{"codebase_version":"v3.0"}'),
        ("videos/camera/file.mp4", b"video"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "README.md").write_text(
        "---\nlicense: cc-by-4.0\ntags: [robotics]\n---\n# Original publisher\nOriginal attribution and links.\n"
    )
    lineage = tmp_path / "remote-lineage.json"
    lineage.write_text(
        json.dumps(
            {"processed_repo_id": "lerobot/example", "source": {"repo_id": "owner/source", "revision": SHA}}
        )
    )
    calls = []

    class API:
        def dataset_info(self, repo_id, *, revision=None):
            return SimpleNamespace(sha=SHA, private=False)

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(tags=[SimpleNamespace(name="v3.0")])

        def create_commit(self, **kwargs):
            operations = {
                op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in kwargs["operations"]
            }
            calls.append(("commit", {**kwargs, "operations": operations}))
            return SimpleNamespace(oid=COMMIT)

        def create_tag(self, **kwargs):
            calls.append(("tag", kwargs))

    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **k: str(lineage))
    cfg = AnnotationPipelineConfig(
        repo_id="lerobot/example",
        revision=SHA,
        new_repo_id="lerobot/example",
        release_tag="finerobotics-v0",
        expected_target_revision=SHA,
        redistribution_permission="Publisher CC-BY-4.0 terms checked; attribution retained",
    )
    return root, cfg, calls, API


def test_one_guarded_commit_preserves_license_and_skips_video(publication):
    root, cfg, calls, _ = publication
    hub.publish_annotation(
        root, cfg, changed_paths=[root / "data/chunk-000/file-000.parquet", root / "meta/info.json"]
    )
    commit = calls[0][1]
    assert commit["parent_commit"] == SHA
    assert "videos/camera/file.mp4" not in commit["operations"]
    card = commit["operations"]["README.md"].decode()
    assert "license: cc-by-4.0" in card and "Original attribution and links." in card
    assert "finerobotics\n" not in card
    assert "lerobot/example" in card and "revision='finerobotics-v0'" in card
    assert json.loads(commit["operations"][hub.LINEAGE])["upstream_source"]["repo_id"] == "owner/source"
    assert hub.LINEAGE in commit["operations"]
    assert calls[1] == (
        "tag",
        {"repo_id": "lerobot/example", "repo_type": "dataset", "tag": "finerobotics-v0", "revision": COMMIT},
    )
    assert len(calls) == 2  # no format-tag deletion/repoint or separate card commit


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("new_repo_id", None, "explicit"),
        ("revision", "main", "pinned"),
        ("release_tag", "v3.0", "distinct"),
        ("expected_target_revision", "c" * 40, "Target changed"),
        ("redistribution_permission", None, "permission"),
        ("max_publish_files", 1, "single-commit"),
    ],
)
def test_publication_guards(publication, field, value, match):
    root, cfg, calls, _ = publication
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        hub.publish_annotation(root, cfg, changed_paths=[root / "meta/info.json"])
    assert not calls


def test_concurrent_change_is_visible(publication, monkeypatch):
    root, cfg, calls, api_class = publication

    def concurrent(self, **kwargs):
        raise RuntimeError("parent commit changed")

    monkeypatch.setattr(api_class, "create_commit", concurrent)
    with pytest.raises(RuntimeError, match="parent commit changed"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls


def test_tag_failure_is_visible(publication, monkeypatch):
    root, cfg, calls, api_class = publication

    def fail(self, **kwargs):
        raise RuntimeError("tag failed")

    monkeypatch.setattr(api_class, "create_tag", fail)
    with pytest.raises(RuntimeError, match="tag failed"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert len(calls) == 1


def test_missing_license_is_not_invented(publication):
    root, cfg, calls, _ = publication
    (root / "README.md").write_text("---\ntags: [robotics]\n---\nOriginal unlicensed source.\n")
    cfg.redistribution_permission = "Written publisher permission stored in legal review record"
    hub.publish_annotation(root, cfg, changed_paths=[])
    card = calls[0][1]["operations"]["README.md"].decode()
    assert "apache" not in card and "license:" not in card


def test_existing_release_tag_not_moved(publication, monkeypatch):
    root, cfg, calls, api_class = publication
    monkeypatch.setattr(
        api_class,
        "list_repo_refs",
        lambda *a, **k: SimpleNamespace(tags=[SimpleNamespace(name=cfg.release_tag)]),
    )
    with pytest.raises(ValueError, match="already exists"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls


def test_new_processed_target_complete_copy_and_preflight(publication, monkeypatch):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    root, cfg, calls, api_class = publication
    cfg.repo_id, cfg.expected_target_revision = "owner/source", None

    def absent_until_created(self, repo_id, *, revision=None):
        if repo_id == cfg.repo_id:
            return SimpleNamespace(sha=SHA, private=False)
        if not any(name == "create_repo" for name, _ in calls):
            raise RepositoryNotFoundError(
                "not found",
                response=httpx.Response(404, request=httpx.Request("GET", "https://test.example")),
            )
        return SimpleNamespace(sha=SHA, private=False)

    monkeypatch.setattr(api_class, "dataset_info", absent_until_created)
    cfg.max_publish_files = 1
    with pytest.raises(ValueError, match="single-commit"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls  # not even an empty remote repo before preflight passes
    cfg.max_publish_files = 512
    hub.publish_annotation(root, cfg, changed_paths=[])
    assert calls[0][0] == "create_repo"
    commit = calls[1][1]
    assert commit["parent_commit"] == SHA
    assert "videos/camera/file.mp4" in commit["operations"]
    assert "data/chunk-000/file-000.parquet" in commit["operations"]


def test_update_cannot_patch_unrelated_publisher_source(publication):
    root, cfg, calls, _ = publication
    cfg.repo_id = "owner/unrelated"
    with pytest.raises(ValueError, match="start from.*processed"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls


def test_preflight_runs_before_model_setup(publication, single_episode_root, monkeypatch):
    from lerobot.annotations.processing import run_annotation_pipeline
    from tests.annotations.test_processing import CLIENTS, FACTORY

    _, cfg, calls, _ = publication
    cfg.push_to_hub = True
    cfg.redistribution_permission = None
    CLIENTS.clear()
    with pytest.raises(ValueError, match="permission"):
        run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert not CLIENTS and not calls


@pytest.mark.parametrize("new_target", [False, True])
def test_preflight_counts_generated_shard_metadata_before_inference(
    publication, fixture_dataset_root, monkeypatch, new_target
):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    from lerobot.annotations.processing import run_annotation_pipeline
    from tests.annotations.test_processing import CLIENTS, FACTORY

    _, cfg, calls, api_class = publication
    root = fixture_dataset_root
    cfg.push_to_hub = True
    cfg.interjections.enabled = cfg.vqa.enabled = False
    cfg.vlm.auto_serve = False
    cfg.max_publish_files = 7  # 2 data + 2 ownership + 2 windows + info + README + lineage = 9
    if new_target:
        cfg.repo_id, cfg.expected_target_revision = "owner/source", None

        def absent(self, repo_id, *, revision=None):
            if repo_id == cfg.repo_id:
                return SimpleNamespace(sha=SHA, private=False)
            raise RepositoryNotFoundError(
                "absent", response=httpx.Response(404, request=httpx.Request("GET", "https://test.example"))
            )

        monkeypatch.setattr(api_class, "dataset_info", absent)
    CLIENTS.clear()
    with pytest.raises(ValueError, match="single-commit"):
        run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert not CLIENTS and not calls


@pytest.mark.parametrize(
    "source_private,requested,allow_public,expected",
    [
        (False, None, False, False),
        (False, True, False, True),
        (True, None, False, True),
        (True, False, False, None),
        (True, False, True, False),
        (True, True, False, True),
    ],
)
def test_publication_inherits_visibility_and_requires_private_source_opt_in(
    publication, monkeypatch, source_private, requested, allow_public, expected
):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    root, cfg, calls, api_class = publication
    cfg.repo_id, cfg.expected_target_revision = "owner/source", None
    cfg.push_private = requested
    cfg.allow_public_from_private = allow_public
    revisions = []

    def info(self, repo_id, *, revision=None):
        if repo_id == cfg.repo_id:
            revisions.append(revision)
            return SimpleNamespace(sha=SHA, private=source_private)
        if not calls:
            raise RepositoryNotFoundError(
                "absent", response=httpx.Response(404, request=httpx.Request("GET", "https://test.example"))
            )
        return SimpleNamespace(sha=SHA, private=calls[0][1]["private"])

    monkeypatch.setattr(api_class, "dataset_info", info)
    if expected is None:
        for action in (
            hub.preflight_publication,
            lambda root, cfg: hub.publish_annotation(root, cfg, changed_paths=[]),
        ):
            with pytest.raises(ValueError, match="allow_public_from_private"):
                action(root, cfg)
        assert not calls
    else:
        hub.publish_annotation(root, cfg, changed_paths=[])
        assert calls[0][0] == "create_repo" and calls[0][1]["private"] is expected
    assert revisions and all(revision == SHA for revision in revisions)


@pytest.mark.parametrize("missing_visibility", [False, True])
def test_existing_target_visibility_cannot_be_ignored(publication, monkeypatch, missing_visibility):
    root, cfg, calls, api_class = publication
    if missing_visibility:
        monkeypatch.setattr(api_class, "dataset_info", lambda *a, **k: SimpleNamespace(sha=SHA, private=None))
    else:
        cfg.push_private = True  # Existing processed target is public; never silently ignore this.
    with pytest.raises(ValueError, match="visibility"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls


def test_visibility_is_rechecked_at_publication(publication, monkeypatch):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    root, cfg, calls, api_class = publication
    cfg.repo_id, cfg.expected_target_revision = "owner/source", None
    cfg.push_private = False
    source_private = False

    def info(self, repo_id, *, revision=None):
        if repo_id == cfg.repo_id:
            return SimpleNamespace(sha=SHA, private=source_private)
        raise RepositoryNotFoundError(
            "absent", response=httpx.Response(404, request=httpx.Request("GET", "https://test.example"))
        )

    monkeypatch.setattr(api_class, "dataset_info", info)
    hub.preflight_publication(root, cfg)
    source_private = True
    with pytest.raises(ValueError, match="allow_public_from_private"):
        hub.publish_annotation(root, cfg, changed_paths=[])
    assert not calls


def test_local_source_publishes_with_content_provenance(
    publication, unaligned_episode_root, fragmented_parquet_reader, monkeypatch
):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    from lerobot.annotations import processing
    from lerobot.annotations.processing_source import LeRobotEpisodeSource
    from tests.annotations._helpers import StubFrameProvider
    from tests.annotations.test_processing import FACTORY, config

    _, cfg, calls, api_class = publication
    cfg.repo_id = cfg.revision = cfg.expected_target_revision = None

    def absent(self, repo_id):
        if not calls:
            raise RepositoryNotFoundError(
                "not found",
                response=httpx.Response(404, request=httpx.Request("GET", "https://test.example")),
            )
        return SimpleNamespace(sha=SHA)

    monkeypatch.setattr(api_class, "dataset_info", absent)
    root = unaligned_episode_root
    original = LeRobotEpisodeSource(root).dataset_ref.revision
    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    assert processing.run_annotation_pipeline(config(), root, client_factory=FACTORY).validation_report.ok
    hub.publish_annotation(root, cfg, changed_paths=[])
    assert calls[0][1]["private"] is True  # Unknown local-only visibility defaults to private.
    source = json.loads(calls[1][1]["operations"][hub.LINEAGE])["source"]
    assert source == {"type": "local", "native_content_sha256": original}
