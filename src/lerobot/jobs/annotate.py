# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Compatibility launcher: one pinned HF Jobs node, serialized resolved config."""

import copy

import draccus
from huggingface_hub import HfApi, get_token

from lerobot.data_processing.artifacts import ArtifactStore
from lerobot.data_processing.bundles import write_bundle
from lerobot.data_processing.types import Resources

from .processing import (
    dispatch_bundle,
    follow_processing_job,
    require_persistent_remote,
)


def submit_annotate_to_hf(cfg):
    """Keep --job.target without argv replay, source uploads, or ephemeral-only results."""
    if cfg.runtime.backend not in {"local", "hf_jobs"}:
        raise ValueError("--job.target conflicts with the requested processing backend")
    if not cfg.repo_id:
        raise ValueError("Remote annotation requires --repo_id; a host-local --root is not uploaded")
    if not cfg.runtime.run_uri:
        raise ValueError("Remote annotation requires persistent runtime.run_uri")
    store = ArtifactStore(cfg.runtime.run_uri)
    require_persistent_remote(store)
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")
    remote = copy.deepcopy(cfg)
    remote.revision = HfApi(token=token).dataset_info(cfg.repo_id, revision=cfg.revision).sha
    remote.root = remote.staging_dir = None
    remote.runtime.backend = "local"
    remote.job.target = "local"
    config = draccus.encode(remote)
    code = cfg.runtime.hf_jobs.code_revision or cfg.job.lerobot_ref
    key, digest = write_bundle(store, "annotate", config, code)
    if cfg.runtime.mode == "plan":
        return key
    names = tuple(
        dict.fromkeys(
            (*cfg.runtime.hf_jobs.secret_env, *((cfg.vlm.api_key_env,) if cfg.vlm.api_key_env else ()))
        )
    )
    job = dispatch_bundle(
        store,
        key,
        digest,
        image=cfg.runtime.hf_jobs.gpu_image or cfg.job.image,
        flavor=cfg.job.target if cfg.job.is_remote else cfg.runtime.hf_jobs.gpu_flavor,
        code_revision=code,
        timeout=cfg.job.timeout,
        code_repository=cfg.runtime.hf_jobs.code_repository,
        namespace=cfg.runtime.hf_jobs.namespace,
        bootstrap_packages=cfg.runtime.hf_jobs.bootstrap_packages,
        secret_env=names,
        labels=dict.fromkeys(cfg.job.tags, "true"),
        resources=Resources(
            cpus=cfg.runtime.workers, gpus=max(1, cfg.vlm.num_gpus) if cfg.vlm.auto_serve else 0
        ),
    )
    print(f"Job submitted: {job.id}\nMonitor: hf jobs logs {job.id}\nPersistent outputs: {store.uri}")
    if follow_processing_job(job, detach=cfg.job.detach):
        print("Annotation complete; release files and index are retained in persistent storage.")
    return job
