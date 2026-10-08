# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Serialized processing bundles and bounded HF Jobs worker groups.

Images must provide Python 3.12+, git, ffmpeg and source/model dependencies.
Only explicitly trusted, installed Python factories may execute. No argv replay.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import draccus
from huggingface_hub import get_token, inspect_job, list_jobs_hardware, run_job

from lerobot.data_processing.artifacts import ArtifactStore
from lerobot.data_processing.bundles import write_bundle
from lerobot.data_processing.planner import StagePlan, load_module
from lerobot.data_processing.runtime import finalize_stage
from lerobot.data_processing.types import Artifact, Resources, fingerprint
from lerobot.data_processing.worker import accepted_in_shard, run_worker_group

from .processing_worker import assigned_shards


def require_persistent_remote(store):
    protocols = store.fs.protocol if isinstance(store.fs.protocol, tuple) else (store.fs.protocol,)
    if store.is_local or "memory" in protocols:
        raise ValueError("HF Jobs requires a persistent remote runtime.run_uri, not local/memory storage")


def build_pod_command(run_uri, bundle_key, digest, code_revision):
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision or ""):
        raise ValueError("HF Jobs code_revision must be an immutable 40-character Git commit SHA")
    # The pinned image owns model/source dependencies, and the pinned code owns
    # LeRobot. --no-deps avoids replacing an image's deliberate CUDA/vLLM pins.
    spec = f"lerobot @ git+https://github.com/huggingface/lerobot.git@{code_revision}"
    install = shlex.join(["python", "-m", "pip", "install", "--no-deps", spec])
    driver = shlex.join(
        [
            "python",
            "-m",
            "lerobot.jobs.processing",
            "--run-uri",
            run_uri,
            "--bundle-key",
            bundle_key,
            "--sha256",
            digest,
        ]
    )
    return [
        "sh",
        "-c",
        f"{install} && export LEROBOT_PROCESSING_CODE_REVISION={code_revision} && exec {driver}",
    ]


def _secrets(names):
    secrets = {"HF_TOKEN": get_token()}
    if not secrets["HF_TOKEN"]:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not os.environ.get(name):
            raise ValueError(f"Missing or invalid secret environment reference: {name}")
        secrets[name] = os.environ[name]
    return secrets


def validate_hardware(flavor, resources):
    hardware = next((item for item in list_jobs_hardware() if item.name == flavor), None)
    if hardware is None:
        raise ValueError(f"Unknown HF Jobs flavor: {flavor}")
    cpu = re.fullmatch(r"([0-9]+) vCPU", hardware.cpu)
    memory = re.fullmatch(r"([0-9.]+) GB", hardware.ram)
    if not cpu or not memory:
        raise ValueError("Unsupported hardware metadata units; cannot validate resource admission")
    gpus = (
        int(hardware.accelerator.quantity)
        if hardware.accelerator and hardware.accelerator.type == "gpu"
        else 0
    )
    if resources.cpus > int(cpu[1]) or resources.memory_gb > float(memory[1]) or resources.gpus > gpus:
        raise ValueError(f"Requested CPU/GPU/memory exceeds HF flavor {flavor}")


def dispatch_bundle(
    store,
    key,
    digest,
    *,
    image,
    flavor,
    code_revision,
    timeout,
    secret_env=(),
    labels=None,
    resources=Resources(),
):
    require_persistent_remote(store)
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image or ""):
        raise ValueError("Choose an immutable image@sha256:digest before HF Jobs dispatch")
    if not flavor or flavor == "local":
        raise ValueError("An HF Jobs hardware flavor is required")
    command = build_pod_command(store.uri, key, digest, code_revision)
    validate_hardware(flavor, resources)
    # Refuse to duplicate an active job for this exact bundle on resume.
    for path in store.list(f"submissions/hf/{digest}/*.json"):
        previous = store.read_json(path)
        info = inspect_job(previous["job_id"])
        stage = getattr(info.status.stage, "value", info.status.stage)
        if stage not in {"COMPLETED", "CANCELED", "ERROR", "DELETED"}:
            raise RuntimeError(f"HF Job {info.id} is still active; wait or cancel it before resuming")
    job = run_job(
        image=image,
        command=command,
        flavor=flavor,
        timeout=timeout,
        secrets=_secrets(secret_env),
        labels={"lerobot": "true", **(labels or {})},
        env={"OMP_NUM_THREADS": str(resources.cpus), "MKL_NUM_THREADS": str(resources.cpus)},
    )
    try:
        store.put_json(f"submissions/hf/{digest}/{uuid.uuid4().hex}.json", {"job_id": job.id, "bundle": key})
    except Exception as exc:
        raise RuntimeError(
            f"HF Job {job.id} was submitted but could not be recorded; inspect it explicitly"
        ) from exc
    return job


def follow_processing_job(job, *, detach=False):
    # Reuse established log/retry/cancellation semantics without importing the
    # training dependency tree on CPU workers or detached launchers.
    if detach:
        return False
    from .hf import follow_job

    return follow_job(job.id, detach=False)


def run_hf_stage(store, plan, runtime):
    """A controller waits for bounded groups, then verifies actual durable outputs.

    Payloads/factories must address remote-readable inputs. Local source paths are
    not automatically uploaded. Annotation/conversion compatibility jobs instead
    use a single-node controller and its local persistent worker pool.
    """
    require_persistent_remote(store)
    if all(
        len(accepted_in_shard(store, plan, s)) == len(plan.read_shard(store, s)) for s in range(plan.shards)
    ):
        return finalize_stage(store, plan)
    cfg = runtime.hf_jobs
    resources = load_module(plan.factory, plan.config).spec.resources
    image = cfg.gpu_image if resources.gpus else cfg.cpu_image
    flavor = cfg.gpu_flavor if resources.gpus else cfg.cpu_flavor
    groups = min(runtime.workers, plan.shards)
    for start in range(0, groups, cfg.max_parallel):
        jobs = []
        for index in range(start, min(groups, start + cfg.max_parallel)):
            shards = assigned_shards(plan.shards, groups, index)
            if all(len(accepted_in_shard(store, plan, s)) == len(plan.read_shard(store, s)) for s in shards):
                continue
            key, digest = write_bundle(
                store,
                "worker",
                {
                    "plan_id": plan.plan_id,
                    "shards": shards,
                    "batch_size": runtime.batch_size,
                    "max_retries": runtime.max_retries,
                },
                cfg.code_revision,
            )
            jobs.append(
                dispatch_bundle(
                    store,
                    key,
                    digest,
                    image=image,
                    flavor=flavor,
                    code_revision=cfg.code_revision,
                    timeout=cfg.timeout,
                    secret_env=cfg.secret_env,
                    resources=resources,
                )
            )
        for job in jobs:
            follow_processing_job(job)
    return finalize_stage(store, plan)


def submit_convert_to_hf(cfg):
    """Single-node CPU conversion from a pinned remotely-acquirable source archive.

    Source factories must already be in the pinned code/image. This initial
    launcher deliberately rejects local-only sources instead of uploading them.
    """
    if not cfg.runtime.run_uri:
        raise ValueError("Remote conversion requires persistent runtime.run_uri")
    store = ArtifactStore(cfg.runtime.run_uri)
    require_persistent_remote(store)
    if (
        not cfg.source.get("archive_uri")
        or not cfg.source.get("archive_sha256")
        or not cfg.source.get("manifest")
    ):
        raise ValueError("Remote conversion requires a pinned archive_uri/archive_sha256 and manifest recipe")
    raw = ArtifactStore(cfg.source["archive_uri"])
    require_persistent_remote(raw)
    remote = copy.deepcopy(cfg)
    identifier = fingerprint({"factory": cfg.source_factory, "source": cfg.source})
    remote.source["manifest"] = f"inputs/{identifier}/raw/{Path(cfg.source['manifest']).name}"
    remote.output = Path(f"outputs/{fingerprint(draccus.encode(cfg))}/converted")
    remote.runtime.backend = "local"
    jobs = cfg.runtime.hf_jobs
    key, digest = write_bundle(store, "convert", draccus.encode(remote), jobs.code_revision)
    if cfg.runtime.mode == "plan":
        return key
    job = dispatch_bundle(
        store,
        key,
        digest,
        image=jobs.cpu_image,
        flavor=jobs.cpu_flavor,
        code_revision=jobs.code_revision,
        timeout=jobs.timeout,
        secret_env=jobs.secret_env,
        resources=Resources(cpus=cfg.runtime.workers * cfg.encoder_threads),
    )
    follow_processing_job(job, detach=jobs.detach)
    return job


def retain_release(store, root: Path, *, source=None, config=None):
    """Keep a complete release index on persistent storage before a pod disappears.

    All frame/annotation files are Parquet. JSON here is control/provenance, not
    annotation storage. Videos are content-addressed once and reused on retries.
    """
    root = root.resolve()
    files, source_files = [], []
    paths = [
        (f"{name}/{path.relative_to((root / name).resolve()).as_posix()}", path)
        for name in ("data", "meta", "videos")
        for path in (root / name).resolve().rglob("*")
        if path.is_file()
    ]
    for name in ("README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"):
        if (root / name).is_file():
            paths.append((name, root / name))
    for relative, path in sorted(paths):
        from lerobot.data_processing.artifacts import file_checksum

        digest, size = file_checksum(path)
        if (
            relative.startswith("videos/")
            and source
            and source.get("repo_id")
            and re.fullmatch(r"[0-9a-f]{40}", source.get("revision") or "")
        ):
            source_files.append({"path": relative, "sha256": digest, "size": size})
            continue  # pinned original videos remain on the Hub, not copied again
        key = f"release_files/{digest}/{path.name}"
        if store.exists(key):
            if store.checksum(key) != (digest, size):
                raise ValueError("Persistent release file checksum mismatch")
            artifact = Artifact(key, digest, size)
        else:
            artifact = store.put_file(key, path)
        files.append({"path": relative, "artifact": asdict(artifact)})
    release = {"version": 1, "source": source, "config": config, "files": files, "source_files": source_files}
    key = f"releases/{fingerprint(release)}.json"
    if not store.exists(key):
        store.put_json(key, release)
    return key


def restore_release(store, key, destination: Path):
    """Materialize a retained ordinary LeRobot dataset without running annotation."""
    if destination.exists():
        raise FileExistsError(destination)
    if store.checksum(key)[0] != Path(key).stem:
        raise ValueError("Retained release index checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lerobot-restore-", dir=destination.parent) as directory:
        temporary = Path(directory) / "dataset"
        _restore_release_into(store, key, temporary)
        temporary.rename(destination)
    return destination


def _restore_release_into(store, key, destination):
    release = store.read_json(key)
    for entry in release["files"]:
        relative = entry["path"]
        ArtifactStore(destination).path(relative)  # validate before writing
        artifact = Artifact(**entry["artifact"])
        if not store.verify(artifact):
            raise ValueError("Retained release checksum mismatch")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with store.open(artifact.path) as source, target.open("wb") as output:
            import shutil

            shutil.copyfileobj(source, output)
    if release.get("source_files"):
        import shutil

        from huggingface_hub import hf_hub_download

        from lerobot.data_processing.artifacts import file_checksum

        for entry in release["source_files"]:
            ArtifactStore(destination).path(entry["path"])
            source = release["source"]
            path = Path(
                hf_hub_download(
                    source["repo_id"], entry["path"], repo_type="dataset", revision=source["revision"]
                )
            )
            if file_checksum(path) != (entry["sha256"], entry["size"]):
                raise ValueError("Pinned source video checksum mismatch")
            target = destination / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


def execute_bundle(store, key, digest):
    with store.open(key) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Processing bundle checksum mismatch")
    bundle = json.loads(data)
    if bundle["version"] != 1:
        raise ValueError("Unsupported processing bundle version")
    if os.environ.get("LEROBOT_PROCESSING_CODE_REVISION") != bundle["code_revision"]:
        raise ValueError("The running code does not match the bundle's pinned revision")
    config = copy.deepcopy(bundle["config"])
    action = bundle["action"]
    if action == "worker":
        plan = StagePlan.load(store, config["plan_id"], verify_work=False)
        run_worker_group(
            store.uri, plan.plan_id, config["shards"], config["batch_size"], config["max_retries"]
        )
        return None
    if action == "annotate":
        from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
        from lerobot.scripts.lerobot_annotate import _resolve_root, _run_annotation

        cfg = draccus.decode(AnnotationPipelineConfig, config)
        root = _resolve_root(cfg)
        summary = _run_annotation(cfg, root)
        source = {"repo_id": cfg.repo_id, "revision": cfg.revision}
    elif action == "convert":
        from lerobot.data_processing.conversion import ConvertConfig, convert_dataset

        cfg = draccus.decode(ConvertConfig, config)
        from lerobot.utils.constants import HF_LEROBOT_HOME

        scratch = HF_LEROBOT_HOME / "processing"
        cfg.source["manifest"] = str(scratch / cfg.source["manifest"])
        cfg.output = scratch / cfg.output
        convert_dataset(cfg)
        root, source = cfg.output, cfg.source
    else:
        raise ValueError(f"Unknown processing bundle action: {action}")
    release = retain_release(store, root, source=source, config=config)
    print(f"Persistent release: {store.uri}/{release}")
    if action == "annotate" and cfg.push_to_hub and cfg.runtime.mode != "plan":
        from lerobot.scripts.lerobot_annotate import _changed_paths, _push_to_hub

        _push_to_hub(root, cfg, changed_paths=_changed_paths(root, summary))
    return release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-uri", required=True)
    parser.add_argument("--bundle-key", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    execute_bundle(ArtifactStore(args.run_uri), args.bundle_key, args.sha256)


if __name__ == "__main__":
    main()
