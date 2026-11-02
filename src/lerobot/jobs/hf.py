# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a lerobot training on HF Jobs (HuggingFace GPUs).

Ported and simplified from lelab's runners/hf_cloud.py: no UI log queue, no
registry — just submit and stream to stdout.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import re
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from huggingface_hub import (
    HfApi,
    create_repo,
    fetch_job_logs,
    get_token,
    inspect_job,
    run_job,
    upload_file,
)

from lerobot.common.train_utils import (
    load_hub_training_metadata,
    load_training_metadata,
    push_checkpoint_to_hub,
)
from lerobot.configs import parser
from lerobot.configs.train import check_resolved_checkpoint_format, default_resume_output_dir
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP

from .dataset import count_dataset_episodes, ensure_dataset_available

if TYPE_CHECKING:
    from lerobot.configs.parallelism import ParallelismConfig
    from lerobot.configs.train import TrainPipelineConfig

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# HF Jobs names multi-GPU flavors `<gpu>x<count>` (`a10g-largex2`, `l4x4`, `h200x8`, `rtx-pro-6000x4`);
# single-GPU flavors carry no count or `x1` (`a10g-large`, `l4x1`, `h200`), and CPU flavors start with
# `cpu-`. Every accelerator in the catalogue is an Nvidia GPU (`hf jobs hardware` lists it).
_FLAVOR_GPU_COUNT_RE = re.compile(r"x(\d+)$")

_TERMINAL_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}

# huggingface_hub 1.x runs on httpx: transient HTTP/transport failures surface as
# httpx.HTTPError and socket-level errors as OSError. Catching only these keeps real
# bugs (TypeError, AttributeError, ...) from being silently retried or counted as
# job failures.
_TRANSIENT_NET_ERRORS = (OSError, httpx.HTTPError)

# Always attached to remote jobs and pushed datasets so LeRobot-originated work
# is identifiable on the Hub; callers (e.g. LeLab) add their own via --job.tags.
LEROBOT_TAG = "lerobot"


def resolve_job_tags(extra: list[str] | None) -> list[str]:
    """Return the tag list for a run: the lerobot tag plus any extras, deduped, order-stable."""
    return list(dict.fromkeys([LEROBOT_TAG, *(extra or [])]))


def flavor_gpu_count(flavor: str) -> int:
    """Return how many GPUs an HF Jobs flavor provides, read from its name (0 for CPU flavors)."""
    if flavor.startswith("cpu"):
        return 0
    match = _FLAVOR_GPU_COUNT_RE.search(flavor)
    return int(match.group(1)) if match else 1


def resolve_nproc_per_node(
    flavor: str,
    requested: int | None,
    *,
    recorded: int | None = None,
    resume: bool = False,
    streaming: bool = False,
) -> int:
    """Return how many training processes a job on `flavor` launches.

    A fresh run uses every GPU of the flavor unless `requested` (`--job.nproc_per_node`) says
    otherwise. A resume defaults to `recorded`, the data-parallel world size the checkpoint was
    trained with: changing it mid-run changes the effective batch (and the trainer refuses it for
    streaming datasets). When that size is unknown (`recorded=None`), a resume runs one process,
    as every resumed job did before multi-GPU launches existed.

    An explicit `requested` above the flavor's GPUs is allowed with a warning, since the GPU count
    is only read from the flavor name.

    Raises:
        ValueError: On a resume, if `recorded` exceeds the flavor's GPUs and no explicit value was
            given, or if an explicit value differs from `recorded` on a streaming dataset.
    """
    gpus = flavor_gpu_count(flavor)
    available = max(gpus, 1)
    if requested is None:
        if not resume:
            return available
        if recorded is None:
            if available > 1:
                logging.warning(
                    "This checkpoint doesn't record the data-parallel world size it was trained with; "
                    "resuming on 1 process. Pass --job.nproc_per_node=N to resume on N GPUs."
                )
            return 1
        if recorded > available:
            raise ValueError(
                f"The checkpoint was trained on {recorded} processes but flavor {flavor!r} has "
                f"{gpus} GPU(s) (counted from its name). Resume on a flavor with at least "
                f"{recorded} GPUs, or pass --job.nproc_per_node explicitly."
            )
        return recorded

    if requested > available:
        logging.warning(
            f"--job.nproc_per_node={requested} exceeds the {gpus} GPU(s) counted from the name of "
            f"flavor {flavor!r}. Launching {requested} processes anyway: the job crashes if the "
            "flavor really has fewer GPUs."
        )
    if resume and recorded is not None and requested != recorded:
        if streaming:
            raise ValueError(
                f"--job.nproc_per_node={requested} differs from the {recorded} data-parallel "
                "processes the checkpoint was trained with, and a streaming dataset only resumes "
                f"on the same data-parallel world size. Use --job.nproc_per_node={recorded}."
            )
        logging.warning(
            f"Resuming on {requested} processes a checkpoint trained on {recorded}: the effective "
            "batch size changes mid-run, and the data order is no longer sample-exact."
        )
    return requested


def checkpoint_dp_world_size(cfg: TrainPipelineConfig) -> int | None:
    """Return the data-parallel world size the checkpoint being resumed was trained with, or None.

    Reads the `training_step.json` the trainer records next to every checkpoint: from disk for a
    local `--config_path`, or (one small file) from the latest `checkpoints/<step>/` of a Hub repo.
    The checkpoint's train_config.json can't tell: a resolved `dp_replicate=1` looks the same as the
    unresolved default, and an interrupted job's repo root holds the staged, never-resolved config.
    Returns None only when the checkpoint doesn't record the size (older checkpoints); Hub errors
    (network, auth, a repo without checkpoints) propagate rather than silently resuming on 1 GPU.
    """
    config_path = parser.parse_arg("config_path")
    if not config_path:
        return None
    if Path(config_path).exists():  # the same local/Hub split as `_build_resume_job`
        if cfg.checkpoint_path is None:
            return None  # `_build_resume_job` reports it
        training_state_dir = Path(cfg.checkpoint_path) / TRAINING_STATE_DIR
        metadata = (
            load_training_metadata(training_state_dir)
            if (training_state_dir / TRAINING_STEP).is_file()
            else None
        )
    else:
        metadata = load_hub_training_metadata(config_path)
    if metadata is None:
        return None
    dp_world_size = metadata["dp_world_size"]
    return int(dp_world_size) if dp_world_size is not None else None


def fit_resumed_ddp_degree(parallelism: ParallelismConfig, nproc_per_node: int) -> list[str]:
    """Re-fit a resumed non-sharded run's `dp_replicate` to the launch; return the pod flags for it.

    A checkpoint's config records its resolved degrees, so a 2-process DDP run saves
    `dp_replicate=2`, which the pod would reject on any other world size. For plain DDP that degree
    is just the process count (every rank holds the full model and optimizer state), so it is safe
    to change on resume. A user's explicit `--parallelism.dp_replicate` is left alone, and sharded
    topologies are never touched: their degrees decide how DCP reshards the state.

    Args:
        parallelism (ParallelismConfig): The resumed run's (unresolved) degrees, updated in place.
        nproc_per_node (int): The processes the job launches.

    Returns:
        list[str]: The `--parallelism.dp_replicate=N` flag the pod needs, or nothing.
    """
    if (
        parallelism.is_sharded
        or parallelism.dp_replicate in (1, nproc_per_node)
        or parser.parse_arg("parallelism.dp_replicate") is not None
    ):
        return []
    parallelism.dp_replicate = nproc_per_node
    return [f"--parallelism.dp_replicate={nproc_per_node}"]


def describe_launch(
    nproc_per_node: int,
    flavor: str,
    *,
    requested: int | None,
    recorded: int | None,
    resume: bool,
) -> str | None:
    """Spell out how many processes the job launches and why; None for a plain single-GPU flavor."""
    gpus = flavor_gpu_count(flavor)
    if requested is not None:
        reason = "set by --job.nproc_per_node"
    elif resume:
        reason = (
            f"checkpoint was trained on {recorded}"
            if recorded is not None
            else "checkpoint doesn't record its process count"
        )
    else:
        reason = "one per GPU"
    gpu_count = f"{gpus} GPU{'' if gpus == 1 else 's'}"
    if nproc_per_node > 1:
        return f"torchrun, {nproc_per_node} processes ({reason}; flavor has {gpu_count})"
    if gpus > 1:
        return f"single process, uses 1 of the flavor's {gpu_count} ({reason})"
    return None


def build_launch(train_args: list[str], nproc_per_node: int) -> tuple[list[str], dict[str, str] | None]:
    """Return the pod command and the environment it needs.

    A single process keeps the plain `lerobot-train <args>` command and the image's environment.
    Several processes run under `torchrun`: the trainer only goes multi-process when torchrun
    launches it (it reads `WORLD_SIZE`), and `--standalone` sets up a single-node rendezvous on a
    free local port. The `lerobot-gpu` image bakes `CUDA_VISIBLE_DEVICES=0` in
    (docker/Dockerfile.internal, for single-GPU CI), which would hide all but one GPU from every
    rank, so the ranks' GPUs are exposed explicitly; the container runtime numbers a pod's GPUs
    from 0.
    """
    if nproc_per_node == 1:
        return ["lerobot-train", *train_args], None
    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={nproc_per_node}",
        "-m",
        "lerobot.scripts.lerobot_train",
        *train_args,
    ]
    return command, {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(nproc_per_node))}


def describe_batch_size(cfg: TrainPipelineConfig, parallelism: ParallelismConfig) -> str:
    """Spell out the effective batch size of a remote run from its resolved topology.

    Every data-parallel worker loads its own `--batch_size` micro-batch per step.
    """
    dp_world_size = parallelism.dp_world_size
    grad_accum = cfg.accelerator.gradient_accumulation.steps
    effective = cfg.batch_size * dp_world_size * grad_accum
    return (
        f"{cfg.batch_size} per worker x {dp_world_size} data-parallel worker(s) x {grad_accum} "
        f"grad accum = {effective} samples per optimizer step"
    )


def build_repo_id(username: str, job_name: str, now: dt.datetime) -> str:
    """Generate the model repo id for a remote run: <user>/<job_name>_<timestamp>."""
    slug = _SLUG_RE.sub("-", job_name).strip("-") or "train"
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{username}/{slug}_{stamp}"


def build_remote_config_file(cfg, repo_id: str, dest: Path, tags: list[str] | None = None) -> Path:
    """Write a train_config.json for the pod, with remote overrides applied.

    The pod runs `lerobot-train --config_path=<dest>` and downloads the dataset
    by repo_id into its own cache. Client-only fields are stripped so the config
    is accepted by the trainer image: `job` (pure client orchestration) is always
    removed, and `save_checkpoint_to_hub` is removed unless explicitly enabled —
    older lerobot images reject unknown keys, so the default keeps the config
    compatible with the released `lerobot-gpu` image. `tags` are merged into
    policy.tags so the trained model the pod pushes carries them too.
    """
    remote = copy.deepcopy(cfg)
    remote.policy.push_to_hub = True
    remote.policy.repo_id = repo_id
    # Don't pin the client's resolved device (e.g. "mps"); let the pod auto-detect its GPU.
    remote.policy.device = None
    # Drop any host-local dataset root; the pod resolves the dataset by repo_id.
    remote.dataset.root = None
    if tags:
        existing = list(remote.policy.tags or [])
        remote.policy.tags = existing + [t for t in tags if t not in existing]

    # Encode to the canonical, pod-parseable dict, then drop the keys the released
    # trainer image doesn't know about.
    data = remote.to_dict()
    data.pop("job", None)
    # `policy.dtype` is newer than the released trainer image. Omit it only when both the value and the
    # policy class default are None: the pod then decodes the missing key back to the same None.
    policy = data.get("policy")
    if (
        isinstance(policy, dict)
        and policy.get("dtype") is None
        and getattr(type(remote.policy), "dtype", None) is None
    ):
        policy.pop("dtype", None)
    if not remote.save_checkpoint_to_hub:
        data.pop("save_checkpoint_to_hub", None)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=4))
    return dest


def _stage_config_on_hub(cfg, repo_id: str, token: str, tags: list[str] | None = None) -> str:
    """Upload train_config.json to the model repo and return the repo_id for --config_path."""
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = build_remote_config_file(cfg, repo_id, Path(tmp) / "train_config.json", tags=tags)
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo="train_config.json",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    return repo_id


def _tail_logs(
    job_id: str,
    done: threading.Event,
    success_marker: str | None = None,
    success_event: threading.Event | None = None,
) -> None:
    """Stream job logs to stdout, reconnecting on dropped streams until done is set.

    Each reconnect re-fetches the full buffered log, so we track how many lines
    were already printed and skip them — otherwise a fast-failing job's traceback
    gets reprinted on every reconnect.

    When `success_marker` appears in a line, set `success_event` and `done` so the
    caller can finish as soon as the trained model lands on the Hub, rather than
    waiting out the platform's post-run finalization (which can add ~30s).
    """
    printed = 0
    while not done.is_set():
        try:
            seen = 0
            for line in fetch_job_logs(job_id=job_id, follow=True):
                seen += 1
                if seen <= printed:
                    continue  # already shown on a previous connection
                printed = seen
                # fetch_job_logs yields SSE data without trailing newlines, so add one
                # per entry — otherwise all log lines concatenate onto a single line.
                print(line.rstrip("\n"), flush=True)
                if success_marker and success_event is not None and success_marker in line:
                    success_event.set()
                    done.set()
                    return
                if done.is_set():
                    return
            # Stream closed cleanly. Wait a moment so the status poller can mark
            # the job terminal before we reconnect (avoids re-tailing the buffer).
            if done.wait(3):
                return
        except _TRANSIENT_NET_ERRORS:
            if done.wait(2):
                return


def _poll_until_done(
    job_id: str,
    done: threading.Event,
    poll_interval: float = 5.0,
    status_holder: dict | None = None,
    max_failures: int = 6,
) -> str | None:
    """Poll inspect_job until a terminal stage or until `done` is set.

    Returns the terminal stage string, or None if `done` was set first (detach)
    or after `max_failures` consecutive inspect_job errors. When a terminal stage
    is reached and `status_holder` is given, records `status_holder["message"]`
    (the platform's status message, e.g. "Job timeout").
    """
    failures = 0
    while not done.is_set():
        try:
            info = inspect_job(job_id=job_id)
            failures = 0
            # `stage` is an enum in some huggingface_hub versions and a plain str in others.
            stage = getattr(info.status.stage, "value", info.status.stage)
            if stage in _TERMINAL_STAGES:
                if status_holder is not None:
                    status_holder["message"] = getattr(info.status, "message", None)
                done.set()
                return stage
        except _TRANSIENT_NET_ERRORS:
            failures += 1
            if failures >= max_failures:
                done.set()
                return None
        done.wait(poll_interval)
    return None


def follow_job(job_id: str, *, detach: bool = False, success_marker: str | None = None) -> bool:
    """Watch a submitted job to the end, streaming its logs to stdout.

    Returns True when the job finished successfully and False when we stopped watching
    without a verdict — `detach`, or the user pressing Ctrl-C, which detaches rather than
    cancelling the remote job. Raises RuntimeError when the job reaches a terminal stage
    other than COMPLETED.

    `success_marker` finishes as soon as that string appears in the logs instead of waiting
    out the platform's post-run finalization (~30s). Callers that have a log line meaning
    "the artifact is on the Hub" should pass it; without one, completion is stage-based.
    """
    if detach:
        return False

    done = threading.Event()
    detached = threading.Event()
    marker_seen = threading.Event()
    stage_holder: dict[str, str | None] = {}

    def _poll() -> None:
        stage_holder["stage"] = _poll_until_done(job_id, done, status_holder=stage_holder)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    log_thread = threading.Thread(
        target=_tail_logs, args=(job_id, done, success_marker, marker_seen), daemon=True
    )
    log_thread.start()

    def _detach(sig, frame):
        detached.set()
        done.set()
        print("\nDetached. Job is still running.")
        print(f"  Monitor: hf jobs logs {job_id}")
        print(f"  Cancel:  hf jobs cancel {job_id}")

    # signal.signal only works on the main thread; when called from a worker thread
    # (e.g. an orchestration framework) skip the Ctrl-C-detaches-instead-of-cancels
    # handler rather than crashing with ValueError.
    install_sigint = threading.current_thread() is threading.main_thread()
    original_sigint = signal.getsignal(signal.SIGINT) if install_sigint else None
    if install_sigint:
        signal.signal(signal.SIGINT, _detach)
    try:
        # Timeout-based join so SIGINT is delivered to the main thread promptly.
        while poll_thread.is_alive():
            poll_thread.join(timeout=0.5)
        log_thread.join(timeout=5)
    finally:
        if install_sigint:
            signal.signal(signal.SIGINT, original_sigint)

    if detached.is_set():
        return False
    if marker_seen.is_set():
        return True

    stage = stage_holder.get("stage")
    if stage != "COMPLETED":
        message = stage_holder.get("message")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"Job {job_id} ended with stage={stage}{detail}. Check logs: hf jobs logs {job_id}"
        )
    return True


def _pod_forwarded_args(
    argv: list[str], drop_names: tuple[str, ...] = (), drop_prefixes: tuple[str, ...] = ()
) -> list[str]:
    """User CLI overrides to replay on the pod, minus flags the submitter sets itself.

    Handles both `--name=value` and `--name value` forms. Forwarding the user's overrides (e.g.
    `--steps`, `--save_checkpoint_to_hub`) makes a remote resume behave like the same local command.
    """
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        name = tok.split("=", 1)[0]
        if name in drop_names or any(name.startswith(p) for p in drop_prefixes):
            if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                skip_next = True  # also drop the space-separated value
            continue
        out.append(tok)
    return out


def _build_resume_job(cfg: TrainPipelineConfig, username: str) -> tuple[str, list[str]]:
    """Resolve the model repo and the `lerobot-train` args to resume a run on a job.

    A Hub `config_path` is resumed from directly: its checkpoint config already targets that repo,
    so new checkpoints continue the lineage there. A local `config_path` has its checkpoint uploaded
    to a new PRIVATE repo first, and the resumed run is forced to push back to it. The args always
    carry `--job.target=local` so the checkpoint's saved `job.target` can't make the pod re-dispatch
    itself. The caller wraps them into the pod command (see `build_launch`).
    """
    config_path = parser.parse_arg("config_path")
    if not config_path:
        # validate() only enforces this on its own resume path; keep the helper self-contained.
        raise ValueError("A config_path is expected when resuming a run.")
    forwarded = _pod_forwarded_args(
        sys.argv[1:],
        drop_names=("--config_path", "--policy.repo_id", "--policy.push_to_hub", "--dataset.root"),
        drop_prefixes=("--job.",),
    )

    if Path(config_path).exists():
        # Local checkpoint: stage it on the Hub so the pod can resume from it, and push back there.
        if cfg.checkpoint_path is None:
            raise ValueError(
                f"Cannot resume from local checkpoint {config_path!r}: --policy.path / "
                "--reward_model.path take priority over --resume, so no checkpoint was resolved. "
                "Drop them to resume from --config_path."
            )
        # Resolve so a `last` symlink uploads under its real step name (digit), which the pod's
        # latest-checkpoint lookup keys on.
        checkpoint_dir = Path(cfg.checkpoint_path).resolve()
        source_repo = build_repo_id(username, cfg.job_name or "train", dt.datetime.now(dt.UTC))
        push_checkpoint_to_hub(checkpoint_dir, source_repo, private=True)
        extra = [f"--policy.repo_id={source_repo}", "--policy.push_to_hub=true"]
    else:
        source_repo = config_path
        extra = []

    train_args = [*forwarded, f"--config_path={source_repo}", "--job.target=local", *extra]
    return source_repo, train_args


def submit_to_hf(cfg: TrainPipelineConfig) -> None:
    """Submit a training job to HF Jobs infrastructure.

    Validates cfg, resolves credentials, ensures the dataset is on the Hub, then either stages a
    sanitized config (fresh run) or resumes from a checkpoint repo, submits the job, and tails logs
    until completion or detaches immediately. Ctrl-C detaches without cancelling the remote job.
    On a multi-GPU flavor the pod runs the trainer under `torchrun`, one process per GPU (see
    `resolve_nproc_per_node`); a resume keeps the checkpoint's data-parallel world size.
    """
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    api = HfApi(token=token)
    user_info = api.whoami(token=token)
    username = user_info["name"]

    now = dt.datetime.now(dt.UTC)
    fresh_repo_id: str | None = None
    if not cfg.resume:
        # Resolve the model repo and mark it for push BEFORE validate(): validate() requires repo_id
        # to be set whenever push_to_hub is True. (A resume reuses the checkpoint's repo instead.)
        if cfg.policy is not None:
            base_name = cfg.job_name or cfg.policy.type
            fresh_repo_id = cfg.policy.repo_id or build_repo_id(username, base_name, now)
            cfg.policy.repo_id = fresh_repo_id
            cfg.policy.push_to_hub = True
        else:
            # Path-based policy is resolved inside validate(); fall back to a generic slug.
            fresh_repo_id = build_repo_id(username, cfg.job_name or "train", now)

    cfg.validate()

    if cfg.is_reward_model_training:
        raise ValueError(
            "Remote training via --job.target only supports policy training, not reward models. "
            "Run reward-model training locally."
        )

    flavor = cfg.job.target
    if flavor is None:
        raise ValueError("submit_to_hf needs a remote --job.target (an HF Jobs flavor).")
    # Size the launch before anything is pushed to the Hub, so a bad topology fails for free.
    # Training rejects context parallelism, so a checkpoint's dp world size is its process count.
    recorded = checkpoint_dp_world_size(cfg) if cfg.resume else None
    nproc_per_node = resolve_nproc_per_node(
        flavor,
        cfg.job.nproc_per_node,
        recorded=recorded,
        resume=cfg.resume,
        streaming=cfg.dataset.streaming,
    )
    # The pod resolves the same degrees against torchrun's WORLD_SIZE; check them here instead.
    parallelism = copy.deepcopy(cfg.parallelism)
    resume_overrides = fit_resumed_ddp_degree(parallelism, nproc_per_node) if cfg.resume else []
    try:
        parallelism.resolve(nproc_per_node)
        check_resolved_checkpoint_format(cfg.checkpoint_format, parallelism)
    except ValueError as e:
        if cfg.resume and cfg.parallelism.is_sharded:
            hint = (
                f"A sharded checkpoint resumes on {nproc_per_node} process(es) only with "
                f"--parallelism.* degrees that multiply to {nproc_per_node} (e.g. "
                "--parallelism.dp_shard=-1; DCP reshards the state), or on its original size "
                "with --job.nproc_per_node."
            )
        else:
            hint = "Adjust --parallelism.*, --checkpoint_format, --job.nproc_per_node or --job.target."
        raise ValueError(
            f"The run does not fit {nproc_per_node} process(es) on flavor {flavor!r}: {e} {hint}"
        ) from e
    if cfg.dataset.streaming and parallelism.dp_world_size > 1:
        # The trainer splits a streaming dataset's episodes across the data-parallel ranks and
        # fails on a rank left without one; catch that here from the dataset's info.json alone.
        episodes = count_dataset_episodes(cfg.dataset, api=api)
        if episodes is not None and episodes < parallelism.dp_world_size:
            raise ValueError(
                f"Streaming dataset {cfg.dataset.repo_id!r} has {episodes} training episode(s), "
                f"fewer than the {parallelism.dp_world_size} data-parallel processes it is split "
                f"across (each needs at least one). Launch at most {episodes} process(es) with "
                "--job.nproc_per_node, or use a map-style dataset (--dataset.streaming=false)."
            )

    secrets: dict[str, str] = {"HF_TOKEN": token}
    if cfg.trackio.enable and not cfg.trackio.space_id:
        raise ValueError(
            "trackio is enabled but no `--trackio.space_id` is set. A remote job's local Trackio database "
            "vanishes with the pod, so pass a Hugging Face Space, e.g. `--trackio.space_id=<user>/<space>`."
        )

    tags = resolve_job_tags(cfg.job.tags)
    # The dataset must be reachable from the pod for both fresh and resumed runs; a local-only
    # dataset is pushed PRIVATE here. Hoisted before the resume/fresh branch since it applies to both.
    ensure_dataset_available(cfg.dataset.repo_id, api=api, tags=tags)

    if fresh_repo_id is None:
        # Resuming (fresh_repo_id is only set for fresh runs): the model repo comes from the checkpoint.
        repo_id, train_args = _build_resume_job(cfg, username)
        train_args += resume_overrides
        if nproc_per_node > 1 and parser.parse_arg("output_dir") is None:
            # Each rank would otherwise stamp its own resume dir from its own clock; pin one dir
            # for all of them.
            train_args.append(f"--output_dir={default_resume_output_dir(now)}")
    else:
        config_repo_id = _stage_config_on_hub(cfg, fresh_repo_id, token, tags=tags)
        repo_id = fresh_repo_id
        train_args = [f"--config_path={config_repo_id}"]
    command, env = build_launch(train_args, nproc_per_node)

    print(f"Submitting job to HF Jobs (flavor={flavor}, image={cfg.job.image}) ...")
    launch = describe_launch(
        nproc_per_node, flavor, requested=cfg.job.nproc_per_node, recorded=recorded, resume=cfg.resume
    )
    if launch is not None:
        print(f"  Launch:     {launch}")
    if nproc_per_node > 1:
        print(
            "              --steps counts per-worker micro-batches; the learning rate and steps "
            "are not rescaled for the larger batch."
        )
    print(f"  Batch size: {describe_batch_size(cfg, parallelism)}")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        env=env,
        flavor=flavor,
        secrets=secrets,
        timeout=cfg.job.timeout,
        # HF Jobs labels are key/value; expose each tag as a queryable label.
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:   {job_url}")
    print(f"  Model repo: https://huggingface.co/{repo_id}")
    print(f"  Monitor:    hf jobs logs {job_id}")
    print(f"  Cancel:     hf jobs cancel {job_id}")

    # Finish as soon as the model is pushed, rather than waiting out the platform's
    # post-run finalization before the job stage flips to COMPLETED. This matches the
    # exact log line emitted by lerobot.common.train_utils.publish_trained_model — the two must stay
    # in sync. If it ever stops matching we just fall back to stage-based completion
    # (~30s slower), so the contract is an optimization, not a correctness requirement.
    success_marker = f"Model pushed to https://huggingface.co/{repo_id}"
    if follow_job(job_id, detach=cfg.job.detach, success_marker=success_marker):
        print(f"\nTraining complete — model pushed to https://huggingface.co/{repo_id}")
