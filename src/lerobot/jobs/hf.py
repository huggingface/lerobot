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
import netrc
import os
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

from lerobot.common.train_utils import push_checkpoint_to_hub
from lerobot.configs import parser

from .dataset import ensure_dataset_available

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

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
    tags = [LEROBOT_TAG, *(extra or [])]
    seen: set[str] = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def resolve_wandb_api_key() -> str | None:
    """Host's wandb key for forwarding to the job: $WANDB_API_KEY, else ~/.netrc."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        rc = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = rc.authenticators("api.wandb.ai")
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


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
    """Resolve the model repo and pod command to resume a run on a job.

    A Hub `config_path` is resumed from directly: its checkpoint config already targets that repo,
    so new checkpoints continue the lineage there. A local `config_path` has its checkpoint uploaded
    to a new PRIVATE repo first, and the resumed run is forced to push back to it. The pod command
    always carries `--job.target=local` so the checkpoint's saved `job.target` can't make the pod
    re-dispatch itself.
    """
    config_path = parser.parse_arg("config_path")
    forwarded = _pod_forwarded_args(
        sys.argv[1:],
        drop_names=("--config_path", "--policy.repo_id", "--policy.push_to_hub", "--dataset.root"),
        drop_prefixes=("--job.",),
    )

    if Path(config_path).exists():
        # Local checkpoint: stage it on the Hub so the pod can resume from it, and push back there.
        # Resolve so a `last` symlink uploads under its real step name (digit), which the pod's
        # latest-checkpoint lookup keys on.
        checkpoint_dir = Path(cfg.checkpoint_path).resolve()
        source_repo = build_repo_id(username, cfg.job_name or "train", dt.datetime.now(dt.UTC))
        push_checkpoint_to_hub(checkpoint_dir, source_repo, private=True)
        extra = [f"--policy.repo_id={source_repo}", "--policy.push_to_hub=true"]
    else:
        source_repo = config_path
        extra = []

    command = [
        "lerobot-train",
        *forwarded,
        f"--config_path={source_repo}",
        "--job.target=local",
        *extra,
    ]
    return source_repo, command


def submit_to_hf(cfg: TrainPipelineConfig) -> None:
    """Submit a training job to HF Jobs infrastructure.

    Validates cfg, resolves credentials, ensures the dataset is on the Hub, then either stages a
    sanitized config (fresh run) or resumes from a checkpoint repo, submits the job, and tails logs
    until completion or detaches immediately. Ctrl-C detaches without cancelling the remote job.
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

    secrets: dict[str, str] = {"HF_TOKEN": token}
    if cfg.wandb.enable:
        wandb_key = resolve_wandb_api_key()
        if wandb_key is None:
            raise ValueError(
                "wandb is enabled but no WANDB_API_KEY found. "
                "Set it via `export WANDB_API_KEY=...` or add it to ~/.netrc."
            )
        secrets["WANDB_API_KEY"] = wandb_key

    tags = resolve_job_tags(cfg.job.tags)
    # The dataset must be reachable from the pod for both fresh and resumed runs; a local-only
    # dataset is pushed PRIVATE here. Hoisted before the resume/fresh branch since it applies to both.
    ensure_dataset_available(cfg.dataset.repo_id, api=api, tags=tags)

    if cfg.resume:
        repo_id, command = _build_resume_job(cfg, username)
    else:
        config_repo_id = _stage_config_on_hub(cfg, fresh_repo_id, token, tags=tags)
        repo_id = fresh_repo_id
        command = ["lerobot-train", f"--config_path={config_repo_id}"]

    print(f"Submitting job to HF Jobs (flavor={cfg.job.target}, image={cfg.job.image}) ...")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
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

    if cfg.job.detach:
        return

    done = threading.Event()
    detached = threading.Event()
    pushed_ok = threading.Event()
    stage_holder: dict[str, str | None] = {}

    def _poll() -> None:
        stage_holder["stage"] = _poll_until_done(job_id, done, status_holder=stage_holder)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    # Finish as soon as the model is pushed, rather than waiting out the platform's
    # post-run finalization before the job stage flips to COMPLETED. This matches the
    # exact log line emitted by PreTrainedPolicy.push_model_to_hub — the two must stay
    # in sync. If it ever stops matching we just fall back to stage-based completion
    # (~30s slower), so the contract is an optimization, not a correctness requirement.
    success_marker = f"Model pushed to https://huggingface.co/{repo_id}"
    log_thread = threading.Thread(
        target=_tail_logs, args=(job_id, done, success_marker, pushed_ok), daemon=True
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
        return

    if pushed_ok.is_set():
        print(f"\nTraining complete — model pushed to https://huggingface.co/{repo_id}")
        return

    stage = stage_holder.get("stage")
    if stage != "COMPLETED":
        message = stage_holder.get("message")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"Job {job_id} ended with stage={stage}{detail}. Check logs: hf jobs logs {job_id}"
        )
