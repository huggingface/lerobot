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
import io
import json
import netrc
import os
import re
import signal
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
from huggingface_hub import (
    HfApi,
    create_repo,
    fetch_job_logs,
    get_token,
    inspect_job,
    run_job,
    upload_file,
)

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

_TERMINAL_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}

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

    # Round-trip through draccus to get the canonical, pod-parseable layout, then
    # drop the keys the released trainer image doesn't know about.
    buf = io.StringIO()
    with draccus.config_type("json"):
        draccus.dump(remote, buf, indent=4)
    data = json.loads(buf.getvalue())
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
        except Exception:
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
            stage = info.status.stage.value
            if stage in _TERMINAL_STAGES:
                if status_holder is not None:
                    status_holder["message"] = getattr(info.status, "message", None)
                done.set()
                return stage
        except Exception:
            failures += 1
            if failures >= max_failures:
                done.set()
                return None
        done.wait(poll_interval)
    return None


def submit_to_hf(cfg: TrainPipelineConfig) -> None:
    """Submit a training job to HF Jobs infrastructure.

    Validates cfg, resolves credentials, stages the config on the Hub, submits
    the job, then either tails logs until completion or detaches immediately.
    Ctrl-C detaches without cancelling the remote job.
    """
    from lerobot.jobs.dataset import ensure_dataset_available

    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    api = HfApi(token=token)
    user_info = api.whoami(token=token)
    username = user_info["name"]

    now = dt.datetime.now()
    if cfg.policy is not None:
        base_name = cfg.job_name or cfg.policy.type
        repo_id = cfg.policy.repo_id or build_repo_id(username, base_name, now)
        cfg.policy.repo_id = repo_id
        cfg.policy.push_to_hub = True
    else:
        # Path-based policy is resolved inside validate(); fall back to a generic slug.
        repo_id = build_repo_id(username, cfg.job_name or "train", now)

    cfg.validate()

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
    ensure_dataset_available(cfg.dataset.repo_id, api=api, tags=tags)

    config_repo_id = _stage_config_on_hub(cfg, repo_id, token, tags=tags)
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
    # post-run finalization before the job stage flips to COMPLETED.
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

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _detach)
    try:
        # Timeout-based join so SIGINT is delivered to the main thread promptly.
        while poll_thread.is_alive():
            poll_thread.join(timeout=0.5)
        log_thread.join(timeout=5)
    finally:
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
