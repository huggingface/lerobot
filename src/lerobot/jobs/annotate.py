# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Run ``lerobot-annotate`` on HF Jobs (HuggingFace GPUs).

Same shape as the training submitter in ``hf.py``, with one difference: the
annotation pipeline serves its own VLM, so the pod starts from the official
``vllm/vllm-openai`` image (which has no lerobot) instead of the prebuilt
``lerobot-gpu`` image, and installs lerobot on top before running.

Because there is no config repo to stage, the pod replays the user's own CLI
flags — everything except the client-only ``--job.*`` and the host-local
``--root``, which is replaced by ``--repo_id`` so the pod pulls the dataset
from the Hub.
"""

from __future__ import annotations

import shlex
import signal
import sys
import threading
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, get_token, run_job

from .dataset import ensure_dataset_available

# Package-internal reuse of the training submitter's job plumbing: the polling,
# log tailing and argv forwarding are identical for annotation runs.
from .hf import _pod_forwarded_args, _poll_until_done, _tail_logs, resolve_job_tags

if TYPE_CHECKING:
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig

LEROBOT_GIT_URL = "https://github.com/huggingface/lerobot.git"

# Mirrors the pins in pyproject.toml. The vLLM image resolves dependencies on its
# own otherwise, and pulls av 18 / datasets 5 / draccus 0.11 — each of which breaks
# lerobot at import time. `--upgrade-strategy only-if-needed` keeps vLLM's own
# (torch, transformers, ...) pins intact.
_RUNTIME_REQUIREMENTS = (
    "'datasets>=4.7.0,<5.0.0' 'pyarrow>=21.0.0,<30.0.0' 'av>=15.0.0,<16.0.0' 'draccus==0.10.0' "
    "'pandas>=2.0.0,<3.0.0' jsonlines gymnasium torchcodec mergedeep pyyaml-include toml typing-inspect "
    "openai"
)

# Flags the submitter resolves itself instead of forwarding verbatim: `--root`
# names a directory only this machine has, `--repo_id` is re-emitted from the
# config, and `--config_path` names a local file (rejected up front by
# `submit_annotate_to_hf`). `--job.*` is dropped separately, by prefix.
_SUBMITTER_OWNED_ARGS = ("--root", "--repo_id", "--config_path")


def build_pod_setup(lerobot_ref: str) -> str:
    """Shell prelude that turns the vLLM image into a ``lerobot-annotate`` runtime."""
    spec = f"lerobot @ git+{LEROBOT_GIT_URL}@{lerobot_ref}"
    return (
        # git to install from the repo, ffmpeg to decode the dataset's videos.
        "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
        f"pip install --no-deps {shlex.quote(spec)} && "
        f"pip install --upgrade-strategy only-if-needed {_RUNTIME_REQUIREMENTS} && "
        # vLLM's cudagraph memory estimate over-reserves and starves the KV cache;
        # PyAV is the video backend the server can decode our frames with.
        "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
        "export VLLM_VIDEO_BACKEND=pyav"
    )


def build_pod_command(repo_id: str, lerobot_ref: str, argv: list[str]) -> list[str]:
    """Build the ``bash -c`` command the pod runs: setup prelude, then annotation.

    ``argv`` is the user's CLI (``sys.argv[1:]``) minus the flags in
    ``_SUBMITTER_OWNED_ARGS``; ``--repo_id`` is re-added from the config so the pod
    always annotates the dataset we just made sure is reachable on the Hub.
    ``--job.target=local`` stops the pod from re-dispatching to itself.
    """
    forwarded = _pod_forwarded_args(argv, drop_names=_SUBMITTER_OWNED_ARGS, drop_prefixes=("--job.",))
    annotate = shlex.join(["lerobot-annotate", f"--repo_id={repo_id}", *forwarded, "--job.target=local"])
    return ["bash", "-c", f"{build_pod_setup(lerobot_ref)} && {annotate}"]


def submit_annotate_to_hf(cfg: AnnotationPipelineConfig) -> None:
    """Submit an annotation run to HF Jobs infrastructure.

    Resolves credentials, makes sure the source dataset is reachable from the pod,
    submits the job, then tails its logs until the job reaches a terminal stage —
    or returns immediately with ``--job.detach``. Ctrl-C detaches without
    cancelling the remote job.
    """
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    if cfg.repo_id is None:
        raise ValueError(
            "Remote annotation requires --repo_id: the pod downloads the dataset from the Hub, "
            "and --root only names a directory on this machine."
        )

    argv = sys.argv[1:]
    if any(tok.split("=", 1)[0] == "--config_path" for tok in argv):
        raise ValueError(
            "--config_path is not supported with a remote --job.target: the pod cannot read a "
            "local config file. Pass the settings as CLI flags instead."
        )

    if not cfg.push_to_hub:
        # The pod's filesystem is discarded when the job ends, so without a push the
        # run produces nothing. Warn rather than fail: a smoke test over
        # --only_episodes that only inspects the logs is a legitimate use.
        print(
            "WARNING: --push_to_hub is off. The annotated dataset lives only on the pod and is "
            "discarded when the job ends. Pass --push_to_hub=true to keep the result."
        )

    api = HfApi(token=token)
    tags = resolve_job_tags(cfg.job.tags)
    ensure_dataset_available(cfg.repo_id, api=api, tags=tags)

    command = build_pod_command(cfg.repo_id, cfg.job.lerobot_ref, argv)

    print(f"Submitting job to HF Jobs (flavor={cfg.job.target}, image={cfg.job.image}) ...")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
        secrets={"HF_TOKEN": token},
        timeout=cfg.job.timeout,
        # HF Jobs labels are key/value; expose each tag as a queryable label.
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:     {job_url}")
    target_repo_id = cfg.new_repo_id or cfg.repo_id
    if cfg.push_to_hub:
        print(f"  Dataset repo: https://huggingface.co/datasets/{target_repo_id}")
    print(f"  Monitor:      hf jobs logs {job_id}")
    print(f"  Cancel:       hf jobs cancel {job_id}")

    if cfg.job.detach:
        return

    done = threading.Event()
    detached = threading.Event()
    stage_holder: dict[str, str | None] = {}

    def _poll() -> None:
        stage_holder["stage"] = _poll_until_done(job_id, done, status_holder=stage_holder)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    log_thread = threading.Thread(target=_tail_logs, args=(job_id, done), daemon=True)
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

    stage = stage_holder.get("stage")
    if stage != "COMPLETED":
        message = stage_holder.get("message")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"Job {job_id} ended with stage={stage}{detail}. Check logs: hf jobs logs {job_id}"
        )

    if cfg.push_to_hub:
        print(f"\nAnnotation complete — dataset pushed to https://huggingface.co/datasets/{target_repo_id}")
    else:
        print("\nAnnotation complete. Note: --push_to_hub was off, so the result stayed on the pod.")
