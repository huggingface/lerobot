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
"""Run ``lerobot-curate-cameras`` on HF Jobs (HuggingFace GPUs).

Same shape as the annotation submitter (``lerobot.jobs.annotate``): the VLM
decision needs a GPU, so the pod boots the ``vllm/vllm-openai`` image, installs
lerobot on top, and replays the user's CLI with ``lerobot-curate-cameras``. The
``--mode=rename`` commit runs from the pod (which holds ``HF_TOKEN``); a bare
``--mode=report`` run leaves its output only on the pod, so we warn.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, get_token, run_job

from .annotate import build_pod_setup
from .dataset import ensure_dataset_available
from .hf import _pod_forwarded_args, follow_job, resolve_job_tags

if TYPE_CHECKING:
    from lerobot.annotations.camera_curation.config import CameraCurationConfig

# Same rationale as the annotate submitter: --root is host-local, --repo_id is
# re-emitted, config files can't be read on the pod, and --job could smuggle a
# remote target back onto the pod.
_SUBMITTER_OWNED_ARGS = ("--root", "--repo_id", "--config_path", "--job")


def _local_config_file_args(cfg: CameraCurationConfig) -> list[str]:
    return ["--config_path", *(f"--{name}" for name in vars(cfg) if is_dataclass(getattr(cfg, name)))]


def build_pod_command(repo_id: str, lerobot_ref: str, argv: list[str]) -> list[str]:
    """``bash -c`` command the pod runs: setup prelude, then curate-cameras."""
    forwarded = _pod_forwarded_args(argv, drop_names=_SUBMITTER_OWNED_ARGS, drop_prefixes=("--job.",))
    curate = shlex.join(
        ["lerobot-curate-cameras", f"--repo_id={repo_id}", *forwarded, "--job.target=local"]
    )
    return ["bash", "-c", f"{build_pod_setup(lerobot_ref)} && {curate}"]


def submit_curate_to_hf(cfg: CameraCurationConfig) -> None:
    """Submit a camera-curation run to HF Jobs and tail its logs."""
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    if cfg.repo_id is None:
        raise ValueError(
            "Remote curation requires --repo_id: the pod downloads the dataset from the Hub, "
            "and --root only names a directory on this machine."
        )

    argv = sys.argv[1:]
    passed = {tok.split("=", 1)[0] for tok in argv}
    used_config_files = sorted(passed.intersection(_local_config_file_args(cfg)))
    if used_config_files:
        raise ValueError(
            f"{', '.join(used_config_files)} cannot be used with a remote --job.target: the pod "
            "cannot read config files from this machine. Pass the settings as CLI flags instead."
        )

    if cfg.mode == "report":
        print(
            "WARNING: --mode=report writes its result into the pod's local copy, which is discarded "
            "when the job ends. Use --mode=rename to commit the result to the Hub."
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
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:     {job_url}")
    print(f"  Dataset repo: https://huggingface.co/datasets/{cfg.repo_id}")
    print(f"  Monitor:      hf jobs logs {job_id}")
    print(f"  Cancel:       hf jobs cancel {job_id}")

    if not follow_job(job_id, detach=cfg.job.detach):
        return

    print("\nCuration complete.")
