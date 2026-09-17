# Copyright 2025 Red Hat, Inc. and contributors. All rights reserved.
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
"""Run a lerobot training on a Kubernetes/OpenShift cluster via Kubeflow Trainer V2.

Submits a TrainJob CRD to the cluster, streams pod logs to stdout, and polls
until completion — same contract as the HF Jobs path in hf.py.
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_TERMINAL_PHASES = {"Succeeded", "Failed"}


def _hub_resumable_config_path(argv: list[str]) -> str | None:
    """Return --config_path from argv if it's a Hub repo id, else None.

    `_pod_forwarded_args` drops `--config_path` as submitter-side (a local checkpoint
    directory wouldn't exist inside the pod), but a Hub repo id resolves identically
    on both sides -- re-add it so `--resume=true --config_path=<hub-repo>` actually
    reaches the pod's `lerobot-train` invocation instead of being silently dropped.
    """
    config_path = None
    for i, tok in enumerate(argv):
        if tok == "--config_path" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif tok.startswith("--config_path="):
            config_path = tok.split("=", 1)[1]
    if not config_path:
        return None
    from pathlib import Path

    if Path(config_path).exists():
        return None
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(config_path)
    except HFValidationError:
        return None
    return config_path


def _pod_forwarded_args(argv: list[str]) -> list[str]:
    """User CLI overrides to replay on the pod, minus host-only flags.

    Drops --config_path, --dataset.root, and all --job.* flags since those are
    submitter-side. The pod always gets --job.target=local to prevent recursive dispatch.
    """
    drop_names = ("--config_path", "--dataset.root")
    drop_prefixes = ("--job.",)
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        name = tok.split("=", 1)[0]
        if name in drop_names or any(name.startswith(p) for p in drop_prefixes):
            if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                skip_next = True
            continue
        out.append(tok)
    return out


def _build_trainjob_manifest(
    cfg: TrainPipelineConfig,
    job_name: str,
    pod_command: list[str],
) -> dict:
    """Build a Kubeflow TrainJob manifest dict from the training config."""
    job_cfg = cfg.job
    gpu_resource = job_cfg.gpu_resource_key
    runtime = job_cfg.resolved_runtime

    env_vars = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]

    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            env_vars.append({"name": "HF_TOKEN", "value": token})
    except ImportError:
        pass

    if cfg.wandb.enable:
        from lerobot.jobs.hf import resolve_wandb_api_key

        wandb_key = resolve_wandb_api_key()
        if wandb_key:
            env_vars.append({"name": "WANDB_API_KEY", "value": wandb_key})

    trainer_spec = {
        "image": job_cfg.image,
        "command": pod_command,
        "numNodes": job_cfg.nodes,
        "env": env_vars,
    }
    if gpu_resource is not None:
        trainer_spec["resourcesPerNode"] = {
            "requests": {gpu_resource: str(job_cfg.gpus)},
            "limits": {gpu_resource: str(job_cfg.gpus)},
        }

    manifest = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name,
            "namespace": job_cfg.namespace,
        },
        "spec": {
            "runtimeRef": {"name": runtime},
            "trainer": trainer_spec,
        },
    }

    return manifest


def _create_trainjob(manifest: dict, kubeconfig: str | None = None) -> str:
    """Create a TrainJob on the cluster and return its name."""
    require_package("kubernetes", extra="kubeflow")
    from kubernetes import client, config

    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    api = client.CustomObjectsApi()
    namespace = manifest["metadata"]["namespace"]
    result = api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=manifest,
    )
    return result["metadata"]["name"]


def _get_trainjob_status(name: str, namespace: str, kubeconfig: str | None = None) -> dict:
    """Get the current status of a TrainJob."""
    from kubernetes import client, config

    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    api = client.CustomObjectsApi()
    obj = api.get_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        name=name,
    )
    return obj.get("status", {})


def _resolve_phase(status: dict) -> str | None:
    """Extract the phase from a TrainJob status dict.

    Kubeflow Trainer V2's TrainJob CRD (v1alpha1) exposes only "Suspended",
    "Complete", and "Failed" condition types (no "Succeeded"). We map
    "Complete" to "Succeeded" to keep the rest of this module's vocabulary
    unchanged.
    """
    conditions = status.get("conditions", [])
    for cond in conditions:
        ctype = cond.get("type", "")
        cstatus = cond.get("status", "")
        if ctype == "Failed" and cstatus == "True":
            return "Failed"
        if ctype == "Complete" and cstatus == "True":
            return "Succeeded"
    # Check if any replicated job has active pods
    for job_status in status.get("jobsStatus", []):
        if job_status.get("active", 0) > 0:
            return "Running"
    return "Pending"


def _get_pod_name_for_job(
    job_name: str, namespace: str, node: int = 0, kubeconfig: str | None = None
) -> str | None:
    """Find the pod name for a TrainJob's worker node."""
    from kubernetes import client, config

    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    v1 = client.CoreV1Api()
    # Kubeflow Trainer V2 provisions pods via JobSet; the TrainJob name is only
    # exposed through the JobSet's own label, not a trainer.kubeflow.org one.
    label_selector = f"jobset.sigs.k8s.io/jobset-name={job_name}"
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

    for pod in pods.items:
        pod_labels = pod.metadata.labels or {}
        # Match job-completion-index for the JobSet naming convention
        if pod_labels.get("batch.kubernetes.io/job-completion-index") == str(node):
            return pod.metadata.name

    # If no specific node match, return the first pod
    if pods.items:
        return pods.items[0].metadata.name
    return None


def _tail_logs(
    job_name: str,
    namespace: str,
    kubeconfig: str | None,
    done: threading.Event,
    success_marker: str | None = None,
    success_event: threading.Event | None = None,
) -> None:
    """Stream training pod logs to stdout until done is set.

    Reconnects on transient errors and waits for the pod to become available.
    """
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    v1 = client.CoreV1Api()
    printed = 0

    while not done.is_set():
        try:
            pod_name = _get_pod_name_for_job(job_name, namespace, node=0, kubeconfig=kubeconfig)
        except ApiException:
            if done.wait(5):
                return
            continue
        if pod_name is None:
            if done.wait(5):
                return
            continue

        try:
            log_stream = v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                follow=True,
                _preload_content=False,
            )
            seen = 0
            buffer = ""

            def _emit(line: str) -> bool:
                """Print a new line and report whether the caller should stop."""
                nonlocal seen, printed
                seen += 1
                if seen <= printed:
                    return False
                printed = seen
                print(line, flush=True)
                if success_marker and success_event is not None and success_marker in line:
                    success_event.set()
                    done.set()
                    return True
                return done.is_set()

            for raw_chunk in log_stream.stream():
                # .stream() yields arbitrary network chunks, not log lines — a chunk
                # can span multiple lines or split one mid-way. Buffer and split on
                # "\n" so `seen` counts real lines and stays comparable across
                # reconnects (each reconnect replays the full log from the start).
                buffer += raw_chunk.decode("utf-8", errors="replace")
                *complete_lines, buffer = buffer.split("\n")
                for line in complete_lines:
                    if _emit(line):
                        return
            if buffer and _emit(buffer):
                return
            if done.wait(3):
                return
        except ApiException as exc:
            if exc.status == 404:
                if done.wait(5):
                    return
            else:
                if done.wait(2):
                    return
        except Exception:
            if done.wait(2):
                return


def _poll_until_done(
    job_name: str,
    namespace: str,
    kubeconfig: str | None,
    done: threading.Event,
    poll_interval: float = 5.0,
    status_holder: dict | None = None,
    max_failures: int = 6,
) -> str | None:
    """Poll TrainJob status until terminal or done is set.

    Returns the terminal phase string, or None if done was set externally.
    """
    failures = 0
    while not done.is_set():
        try:
            status = _get_trainjob_status(job_name, namespace, kubeconfig)
            failures = 0
            phase = _resolve_phase(status)
            if phase in _TERMINAL_PHASES:
                if status_holder is not None:
                    conditions = status.get("conditions", [])
                    for cond in conditions:
                        if cond.get("type") == phase:
                            status_holder["message"] = cond.get("message")
                            break
                done.set()
                return phase
        except Exception:
            failures += 1
            if failures >= max_failures:
                done.set()
                return None
        done.wait(poll_interval)
    return None


def follow_job(
    job_name: str,
    namespace: str,
    kubeconfig: str | None = None,
    *,
    detach: bool = False,
    success_marker: str | None = None,
) -> bool:
    """Watch a submitted TrainJob to the end, streaming its logs to stdout.

    Returns True on success, False on detach/Ctrl-C.
    Raises RuntimeError on non-Succeeded terminal phase.
    """
    if detach:
        return False

    done_event = threading.Event()
    detached = threading.Event()
    marker_seen = threading.Event()
    stage_holder: dict[str, str | None] = {}

    def _poll() -> None:
        stage_holder["stage"] = _poll_until_done(
            job_name, namespace, kubeconfig, done_event, status_holder=stage_holder
        )

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    log_thread = threading.Thread(
        target=_tail_logs,
        args=(job_name, namespace, kubeconfig, done_event, success_marker, marker_seen),
        daemon=True,
    )
    log_thread.start()

    def _detach(sig, frame):
        detached.set()
        done_event.set()
        print("\nDetached. TrainJob is still running on the cluster.")
        print(f"  Status:  kubectl get trainjob {job_name} -n {namespace}")
        print(f"  Logs:    kubectl logs -l jobset.sigs.k8s.io/jobset-name={job_name} -n {namespace} -f")
        print(f"  Cancel:  kubectl delete trainjob {job_name} -n {namespace}")

    install_sigint = threading.current_thread() is threading.main_thread()
    original_sigint = signal.getsignal(signal.SIGINT) if install_sigint else None
    if install_sigint:
        signal.signal(signal.SIGINT, _detach)
    try:
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

    phase = stage_holder.get("stage")
    if phase != "Succeeded":
        message = stage_holder.get("message")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"TrainJob {job_name} ended with phase={phase}{detail}. "
            f"Check logs: kubectl logs -l jobset.sigs.k8s.io/jobset-name={job_name} -n {namespace}"
        )
    return True


def _generate_job_name(cfg: TrainPipelineConfig) -> str:
    """Generate a unique TrainJob name from the config."""
    import datetime as dt
    import re

    base = cfg.job_name or cfg.policy.type if cfg.policy else "train"
    slug = re.sub(r"[^a-z0-9-]", "-", base.lower()).strip("-") or "train"
    # K8s names must be <= 63 chars and DNS-1035 compliant
    # "lerobot-" (8) + slug + "-" (1) + "YYYYMMDD-HHMMSS" (15) = slug + 24
    slug = slug[:39]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"lerobot-{slug}-{stamp}"


def submit_to_kubeflow(cfg: TrainPipelineConfig) -> None:
    """Submit a training job to a Kubernetes cluster via Kubeflow Trainer V2.

    Validates the config, builds a TrainJob manifest, creates it on the cluster,
    and streams logs until completion. Ctrl-C detaches without cancelling the job.
    """
    require_package("kubernetes", extra="kubeflow")

    cfg.validate()

    if cfg.is_reward_model_training:
        raise ValueError(
            "Remote training via --job.target=kubeflow only supports policy training, "
            "not reward models. Run reward-model training locally."
        )

    job_name = _generate_job_name(cfg)

    # Build the pod command: replay user args with --job.target=local to prevent
    # recursive dispatch on the pod.
    forwarded = _pod_forwarded_args(sys.argv[1:])

    # A Hub-repo-id --config_path (e.g. for --resume=true) resolves the same way
    # from the pod as from here, so re-add it after _pod_forwarded_args stripped it.
    # A local-directory config_path is NOT forwarded: it only exists on this machine.
    hub_config_path = _hub_resumable_config_path(sys.argv[1:])
    if hub_config_path is not None:
        forwarded = [*forwarded, f"--config_path={hub_config_path}"]

    pod_command = [
        "lerobot-train",
        *forwarded,
        "--job.target=local",
    ]

    # If push_to_hub is configured, ensure repo_id is set
    if cfg.policy and cfg.policy.push_to_hub and cfg.policy.repo_id:
        success_marker = f"Model pushed to https://huggingface.co/{cfg.policy.repo_id}"
    else:
        success_marker = None

    manifest = _build_trainjob_manifest(cfg, job_name, pod_command)

    namespace = cfg.job.namespace
    print(f"Submitting TrainJob to Kubeflow (namespace={namespace}, runtime={cfg.job.resolved_runtime}) ...")
    print(f"  Image:     {cfg.job.image}")
    print(f"  Nodes:     {cfg.job.nodes}")
    print(f"  GPUs/node: {cfg.job.gpus} ({cfg.job.resolved_gpu_vendor})")

    created_name = _create_trainjob(manifest, kubeconfig=cfg.job.kubeconfig)

    print(f"TrainJob created: {created_name}")
    print(f"  Status:  kubectl get trainjob {created_name} -n {namespace}")
    print(f"  Logs:    kubectl logs -l jobset.sigs.k8s.io/jobset-name={created_name} -n {namespace} -f")
    print(f"  Cancel:  kubectl delete trainjob {created_name} -n {namespace}")

    if follow_job(
        created_name,
        namespace,
        kubeconfig=cfg.job.kubeconfig,
        detach=cfg.job.detach,
        success_marker=success_marker,
    ):
        if cfg.policy and cfg.policy.repo_id:
            print(f"\nTraining complete — model pushed to https://huggingface.co/{cfg.policy.repo_id}")
        else:
            print("\nTraining complete.")
