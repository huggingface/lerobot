#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Training-output persistence: checkpoints, two-phase resume, and hub publishing.

Rank discipline: every function here that can
contain a collective is documented as such and must run on ALL ranks; rank-0-only file writes
sit under one grouped ``is_main_process()`` gate per contiguous region, placed below all
collectives. The leaf save/load helpers carry no rank gates of their own — the exception is
``PreTrainedPolicy._save_pretrained``, whose gate is internal because its collective gather and
its writes live in the same method.
"""

import logging
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from huggingface_hub import HfApi, ModelCard, ModelCardData, snapshot_download
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.__version__ import __version__
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed.checkpoint import (
    is_sharded_module,
    load_sharded_model,
    load_sharded_optimizer,
    save_sharded_model,
    save_sharded_optimizer,
)
from lerobot.distributed.utils import is_main_process
from lerobot.optim import (
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.hub import DCP_ARTIFACT_PATTERNS, HubMixin, find_latest_hub_checkpoint
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import load_rng_state, save_rng_state

if TYPE_CHECKING:
    from accelerate import Accelerator

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def get_step_identifier(step: int, total_steps: int) -> str:
    """Format a step number as the zero-padded identifier used for checkpoint directory names.

    Args:
        step (int): The training step to format.
        total_steps (int): The total number of training steps; sets the padding width
            (minimum 6 digits).

    Returns:
        str: The zero-padded step identifier, e.g. `"005000"`.
    """
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number.

    Args:
        output_dir (Path): The training run's output directory.
        total_steps (int): The total number of training steps; sets the identifier padding.
        step (int): The training step of the checkpoint.

    Returns:
        Path: The checkpoint step directory, `output_dir/checkpoints/<step-identifier>`.
    """
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    """Point the `last` symlink in the checkpoints directory at the given checkpoint.

    Any existing `last` symlink is replaced. The link target is relative to the checkpoints
    directory, so the tree stays valid when the run directory is moved.

    Args:
        checkpoint_dir (Path): The checkpoint step directory the `last` link should target.
    """
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


# ---------------------------------------------------------------------------------------------
# training_step.json
# ---------------------------------------------------------------------------------------------


def save_training_step(step: int, save_dir: Path, cfg: TrainPipelineConfig) -> None:
    """Record the step counter plus everything a resume needs to reason about topology changes.

    `step` counts loop iterations (= micro-batches), so
    the sampler resume offset is `step x batch_size x dp_world_size` with no grad-accum factor.
    `grad_accum_steps` and the parallelism snapshot are recorded so a resume can warn precisely
    when the optimizer-update cadence or the sharding topology changed.

    Args:
        step (int): The training step (micro-batch counter) to record.
        save_dir (Path): The `training_state/` directory to write `training_step.json` into.
        cfg (TrainPipelineConfig): The training config whose batch size, gradient-accumulation,
            and parallelism settings are snapshotted alongside the step.
    """
    state: dict[str, Any] = {
        "step": step,
        "dp_world_size": cfg.parallelism.dp_world_size,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.accelerator.gradient_accumulation.steps,
        "parallelism": {
            "dp_replicate": cfg.parallelism.dp_replicate,
            "dp_shard": cfg.parallelism.dp_shard,
            "ring_degree": cfg.parallelism.context_parallel.ring_degree,
            "ulysses_degree": cfg.parallelism.context_parallel.ulysses_degree,
        },
    }
    write_json(state, save_dir / TRAINING_STEP)


def load_training_step(training_state_dir: Path) -> int:
    """Read the step counter recorded in `training_step.json`.

    Args:
        training_state_dir (Path): The checkpoint's `training_state/` directory.

    Returns:
        int: The recorded training step (micro-batch counter).
    """
    return int(load_json(training_state_dir / TRAINING_STEP)["step"])


def load_training_dp_world_size(checkpoint_dir: Path) -> int | None:
    """Data-parallel world size recorded at checkpoint time, or None for very old checkpoints.

    Falls back to the pre-v0.7 `num_processes` key (world size == dp world size back then, as
    context parallelism did not exist). Legacy fallback for checkpoints written before v0.7;
    remove in the next major version.

    Args:
        checkpoint_dir (Path): The checkpoint step directory (containing `training_state/`).

    Returns:
        int | None: The recorded data-parallel world size, or None for checkpoints written
            before it (or its `num_processes` predecessor) was recorded.
    """
    state = load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP)
    return state.get("dp_world_size", state.get("num_processes"))


def load_training_batch_size(checkpoint_dir: Path) -> int | None:
    """Per-process batch size recorded at checkpoint time, or None for older checkpoints.

    Args:
        checkpoint_dir (Path): The checkpoint step directory (containing `training_state/`).

    Returns:
        int | None: The recorded per-process `batch_size`, or None for checkpoints written
            before it was recorded.
    """
    return load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP).get("batch_size")


def load_training_grad_accum_steps(checkpoint_dir: Path) -> int | None:
    """Gradient-accumulation steps recorded at checkpoint time, or None for older checkpoints.

    Args:
        checkpoint_dir (Path): The checkpoint step directory (containing `training_state/`).

    Returns:
        int | None: The recorded `gradient_accumulation_steps`, or None for checkpoints written
            before it was recorded.
    """
    return load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP).get("grad_accum_steps")


def load_training_parallelism(checkpoint_dir: Path) -> dict[str, int] | None:
    """Parallelism snapshot recorded at checkpoint time, or None for older checkpoints.

    Args:
        checkpoint_dir (Path): The checkpoint step directory (containing `training_state/`).

    Returns:
        dict[str, int] | None: The recorded degrees
            (`{"dp_replicate", "dp_shard", "ring_degree", "ulysses_degree"}`), or None for
            checkpoints written before the snapshot was recorded.
    """
    return load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP).get("parallelism")


# ---------------------------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    accelerator: "Accelerator | None" = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights (checkpoint_format ∈ {safetensors, safetensors_dcp}, or any non-sharded run)
    │   ├── pytorch_model_fsdp_0/  # DCP model shards (checkpoint_format ∈ {dcp, safetensors_dcp})
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  # optimizer param groups (non-sharded runs)
        ├── optimizer_state.safetensors  # optimizer state (non-sharded runs)
        ├── optimizer_0/  # DCP optimizer shards (sharded runs)
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step + dp_world_size/batch_size/grad_accum + topology

    Collective: MUST be called on every rank. Rank-0-only writes are gated internally, so the
    call site needs no rank branches.

    Args:
        checkpoint_dir (Path): The checkpoint step directory to write (e.g. `.../checkpoints/005000`).
        step (int): The training step at that checkpoint.
        cfg (TrainPipelineConfig): The training config used for this run.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer): The optimizer to save the state from.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        preprocessor (PolicyProcessorPipeline | None, optional): The preprocessor/pipeline to save.
            Defaults to None.
        postprocessor (PolicyProcessorPipeline | None, optional): The postprocessor/pipeline to save.
            Defaults to None.
        accelerator (Accelerator | None, optional): The accelerator the policy was prepared with;
            used to unwrap the model and required on sharded runs, where it owns the DCP save
            channels. Defaults to None (plain single-process saves).
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    fmt = cfg.checkpoint_format
    policy_to_save = accelerator.unwrap_model(policy) if accelerator is not None else policy
    sharded = is_sharded_module(policy_to_save)

    # -- model artifact(s): the two collective-capable calls ----------------------------------
    if cfg.peft is not None:
        # PeftModel.save_pretrained is an external API with no internal rank gate, and the
        # adapters are replicated (PEFT x sharded is rejected at validation): main rank writes.
        if is_main_process():
            policy_to_save.save_pretrained(pretrained_dir)
    elif fmt.wants_safetensors or not sharded:
        # Collective when sharded (full gather); writes happen on the main process only in all
        # multi-rank layouts (the gate lives inside _save_pretrained, next to its collective gather).
        policy_to_save.save_pretrained(pretrained_dir)
    if fmt.wants_dcp and sharded:
        save_sharded_model(accelerator, policy_to_save, pretrained_dir)

    # -- sidecar configs: ONE gate for the whole contiguous rank-0-only region ----------------
    if is_main_process():
        if fmt.wants_dcp and not fmt.wants_safetensors:
            # save_pretrained did not run: keep the DCP-only checkpoint self-describing.
            policy_to_save.config.save_pretrained(pretrained_dir)
        cfg.save_pretrained(pretrained_dir)
        if cfg.peft is not None:
            # PEFT's save_pretrained writes only adapter weights + config; the policy config
            # needed to reload the base model is written explicitly.
            policy_to_save.config.save_pretrained(pretrained_dir)
        if preprocessor is not None:
            preprocessor.save_pretrained(pretrained_dir)
        if postprocessor is not None:
            postprocessor.save_pretrained(pretrained_dir)

    save_training_state(
        checkpoint_dir, step, cfg, optimizer, scheduler, accelerator, sharded=sharded, model=policy_to_save
    )
    if accelerator is not None:
        accelerator.wait_for_everyone()


def save_training_state(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    optimizer: Optimizer | dict[str, Optimizer] | None = None,
    scheduler: LRScheduler | None = None,
    accelerator: "Accelerator | None" = None,
    *,
    sharded: bool = False,
    model: PreTrainedPolicy | None = None,
) -> None:
    """Write training_state/. Collective under sharding: call on every rank.

    Args:
        checkpoint_dir (Path): The checkpoint step directory; `training_state/` is created inside it.
        step (int): The training step at that checkpoint.
        cfg (TrainPipelineConfig): The training config used for this run (its topology and
            accumulation settings are recorded in `training_step.json`).
        optimizer (Optimizer | dict[str, Optimizer] | None, optional): The optimizer(s) to save
            the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from.
            Defaults to None.
        accelerator (Accelerator | None, optional): Required when `sharded` is True — it owns
            the DCP optimizer save channel. Defaults to None.
        sharded (bool): The model's sharding state, computed once in `save_checkpoint` and
            threaded here so the two sites cannot disagree. Defaults to False.
        model (PreTrainedPolicy | None, optional): Required only for the sharded optimizer
            channel: torch's optimizer DCP APIs are model-coupled (the state dict is keyed by
            model FQNs), so accelerate's `save_fsdp_optimizer` needs the sharded module
            alongside the optimizer. Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    # All ranks: the directory must exist before the DCP optimizer collective writes into it
    # (exist_ok makes the concurrent mkdir race-free on shared filesystems).
    save_dir.mkdir(parents=True, exist_ok=True)

    if optimizer is not None and sharded:
        if accelerator is None or model is None:
            raise ValueError("Saving a sharded optimizer state requires the accelerator and model.")
        # Collective — all ranks write their DCP shards into optimizer_0/.
        save_sharded_optimizer(accelerator, optimizer, model, save_dir)

    if is_main_process():  # ONE grouped gate for the whole rank-0-only region
        save_training_step(step, save_dir, cfg)
        save_rng_state(save_dir)
        if scheduler is not None:
            save_scheduler_state(scheduler, save_dir)
        if optimizer is not None and not sharded:
            save_optimizer_state(optimizer, save_dir)


# ---------------------------------------------------------------------------------------------
# Two-phase resume
# ---------------------------------------------------------------------------------------------


def resume_before_prepare(cfg: TrainPipelineConfig) -> int:
    """Phase 1 — before `accelerator.prepare()`: restore RNG and return the step counter.

    Pure loaders only. The sampler resume offset is *derived* from the returned step inside the
    dataloader factory, and everything bound to sharded objects (model DCP shards, optimizer,
    scheduler) loads in `resume_after_prepare`.

    Args:
        cfg (TrainPipelineConfig): The resumed training config; `cfg.checkpoint_path` locates
            the checkpoint to restore from.

    Returns:
        int: The training step recorded in the checkpoint (micro-batch counter).

    Raises:
        NotADirectoryError: If the checkpoint has no `training_state/` directory.
    """
    training_state_dir = cfg.checkpoint_path / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)
    _warn_on_resume_changes(cfg)
    load_rng_state(training_state_dir)
    return load_training_step(training_state_dir)


def _warn_on_resume_changes(cfg: TrainPipelineConfig) -> None:
    """One warning naming every recorded run setting this resume changes.

    Changes are legal — DCP reshards weights and optimizer state across topologies and the
    sampler offset adapts — but a changed ``grad_accum_steps`` shifts the optimizer-update
    cadence, so the resume says precisely what differs. The sampler-exactness warnings
    (``dp_world_size``/``batch_size``) live with the sampler math in the dataloader factory.

    Args:
        cfg (TrainPipelineConfig): The resumed training config, compared against the settings
            recorded in the checkpoint at `cfg.checkpoint_path`.
    """
    recorded = {
        "grad_accum_steps": (
            load_training_grad_accum_steps(cfg.checkpoint_path),
            cfg.accelerator.gradient_accumulation.steps,
        ),
    }
    snapshot = load_training_parallelism(cfg.checkpoint_path)
    if snapshot is not None:
        recorded.update(
            {
                "dp_replicate": (snapshot.get("dp_replicate"), cfg.parallelism.dp_replicate),
                "dp_shard": (snapshot.get("dp_shard"), cfg.parallelism.dp_shard),
                "ring_degree": (
                    snapshot.get("ring_degree"),
                    cfg.parallelism.context_parallel.ring_degree,
                ),
                "ulysses_degree": (
                    snapshot.get("ulysses_degree"),
                    cfg.parallelism.context_parallel.ulysses_degree,
                ),
            }
        )
    changed = [f"{key}: {was} -> {now}" for key, (was, now) in recorded.items() if was not in (None, now)]
    if changed and is_main_process():
        logging.warning(
            "Resuming with settings that differ from the checkpoint: " + "; ".join(changed) + ". "
            "Topology changes reshard safely via DCP; a changed grad_accum_steps shifts the "
            "optimizer-update cadence (the step counter keeps counting micro-batches)."
        )


def resume_after_prepare(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator",
    policy: PreTrainedPolicy,
    optimizer: Optimizer | dict[str, Optimizer],
    scheduler: LRScheduler | None,
) -> None:
    """Phase 2 — after `accelerator.prepare()`: model (DCP) -> optimizer -> scheduler.

    Collective under sharding: call on every rank. The model-weight source follows the
    checkpoint's own recorded `checkpoint_format` (on resume, `cfg` was parsed from the
    checkpoint's train_config.json): DCP-bearing formats load shards here into the prepared
    model (whose construction skipped the safetensors load); the safetensors format was already
    loaded by `from_pretrained` before sharding — no model step here.

    Args:
        cfg (TrainPipelineConfig): The resumed training config; `cfg.checkpoint_path` locates
            the checkpoint and `cfg.checkpoint_format` selects the model-weight source.
        accelerator (Accelerator): The accelerator the policy was prepared with; it unwraps the
            model and owns the DCP load channels.
        policy (PreTrainedPolicy): The prepared (possibly sharded) policy to load weights into.
        optimizer (Optimizer | dict[str, Optimizer]): The prepared optimizer(s) to restore.
        scheduler (LRScheduler | None): The scheduler to restore, or None if the run has none.

    Raises:
        FileNotFoundError: If the checkpoint format declares DCP model shards but the shard
            directory is missing (e.g. it was pruned before upload).
    """
    checkpoint_dir = cfg.checkpoint_path
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    unwrapped = accelerator.unwrap_model(policy)
    sharded = is_sharded_module(unwrapped)

    if cfg.checkpoint_format.wants_dcp:
        from accelerate.utils.constants import FSDP_MODEL_NAME

        dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
        if not dcp_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint_format={cfg.checkpoint_format.value} declares DCP model shards, "
                f"but {dcp_dir} is missing. If the shards were pruned, convert what remains "
                "with `lerobot-convert-dcp` or resume from a safetensors checkpoint."
            )
        load_sharded_model(accelerator, unwrapped, pretrained_dir)

    if sharded:
        # Requires the prepared optimizer: FSDP2's prepare rebinds param groups to DTensors but
        # never migrates optimizer.state — DCP reshards it here (works across topology changes).
        load_sharded_optimizer(accelerator, optimizer, unwrapped, training_state_dir)
    else:
        load_optimizer_state(optimizer, training_state_dir)

    if scheduler is not None:
        load_scheduler_state(scheduler, training_state_dir)


# ---------------------------------------------------------------------------------------------
# Hub: checkpoint push (resume artifact) and publishing (distribution artifact)
# ---------------------------------------------------------------------------------------------


def push_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    *,
    private: bool | None = None,
) -> None:
    """Upload a saved checkpoint directory to the Hub under checkpoints/<name>/.

    Called once per save step when save_checkpoint_to_hub is enabled, so a
    timed-out or crashed run still leaves recoverable checkpoints on the Hub.
    The model repo is created idempotently, and the commit is tagged with the
    checkpoint step so a checkpoint can be recovered with
    --policy.pretrained_revision=<step> instead of a commit sha.

    The directory is uploaded verbatim — including DCP shards under the DCP formats: this tree
    exists for *resume*, not distribution, and `resolve_resume_checkpoint` downloads it back
    symmetrically.

    Args:
        checkpoint_dir (Path): The local checkpoint step directory to upload.
        repo_id (str): The Hub model repo to push to (created idempotently if missing).
        private (bool | None): Whether a newly created repo should be private. Defaults to
            None (public unless the organization's default is private).
    """
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"checkpoints/{checkpoint_dir.name}",
        commit_message=f"checkpoint {checkpoint_dir.name}",
    )
    api.create_tag(
        repo_id=repo_id,
        tag=checkpoint_dir.name,
        revision=commit.oid,
        repo_type="model",
        exist_ok=True,
    )


def resolve_resume_checkpoint(repo_id: str, output_dir: Path) -> Path:
    """Download the latest checkpoint of a Hub training repo into a local run dir.

    The symmetric counterpart to `push_checkpoint_to_hub`: given a model repo holding
    `checkpoints/<step>/{pretrained_model,training_state}` subtrees, download the highest-numbered step
    into `output_dir/checkpoints/<step>/`, recreate the local `last` symlink, and return that local
    checkpoint dir. Used to resume training from the Hub on a machine (or HF Jobs pod) that does not
    have the original local run dir.

    Args:
        repo_id (str): The Hub model repo holding `checkpoints/<step>/` subtrees.
        output_dir (Path): The local run directory to download the checkpoint into.

    Returns:
        Path: The local checkpoint step directory, `output_dir/checkpoints/<step>`.

    Raises:
        FileNotFoundError: If the repo contains no checkpoints under `checkpoints/`.
    """
    latest = find_latest_hub_checkpoint(repo_id)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{repo_id}' under '{CHECKPOINTS_DIR}/'. "
            "Was the run trained with --save_checkpoint_to_hub?"
        )
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{latest}/*",
        local_dir=str(output_dir),
    )
    checkpoint_dir = output_dir / latest
    update_last_checkpoint(checkpoint_dir)
    return checkpoint_dir


def publish_trained_model(
    cfg: TrainPipelineConfig,
    model: HubMixin,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    dataset_meta: "LeRobotDatasetMetadata | None",
    *,
    peft_model: Any | None = None,
) -> None:
    """Publish the complete training bundle as a distributable model repo.

    Collective-safe: call on ALL ranks — the model commit gathers sharded weights through
    `save_pretrained`; uploads happen on the main process only (gated inside
    `HubMixin.push_to_hub` and here). Commits, in order: (1) the model (skipped for PEFT —
    adapters replace full weights), (2) the preprocessor, (3) the postprocessor, (4) the bundle
    sidecar: README.md model card + train_config.json (+ adapter weights and the wrapped
    policy's config in the PEFT case). Published repos carry safetensors only — DCP resume
    artifacts are filtered from every upload.

    Args:
        cfg (TrainPipelineConfig): The training config; saved as `train_config.json` and used
            to render the model card.
        model (HubMixin): The trained model to publish; its config supplies the target repo id,
            visibility, license, and tags.
        preprocessor (PolicyProcessorPipeline | None): The preprocessor pipeline to publish
            alongside the model, if any.
        postprocessor (PolicyProcessorPipeline | None): The postprocessor pipeline to publish
            alongside the model, if any.
        dataset_meta (LeRobotDatasetMetadata | None): Dataset metadata for the model card, if
            available.
        peft_model (Any | None): The PEFT wrapper when training adapters; its adapter weights
            replace the full model weights in the published repo. Defaults to None.

    Raises:
        ValueError: If the model config carries no repo id (`--policy.repo_id`).
    """
    model_cfg = model.config
    repo_id = model_cfg.repo_id
    if not repo_id:
        raise ValueError("Publishing requires a repo id (--policy.repo_id).")
    ignore = ["*.tmp", "*.log", *DCP_ARTIFACT_PATTERNS]

    if peft_model is None:
        # Calls are made on the exact objects that own each method (never through PEFT's
        # attribute forwarding), so the peft branch below never touches this path.
        model.push_to_hub(repo_id, private=model_cfg.private, ignore_patterns=ignore)
    if preprocessor is not None:
        preprocessor.push_to_hub(repo_id, private=model_cfg.private)
    if postprocessor is not None:
        postprocessor.push_to_hub(repo_id, private=model_cfg.private)

    if is_main_process():
        api = HfApi()
        repo_id = api.create_repo(repo_id=repo_id, private=model_cfg.private, exist_ok=True).repo_id
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            saved_path.mkdir(parents=True, exist_ok=True)
            if peft_model is not None:
                peft_model.save_pretrained(saved_path)  # adapter weights + adapter config
                model.config.save_pretrained(saved_path)  # PEFT cannot write the policy config
            if hasattr(type(model), "generate_model_card"):
                # Model families with their own card template (reward models) render it
                # themselves; policies use the shared template below.
                card = model.generate_model_card(
                    cfg.dataset.repo_id, model_cfg.type, model_cfg.license, model_cfg.tags
                )
            else:
                card = generate_model_card(model_cfg, cfg=cfg, dataset_meta=dataset_meta)
            card.save(str(saved_path / "README.md"))
            cfg.save_pretrained(saved_path)  # train_config.json
            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload model card and train config",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=ignore,
            )
        # Contract: lerobot.jobs.hf.submit_to_hf watches for this exact "Model pushed to <url>"
        # line to end a remote run early. Keep the wording and URL format in sync.
        logging.info(f"Model pushed to {commit_info.repo_url.url}")

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------------------------

_BASE_MODEL_MAPPING = {
    "smolvla": "lerobot/smolvla_base",
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
    "pi0_fast": "lerobot/pi0fast-base",
    "xvla": "lerobot/xvla-base",
}


def build_card_context(
    cfg: TrainPipelineConfig | None,
    dataset_meta: "LeRobotDatasetMetadata | None",
    input_features: dict | None,
    output_features: dict | None,
) -> dict:
    """Collect optional data for the model-card template.

    Returns plain values only (no Markdown) — the template in
    ``lerobot/templates/lerobot_modelcard_template.md`` decides how and whether to show
    each one. Everything is best-effort: anything unavailable is left empty/None and the
    template simply skips that section, so this never breaks a Hub push.

    Args:
        cfg (TrainPipelineConfig | None): The training config supplying the training section,
            if available.
        dataset_meta (LeRobotDatasetMetadata | None): Dataset metadata supplying the dataset,
            robot-type, and camera sections, if available.
        input_features (dict | None): The policy's input feature declarations, if any.
        output_features (dict | None): The policy's output feature declarations, if any.

    Returns:
        dict: Template context with `training`, `input_features`, `output_features`,
            `dataset`, `robot_type`, and `cameras` entries; unavailable pieces stay
            empty/None.
    """
    context = {
        "training": None,
        "input_features": input_features or {},
        "output_features": output_features or {},
        "dataset": None,
        "robot_type": None,
        "cameras": [],
    }

    if cfg is not None:
        optimizer = getattr(cfg, "optimizer", None)
        context["training"] = {
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "optimizer": getattr(optimizer, "type", None) if optimizer else None,
            "lr": getattr(optimizer, "lr", None) if optimizer else None,
            "lerobot_version": __version__,
        }

    if dataset_meta is not None:
        context["dataset"] = {
            "repo_id": dataset_meta.repo_id,
            "episodes": dataset_meta.total_episodes,
            "frames": dataset_meta.total_frames,
            "fps": dataset_meta.fps,
            "tasks": [str(task) for task in dataset_meta.tasks.index],
        }
        context["robot_type"] = dataset_meta.robot_type
        context["cameras"] = [key.split(".")[-1] for key in dataset_meta.camera_keys]

    return context


def generate_model_card(
    model_cfg: PreTrainedConfig,
    cfg: TrainPipelineConfig | None = None,
    dataset_meta: "LeRobotDatasetMetadata | None" = None,
) -> ModelCard:
    """Render the LeRobot model card for a trained model.

    A free function on purpose: every template variable comes from arguments —
    the model config, the training config, and the dataset metadata — none from a live model.
    Reward-model configs without feature declarations degrade to empty sections.

    Args:
        model_cfg (PreTrainedConfig): The policy config providing type, license, tags, repo id,
            and feature declarations.
        cfg (TrainPipelineConfig | None, optional): The training config for the training and
            dataset card sections. Defaults to None.
        dataset_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata for the
            dataset card sections. Defaults to None.

    Returns:
        ModelCard: The rendered and validated LeRobot model card.
    """
    model_type = model_cfg.type
    card_data = ModelCardData(
        license=model_cfg.license or "apache-2.0",
        library_name="lerobot",
        pipeline_tag="robotics",
        tags=list(set(model_cfg.tags or []).union({"robotics", "lerobot", model_type})),
        model_name=model_type,
        datasets=cfg.dataset.repo_id if cfg is not None else None,
        base_model=_BASE_MODEL_MAPPING.get(model_type),
    )

    context = build_card_context(
        cfg,
        dataset_meta,
        getattr(model_cfg, "input_features", None),
        getattr(model_cfg, "output_features", None),
    )
    # Used by the template to pre-fill commands and the "Fine-tuned from" line.
    context["policy_repo_id"] = getattr(model_cfg, "repo_id", None)
    context["base_model"] = _BASE_MODEL_MAPPING.get(model_type)

    template_card = (
        files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
    )
    card = ModelCard.from_template(card_data, template_str=template_card, **context)
    card.validate()
    return card
