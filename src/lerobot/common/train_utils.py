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
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim import (
    load_optimizer_state,
    load_optimizer_state_dict,
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
from lerobot.utils.hub import find_latest_hub_checkpoint
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def should_save_checkpoint(step: int, save_freq: int, total_steps: int) -> bool:
    """Whether a checkpoint should be saved at ``step``.

    A checkpoint is saved every ``save_freq`` steps and always after the final step. A
    non-positive ``save_freq`` disables periodic saving (only the final checkpoint is
    written), mirroring how ``log_freq``/``eval_freq`` treat non-positive values and
    avoiding a ``ZeroDivisionError`` from ``step % 0``.
    """
    return (save_freq > 0 and step % save_freq == 0) or step == total_steps


def save_training_step(
    step: int, save_dir: Path, num_processes: int | None = None, batch_size: int | None = None
) -> None:
    state: dict = {"step": step}
    # num_processes and batch_size are recorded so a resumed run can detect a changed world size or
    # batch size: the sampler's resume offset is computed from the (num_processes, batch_size) that
    # produced `step`, since both scale how many sampler positions a step consumes (see
    # compute_sampler_state).
    if num_processes is not None:
        state["num_processes"] = num_processes
    if batch_size is not None:
        state["batch_size"] = batch_size
    write_json(state, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def load_training_num_processes(checkpoint_dir: Path) -> int | None:
    """World size recorded at checkpoint time, or None for checkpoints written before it was stored."""
    return load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP).get("num_processes")


def load_training_batch_size(checkpoint_dir: Path) -> int | None:
    """Per-process batch size recorded at checkpoint time, or None for older checkpoints."""
    return load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP).get("batch_size")


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    num_processes: int | None = None,
    batch_size: int | None = None,
    model_state_dict: dict | None = None,
    optim_state_dict: dict | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        preprocessor: The preprocessor/pipeline to save. Defaults to None.
        postprocessor: The postprocessor/pipeline to save. Defaults to None.
        num_processes (int | None, optional): Distributed world size to record for sample-exact
            resume. Defaults to None (not recorded).
        batch_size (int | None, optional): Per-process batch size to record for sample-exact
            resume. Defaults to None (not recorded).
        model_state_dict: Pre-gathered full (unsharded) model state dict. Required under FSDP,
            where `policy.state_dict()` would return sharded tensors; the caller gathers it via a
            cross-rank collective and passes it here so rank 0 can write it directly. It holds
            FSDP's fp32 master weights and is saved as-is (the loader casts to the policy dtype on
            read). When None (DDP / single-GPU), the model is saved the normal way. Defaults to None.
        optim_state_dict: Pre-gathered full (unsharded) optimizer state dict. Required under FSDP
            (gathered alongside `model_state_dict` via `gather_fsdp_state_dicts`); saved in the same
            safetensors format as the single-GPU path. When None, `optimizer.state_dict()` is used.
            Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir, state_dict=model_state_dict)
    cfg.save_pretrained(pretrained_dir)
    if cfg.peft is not None:
        # When using PEFT, policy.save_pretrained will only write the adapter weights + config, not the
        # policy config which we need for loading the model. In this case we'll write it ourselves.
        policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(
        checkpoint_dir,
        step,
        optimizer,
        scheduler,
        num_processes=num_processes,
        batch_size=batch_size,
        optim_state_dict=optim_state_dict,
    )


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    num_processes: int | None = None,
    batch_size: int | None = None,
    optim_state_dict: dict | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
        num_processes (int | None, optional): Distributed world size to record. Defaults to None.
        batch_size (int | None, optional): Per-process batch size to record. Defaults to None.
        optim_state_dict: Pre-gathered full optimizer state dict (for FSDP). Saved instead of
            `optimizer.state_dict()` when provided. Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir, num_processes=num_processes, batch_size=batch_size)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir, optim_state_dict=optim_state_dict)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None, load_optimizer: bool = True
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).
        load_optimizer (bool, optional): Whether to load the optimizer state from disk. Defaults to
            True. Set to False under FSDP, where the sharded optimizer state must be loaded after
            `accelerator.prepare()` via `load_fsdp_optimizer_state` (the optimizer is returned
            untouched here).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    if load_optimizer:
        optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler


def _is_fsdp2(model) -> bool:
    """Whether ``model`` is an FSDP2 ``fully_shard`` module.

    FSDP1 wraps the root in ``FullyShardedDataParallel``; FSDP2 mutates wrapped modules to implement
    ``FSDPModule``. The two APIs have incompatible optimizer-state conversion paths, so this check
    must stay local to the checkpoint helpers rather than relying on Accelerate's shared FSDP enum.
    """
    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


def gather_fsdp_state_dicts(model, optimizer) -> tuple[dict, dict]:
    """Gather the full (unsharded) model and optimizer state dicts under FSDP.

    This must run on every rank with the prepared model and optimizer. FSDP1's ``state_dict_type``
    and FSDP2's distributed-checkpoint APIs both materialize CPU full state only on rank 0; other
    ranks receive empty dictionaries. The resulting parameter-FQN keyed optimizer state is portable
    across FSDP world sizes and is reshaped by ``load_fsdp_optimizer_state`` on resume.
    """
    if _is_fsdp2(model):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        # FSDP2 returns the full CPU tensors only on rank 0, matching the FSDP1 rank0_only
        # contract below; non-main ranks still join the collective but receive empty dictionaries.
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
        return get_model_state_dict(model, options=options), get_optimizer_state_dict(
            model, optimizer, options=options
        )

    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,  # noqa F401
        StateDictType,
    )

    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg, optim_cfg):
        model_state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    return model_state_dict, optim_state_dict


def load_fsdp_optimizer_state(model, optimizer, checkpoint_dir: Path) -> None:
    """Load a portable FSDP optimizer state into the prepared optimizer.

    This cross-rank operation runs after ``accelerator.prepare()``. FSDP1 converts the saved full
    state into its current shard topology, while FSDP2 performs the corresponding DTensor-aware
    conversion in ``set_optimizer_state_dict``. Do not call ``optimizer.load_state_dict`` after the
    FSDP2 setter: it has already installed the correctly sharded state.
    """
    full_osd = load_optimizer_state_dict(checkpoint_dir / TRAINING_STATE_DIR)

    if _is_fsdp2(model):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        set_optimizer_state_dict(
            model,
            optimizer,
            full_osd,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True),
        )
        return

    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,  # noqa F401
        StateDictType,
    )

    # Every rank reads the same full state from the shared checkpoint directory.
    state_cfg = FullStateDictConfig(rank0_only=False)
    optim_cfg = FullOptimStateDictConfig(rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg, optim_cfg):
        sharded_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=full_osd)
    optimizer.load_state_dict(sharded_osd)


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
