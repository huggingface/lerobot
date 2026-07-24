#!/usr/bin/env python

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
"""End-to-end multi-GPU tests for the distributed core.

Sized for a 4-GPU CI lane, these tests execute the sharded code paths nothing else in the tree can
reach — ``fully_shard`` via ``accelerator.prepare``, the DCP branches of ``save_checkpoint`` /
``save_training_state`` / ``resume_after_prepare``, the collective gather inside
``save_pretrained``, and HSDP/DDP gradient reduction — against the tiny
``DummyCheckpointPolicy`` fixture on synthetic data (no datasets, no network, no site paths).

Run on a node with at least 4 GPUs::

    pytest -m multigpu tests/distributed/test_multigpu_training.py -v

Mechanics:

- Plain pytest, no ``torchrun``: each test launches its own ranks with
  ``torch.multiprocessing.spawn`` (spawn start method) and a per-test free TCP port; workers set
  the torchrun-equivalent env (``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_*``) that
  accelerate's ``env://`` initialization consumes.
- Deadlock watchdog (:func:`_spawn`): the spawn context is polled with a deadline instead of a
  blocking join, so a hung collective — the exact failure mode the all-ranks contracts guard
  against — fails the test with ``TimeoutError`` (all workers SIGKILLed) rather than hanging CI.
  A worker exception propagates through ``ProcessContext.join``, which tears down the survivors.
- Workers configure accelerate exclusively through the LeRobot config mirrors
  (``AcceleratorConfig.build(ParallelismConfig)`` after ``resolve(world_size)``) — the same
  construction path ``make_accelerator`` takes; see :func:`_build_accelerator` for why the
  factory itself is not called.
- Without GPUs every test skips (``torch.cuda.device_count()`` gate), so the file is safe to
  collect and run in the CPU lanes.
"""

import json
import os
import socket
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from lerobot.common.train_utils import resume_after_prepare, resume_before_prepare, save_checkpoint
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.distributed.checkpoint import full_model_state_dict, is_sharded_module
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR

# The spawned children re-import this module by name, so this import must resolve there too:
# torch.multiprocessing propagates the parent's sys.path through the spawn preparation data.
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy

SEED = 20260712
HIDDEN = 8  # DummyCheckpointPolicy is one Linear(hidden, hidden): 4 ranks shard dim 0 evenly
BATCH_SIZE = 2
SAVE_STEP = 2  # optimizer steps run before saving in the round-trip workers
PARITY_STEPS = 3
GA_UPDATES = 3
SAMPLES_PER_UPDATE = 4  # per rank per optimizer update — the fixed effective batch of test 5
GRAD_CLIP_NORM = 100.0  # generous: exercises the clip call without perturbing parity
# Generous headroom for cold NCCL init plus the lerobot re-import in 4 spawned children, while
# still bounding a deadlocked collective to minutes instead of a hung CI job.
WATCHDOG_TIMEOUT_S = 240.0
_JOIN_POLL_S = 5.0


# -------------------------------------------------------------------------------------------
# Spawn infrastructure
# -------------------------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn(world_size: int, worker, *args, timeout_s: float = WATCHDOG_TIMEOUT_S) -> None:
    """Run ``worker(rank, world_size, port, *args)`` on ``world_size`` fresh processes.

    Watchdog approach: ``mp.spawn(join=False)`` returns a ``ProcessContext`` whose ``join`` is
    polled under a deadline. On timeout every surviving worker is SIGKILLed and the test fails
    with ``TimeoutError`` — a deadlock can never hang CI. When a worker raises, ``join`` itself
    kills the remaining ranks and re-raises the worker's exception into the test.
    """
    port = _find_free_port()
    context = mp.spawn(worker, args=(world_size, port, *args), nprocs=world_size, join=False)
    deadline = time.monotonic() + timeout_s
    while not context.join(timeout=_JOIN_POLL_S):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            for process in context.processes:
                process.join(timeout=10)
            raise TimeoutError(
                f"{getattr(worker, '__name__', worker)}: {world_size} workers still running "
                f"after {timeout_s}s — presumed deadlock; all workers killed."
            )


def _init_worker_env(rank: int, world_size: int, port: int) -> None:
    """Give the worker the torchrun-equivalent env accelerate's ``env://`` init consumes."""
    # The tests configure accelerate through the config mirrors only; drop any accelerate env
    # fallbacks inherited from the launching shell (what guard_against_env_interference would
    # reject in production — here the env is simply owned by the test).
    for name in list(os.environ):
        if name.startswith(("FSDP_", "PARALLELISM_CONFIG_", "ACCELERATE_")):
            del os.environ[name]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # The fp32 parity tolerances below assume true-fp32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# -------------------------------------------------------------------------------------------
# Shared building blocks
# -------------------------------------------------------------------------------------------


def _make_cfg(
    world_size: int,
    *,
    dp_replicate: int = 1,
    dp_shard: int = 1,
    checkpoint_format: CheckpointFormat = CheckpointFormat.SAFETENSORS,
    grad_accum: int = 1,
) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), batch_size=BATCH_SIZE)
    cfg.checkpoint_format = checkpoint_format
    cfg.parallelism.dp_replicate = dp_replicate
    cfg.parallelism.dp_shard = dp_shard
    cfg.accelerator.mixed_precision = "no"  # fp32 end to end: the parity tests depend on it
    cfg.accelerator.gradient_accumulation.steps = grad_accum
    # The dummy policy declares no _fsdp_wrap_modules; the size-based wrap policy shards its
    # Linear without needing class names (the set_fsdp_wrap_modules no-op branch).
    cfg.accelerator.fsdp.min_num_params = 1
    cfg.parallelism.resolve(world_size)
    return cfg


def _build_accelerator(cfg: TrainPipelineConfig):
    """``cfg.accelerator.build(cfg.parallelism)`` — make_accelerator's construction path.

    Deliberately not ``make_accelerator`` itself: the factory additionally derives ``cpu=`` from
    ``cfg.trainable_config`` (no policy config is attached to these synthetic cfgs) and re-runs
    the env guard — both owned explicitly by the tests (see ``_init_worker_env``).
    """
    return cfg.accelerator.build(cfg.parallelism)


def _make_policy(seed: int) -> DummyCheckpointPolicy:
    """Identically seeded on every rank, so shard/replicate starts from one common init."""
    torch.manual_seed(seed)
    return DummyCheckpointPolicy(DummyCheckpointConfig(hidden=HIDDEN, device="cpu"))


def _batch(step: int, rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Deterministic per-(step, rank) batch: every dp worker sees distinct, reproducible data."""
    generator = torch.Generator().manual_seed(SEED + 1000 * step + rank)
    return {"observation.state": torch.randn(BATCH_SIZE, HIDDEN, generator=generator).to(device)}


def _gather_full(model, optimizer) -> tuple[dict, dict]:
    """Full (unsharded) model + optimizer state via torch's DCP state-dict API — a COLLECTIVE.

    With ``cpu_offload=True`` the dicts materialize on the main rank only; every other rank
    receives a literal ``{}``.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return (
        get_model_state_dict(model, options=options),
        get_optimizer_state_dict(model, optimizer, options=options),
    )


def _assert_tree_equal(reference, actual, path: str) -> None:
    """Exact (bitwise for tensors) equality of nested state dicts, with a failing path."""
    if isinstance(reference, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: {type(actual)} is not a tensor"
        assert reference.dtype == actual.dtype, f"{path}: {reference.dtype} != {actual.dtype}"
        assert reference.shape == actual.shape, f"{path}: {reference.shape} != {actual.shape}"
        assert torch.equal(reference.cpu(), actual.cpu()), f"{path}: tensor values differ"
    elif isinstance(reference, dict):
        assert isinstance(actual, dict), f"{path}: {type(actual)} is not a dict"
        assert set(reference) == set(actual), f"{path}: keys {set(reference) ^ set(actual)} differ"
        for key in reference:
            _assert_tree_equal(reference[key], actual[key], f"{path}.{key}")
    elif isinstance(reference, list | tuple):
        assert type(reference) is type(actual) and len(reference) == len(actual), path
        for index, (ref_item, actual_item) in enumerate(zip(reference, actual, strict=True)):
            _assert_tree_equal(ref_item, actual_item, f"{path}[{index}]")
    else:
        assert reference == actual, f"{path}: {reference!r} != {actual!r}"


# -------------------------------------------------------------------------------------------
# Workers (module-level: torch.multiprocessing.spawn pickles them by reference)
# -------------------------------------------------------------------------------------------


def _train_and_save_worker(rank: int, world_size: int, port: int, tmp_dir: str, fmt_value: str) -> None:
    """FSDP2 (dp_shard=world_size): train SAVE_STEP steps, save_checkpoint, store the gathered
    full model/optimizer state as the rank-0 reference for the resume workers."""
    _init_worker_env(rank, world_size, port)
    tmp = Path(tmp_dir)
    fmt = CheckpointFormat(fmt_value)
    cfg = _make_cfg(world_size, dp_shard=world_size, checkpoint_format=fmt)
    accelerator = _build_accelerator(cfg)
    policy = _make_policy(SEED)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    # FSDP2 requires model and optimizer in one prepare() call (accelerate rebinds param groups).
    policy, optimizer = accelerator.prepare(policy, optimizer)
    assert is_sharded_module(accelerator.unwrap_model(policy)), "prepare() did not shard the policy"

    for step in range(SAVE_STEP):
        loss, _ = policy(_batch(step, rank, accelerator.device))
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    checkpoint_dir = tmp / "checkpoint"
    save_checkpoint(
        checkpoint_dir, step=SAVE_STEP, cfg=cfg, policy=policy, optimizer=optimizer, accelerator=accelerator
    )

    model_state, optimizer_state = _gather_full(policy, optimizer)
    if accelerator.is_main_process:
        from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME

        pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        assert (pretrained_dir / f"{FSDP_MODEL_NAME}_0").is_dir() == fmt.wants_dcp
        assert (pretrained_dir / "model.safetensors").is_file() == fmt.wants_safetensors
        assert (pretrained_dir / "config.json").is_file()
        assert (pretrained_dir / "train_config.json").is_file()
        # Sharded runs always use the DCP optimizer channel, never the safetensors one.
        assert (checkpoint_dir / TRAINING_STATE_DIR / f"{OPTIMIZER_NAME}_0").is_dir()
        assert not (checkpoint_dir / TRAINING_STATE_DIR / "optimizer_state.safetensors").exists()
        torch.save({"model": model_state, "optimizer": optimizer_state}, tmp / "reference_state.pt")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


def _resume_and_verify_worker(rank: int, world_size: int, port: int, tmp_dir: str, fmt_value: str) -> None:
    """Two-phase resume at dp_shard=world_size; the gathered state must match the saved
    reference exactly (DCP round-trips are bit-exact)."""
    _init_worker_env(rank, world_size, port)
    tmp = Path(tmp_dir)
    cfg = _make_cfg(world_size, dp_shard=world_size, checkpoint_format=CheckpointFormat(fmt_value))
    cfg.checkpoint_path = tmp / "checkpoint"
    accelerator = _build_accelerator(cfg)

    assert resume_before_prepare(cfg) == SAVE_STEP  # phase 1: RNG + step counter only

    # Deliberately different init: the DCP load must overwrite every parameter.
    policy = _make_policy(SEED + 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    resume_after_prepare(cfg, accelerator, policy, optimizer, None)  # phase 2: DCP reshard-load

    model_state, optimizer_state = _gather_full(policy, optimizer)
    if accelerator.is_main_process:
        reference = torch.load(tmp / "reference_state.pt", map_location="cpu", weights_only=True)
        _assert_tree_equal(reference["model"], model_state, "model")
        _assert_tree_equal(reference["optimizer"], optimizer_state, "optimizer")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


def _loss_parity_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, dp_replicate: int, dp_shard: int, tag: str
) -> None:
    """Train PARITY_STEPS fp32 steps on per-rank deterministic data; rank 0 records the
    dp-mean loss of every step. Gradient averaging spans the same rank set in any (R, S)
    factorization of the world, so the loss trajectory is topology-invariant."""
    _init_worker_env(rank, world_size, port)
    cfg = _make_cfg(world_size, dp_replicate=dp_replicate, dp_shard=dp_shard)
    accelerator = _build_accelerator(cfg)
    policy = _make_policy(SEED)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    assert is_sharded_module(accelerator.unwrap_model(policy)) == (dp_shard > 1)

    per_step_losses = []
    for step in range(PARITY_STEPS):
        loss, _ = policy(_batch(step, rank, accelerator.device))
        per_step_losses.append(accelerator.gather(loss.detach().reshape(1)).double().mean().item())
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    if accelerator.is_main_process:
        (Path(tmp_dir) / f"losses_{tag}.json").write_text(json.dumps(per_step_losses))
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


def _save_pretrained_all_ranks_worker(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    """The all-ranks contract: every rank calls save_pretrained, the
    collective gather completes (watchdog proves no deadlock), and only rank 0 writes files."""
    _init_worker_env(rank, world_size, port)
    cfg = _make_cfg(world_size, dp_shard=world_size)
    accelerator = _build_accelerator(cfg)
    policy = _make_policy(SEED)
    # FSDP2 prepare requires an optimizer alongside the model even though this test never steps it.
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    unwrapped = accelerator.unwrap_model(policy)
    assert is_sharded_module(unwrapped)

    # Gather semantics: the full dict materializes on the main rank; every other rank
    # receives the literal empty dict.
    reference = full_model_state_dict(unwrapped)
    if accelerator.is_main_process:
        assert set(reference) == {"net.weight", "net.bias"}
    else:
        assert reference == {}

    # Every rank targets its own directory so writes are attributable per rank.
    target = Path(tmp_dir) / f"rank_{rank}"
    unwrapped.save_pretrained(target)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        weights = load_file(target / "model.safetensors")
        assert set(weights) == set(reference)
        for key, tensor in reference.items():
            assert torch.equal(weights[key], tensor), key
        assert (target / "config.json").is_file()
    else:
        assert list(target.rglob("*")) == [], f"rank {rank} wrote files despite the rank-0 gate"
    dist.destroy_process_group()


def _grad_accum_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, micro_batch_size: int, grad_accum: int, tag: str
) -> None:
    """DDP fp32 with the exact accumulate/clip/step/zero_grad pattern of
    ``lerobot_train.update_policy``; rank 0 records the final weights."""
    _init_worker_env(rank, world_size, port)
    assert micro_batch_size * grad_accum == SAMPLES_PER_UPDATE  # fixed effective batch
    cfg = _make_cfg(world_size, dp_replicate=world_size, grad_accum=grad_accum)
    accelerator = _build_accelerator(cfg)
    # The GradientAccumulationPlugin wiring, un-overridden by any env fallback.
    assert accelerator.gradient_accumulation_steps == grad_accum
    policy = _make_policy(SEED)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    policy, optimizer = accelerator.prepare(policy, optimizer)

    # One fixed per-rank sample stream, consumed in order by both variants: update k always
    # covers rows [k * SAMPLES_PER_UPDATE, (k + 1) * SAMPLES_PER_UPDATE).
    generator = torch.Generator().manual_seed(SEED + 7919 * rank)
    stream = torch.randn(GA_UPDATES * SAMPLES_PER_UPDATE, HIDDEN, generator=generator)

    updates_applied = 0
    for micro_step in range(GA_UPDATES * grad_accum):
        rows = stream[micro_step * micro_batch_size : (micro_step + 1) * micro_batch_size]
        batch = {"observation.state": rows.to(accelerator.device)}
        # update_policy's pattern: accumulate() suppresses grad sync and rescales the loss on
        # non-final micro-batches, and AcceleratedOptimizer makes step()/zero_grad() no-ops
        # until sync_gradients is True.
        with accelerator.accumulate(policy):
            loss, _ = policy(batch)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
                updates_applied += 1
            optimizer.step()
            optimizer.zero_grad()
    assert updates_applied == GA_UPDATES  # exactly one optimizer update per accumulation window

    if accelerator.is_main_process:
        state = {key: value.cpu() for key, value in accelerator.unwrap_model(policy).state_dict().items()}
        torch.save(state, Path(tmp_dir) / f"weights_{tag}.pt")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


# -------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
def test_fsdp2_train_save_resume_round_trip(tmp_path):
    """FSDP2 dp_shard=4, checkpoint_format=safetensors_dcp: train -> save_checkpoint -> resume.

    A second spawn resumes through the two-phase path and its gathered model weights and Adam
    state tensors must match the pre-save gathered reference exactly (DCP round-trips are
    bit-exact).
    """
    fmt = CheckpointFormat.SAFETENSORS_AND_DCP.value
    _spawn(4, _train_and_save_worker, str(tmp_path), fmt)
    _spawn(4, _resume_and_verify_worker, str(tmp_path), fmt)


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
def test_hsdp_loss_parity_with_ddp(tmp_path):
    """Same seed and per-rank data: DDP (dp_replicate=4) vs HSDP (2x2), fp32, no AMP.

    Both topologies average gradients over the same four ranks, so per-step dp-mean losses must
    match within tolerance. Exact parity is not expected: DDP all-reduces where HSDP
    reduce-scatters within the shard group and all-reduces across replicas, and the different
    reduction orders accumulate fp32 rounding — rtol=1e-4 leaves orders of magnitude of headroom
    over that noise while still catching any real divergence (wrong averaging, wrong data).
    """
    _spawn(4, _loss_parity_worker, str(tmp_path), 4, 1, "ddp")
    _spawn(4, _loss_parity_worker, str(tmp_path), 2, 2, "hsdp")
    ddp_losses = json.loads((tmp_path / "losses_ddp.json").read_text())
    hsdp_losses = json.loads((tmp_path / "losses_hsdp.json").read_text())
    assert len(ddp_losses) == len(hsdp_losses) == PARITY_STEPS
    for step, (ddp_loss, hsdp_loss) in enumerate(zip(ddp_losses, hsdp_losses, strict=True)):
        assert hsdp_loss == pytest.approx(ddp_loss, rel=1e-4, abs=1e-6), f"step {step}"


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
def test_changed_topology_resume(tmp_path):
    """Save at dp_shard=4 (format=dcp), resume at dp_shard=2 on 2 ranks.

    The DCP load reshards both the model weights and the optimizer state across the topology
    change; the post-resume gathered state must equal the pre-save gathered reference exactly
    (cross-topology resharding is runtime-verified).
    """
    fmt = CheckpointFormat.DCP.value
    _spawn(4, _train_and_save_worker, str(tmp_path), fmt)
    _spawn(2, _resume_and_verify_worker, str(tmp_path), fmt)


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
def test_save_pretrained_all_ranks_no_deadlock(tmp_path):
    """dp_shard=4: save_pretrained on ALL ranks completes under the watchdog.

    Rank 0 writes model.safetensors (+ config.json) whose tensors equal the gathered full state;
    ranks 1-3 write nothing. A rank-gated call would deadlock in the collective gather and be
    killed by :func:`_spawn`'s timeout — completing at all is half of what this test asserts.
    """
    _spawn(4, _save_pretrained_all_ranks_worker, str(tmp_path))


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_gradient_accumulation_equivalence(tmp_path):
    """Fixed effective batch on 2-rank DDP fp32: (batch=4, GA=1) vs (batch=2, GA=2).

    Both variants consume the identical per-rank sample stream in the same order for
    GA_UPDATES optimizer updates, using update_policy's accumulate/clip/step pattern. The final
    weights must agree: accumulate() rescales each micro-loss by 1/GA, so summed mean-of-2
    gradients equal the mean-of-4 gradient up to fp32 summation order — hence allclose with
    rtol=1e-5/atol=1e-6 (roughly 100x the observed associativity noise), not bitwise equality.
    """
    _spawn(2, _grad_accum_worker, str(tmp_path), 4, 1, "ga1")
    _spawn(2, _grad_accum_worker, str(tmp_path), 2, 2, "ga2")
    ga1 = torch.load(tmp_path / "weights_ga1.pt", weights_only=True)
    ga2 = torch.load(tmp_path / "weights_ga2.pt", weights_only=True)
    assert set(ga1) == set(ga2) == {"net.weight", "net.bias"}
    for key in ga1:
        assert torch.allclose(ga1[key], ga2[key], rtol=1e-5, atol=1e-6), key
