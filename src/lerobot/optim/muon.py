"""DTensor-aware Muon optimizer for FSDP2 and MoE expert weights.

Vendored from the upstream LingBot-VLA v2 training stack
(``lingbotvla/optim/muon.py``, Apache-2.0) with numerics kept identical. The
``CombinedOptimizer`` at the bottom is the lerobot-side adaptation: it swaps
upstream's nested ``{"optimizers": [...]}`` checkpoint format for a standard
flat ``Optimizer.state_dict`` so the shared safetensors save/load path in
``lerobot.optim.optimizers`` works unchanged.

``DistributedMuon`` keeps upstream ``torch.optim.Muon`` numerics for 2D
weights and adds batched Newton-Schulz for 3D MoE expert stacks.

For FSDP2-sharded 2D params, same-shape parameters are mega-batched:
stacked into a single tensor, gathered with one NCCL call, orthogonalized
with one batched NS pass, and scattered back locally.

Under plain DDP / single-process, parameters are ordinary tensors and every
parameter takes the local path: Newton-Schulz runs on the full gradient,
which is the mathematically correct input. Under FSDP1 the gradients exposed
to the optimizer are dim-0 row shards, so orthogonalizing them is silently
wrong and world-size dependent — ``lerobot_train`` rejects that combination
(see ``is_muon_optimizer``).
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim.optimizer import Optimizer

try:
    from torch.optim._muon import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_C,
        DEFAULT_NS_STEPS,
        EPS,
        _adjust_lr,
    )

    _MUON_AVAILABLE = True
except ImportError:  # pragma: no cover - torch < 2.9 fallback
    _MUON_AVAILABLE = False
    DEFAULT_A = 3.4445
    DEFAULT_B = -4.7750
    DEFAULT_C = 2.0315
    DEFAULT_NS_STEPS = 5
    EPS = 1e-7

    def _adjust_lr(  # type: ignore[no-redef]
        lr: float,
        adjust_lr_fn: str | None,
        param_shape: Sequence[int],
    ) -> float:
        """Torch 2.8 fallback for ``torch.optim._muon._adjust_lr``."""
        if adjust_lr_fn is None:
            return lr
        fan_out, fan_in = param_shape[:2]
        if adjust_lr_fn == "original":
            return lr * math.sqrt(max(1.0, fan_out / fan_in))
        if adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
        raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")


__all__ = [
    "CombinedOptimizer",
    "DEFAULT_NS_COEFFICIENTS",
    "DEFAULT_NS_STEPS",
    "DistributedMuon",
    "batched_newton_schulz",
    "is_muon_optimizer",
    "split_muon_adamw_params",
]


DEFAULT_NS_COEFFICIENTS: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

_DEFAULT_ADAMW_NAME_PATTERNS: tuple[str, ...] = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
)

_MEGABATCH_MAX_GROUP_SIZE = 32


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run quintic Newton-Schulz on each trailing ``[M, K]`` matrix."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim < 2:
        raise ValueError(f"Input must have ndim >= 2, got shape {tuple(grad.shape)}")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    original_dtype = grad.dtype
    ortho = grad.to(compute_dtype)

    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT

    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    for _ in range(ns_steps):
        A = ortho @ ortho.mT
        if A.ndim == 2:
            gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
            ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        else:
            *batch, M_, K_ = ortho.shape
            B_ = 1
            for d in batch:
                B_ *= d
            A_3d = A.reshape(B_, M_, M_)
            ortho_3d = ortho.reshape(B_, M_, K_)
            gram_update = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a).reshape(*batch, M_, K_)
        del A

    if transposed:
        ortho = ortho.mT

    return ortho.to(original_dtype)


def _is_adamw_by_name(name: str, extra_patterns: Sequence[str]) -> bool:
    lname = name.lower()
    for pat in _DEFAULT_ADAMW_NAME_PATTERNS:
        if pat in lname:
            return True
    for pat in extra_patterns:
        if pat and pat.lower() in lname:
            return True
    return False


def _is_muon_eligible_ndim(param: Tensor) -> bool:
    """Return True for dense linears and 3D MoE expert stacks."""
    return param.ndim in (2, 3)


def split_muon_adamw_params(
    model: "nn.Module",
    no_decay_modules: list[str] | None = None,
    no_decay_params: list[str] | None = None,
    extra_adamw_name_patterns: Sequence[str] | None = None,
) -> tuple[list[Tensor], list[Tensor], list[str], list[str]]:
    """Split model parameters into Muon-eligible weights and AdamW fallback weights."""
    no_decay_modules = no_decay_modules or []
    no_decay_params = no_decay_params or []
    extra_patterns = list(extra_adamw_name_patterns or ())

    forced_adamw_fqns: set = set()
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        is_embedding = isinstance(module, nn.Embedding)
        is_no_decay = cls_name in no_decay_modules
        if is_embedding or is_no_decay:
            for pname, _p in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{pname}" if module_name else pname
                forced_adamw_fqns.add(fqn)

    muon_params: list[Tensor] = []
    adamw_params: list[Tensor] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        muon_ok = _is_muon_eligible_ndim(param)
        forced_adamw = (
            (not muon_ok)
            or name in forced_adamw_fqns
            or _is_adamw_by_name(name, extra_patterns)
            or any(p and p.lower() in name.lower() for p in no_decay_params)
        )
        if forced_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


_KIND_LOCAL = "local"
_KIND_FSDP_GATHER_2D = "fsdp_gather_2d"
_KIND_MOE_LOCAL_3D = "moe_local_3d"
_KIND_MOE_GATHER_3D = "moe_gather_3d"


def _shard_dims(p: DTensor) -> list[int]:
    """Return the list of tensor dims along which ``p`` is sharded."""
    return [pl.dim for pl in p.placements if isinstance(pl, Shard)]


def _classify_param(p: Tensor) -> str:
    """Return one of ``_KIND_*`` describing how Muon should treat ``p``."""
    if not isinstance(p, DTensor):
        return _KIND_LOCAL

    shard_dims = _shard_dims(p)
    if not shard_dims:
        return _KIND_LOCAL

    if p.ndim == 2:
        return _KIND_FSDP_GATHER_2D

    if p.ndim == 3:
        if all(d == 0 for d in shard_dims):
            return _KIND_MOE_LOCAL_3D
        return _KIND_MOE_GATHER_3D

    raise ValueError(
        f"DistributedMuon got an unexpected param rank {p.ndim} "
        f"(shape={tuple(p.shape)}). Only 2D and 3D params are supported."
    )


def _full_grad(grad: Tensor) -> Tensor:
    """Return a replicated tensor, all-gathering DTensor gradients if needed."""
    if isinstance(grad, DTensor):
        return grad.full_tensor()
    return grad


def _wrap_full_as_dtensor_like(full: Tensor, ref: Tensor) -> Tensor:
    """Wrap ``full`` as a DTensor with ``ref``'s placements."""
    if not isinstance(ref, DTensor):
        return full

    mesh = ref.device_mesh
    replicated = DTensor.from_local(
        full,
        device_mesh=mesh,
        placements=[Replicate()] * mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(device_mesh=mesh, placements=ref.placements)


def _get_dtensor_shard_info(p: DTensor) -> tuple[Any, int, int, int]:
    """Extract (process_group, world_size, rank, shard_dim) from a sharded DTensor."""
    mesh = p.device_mesh
    for mesh_dim_idx, placement in enumerate(p.placements):
        if isinstance(placement, Shard):
            pg = mesh.get_group(mesh_dim_idx)
            ws = mesh.size(mesh_dim_idx)
            rk = mesh.get_local_rank(mesh_dim_idx)
            return pg, ws, rk, placement.dim
    raise ValueError(f"No Shard placement found in DTensor with placements={p.placements}")


class DistributedMuon(Optimizer):
    """Muon optimizer with mega-batched Newton-Schulz for FSDP2-sharded params.

    Same-shape FSDP2-sharded 2D parameters are processed together: one batched
    all-gather, one batched NS, then local scatter. This reduces NCCL calls from
    O(num_params) to O(num_unique_shapes).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= float(momentum):
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")

        defaults: dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if not _is_muon_eligible_ndim(p):
                    raise ValueError(
                        "DistributedMuon supports only 2D and 3D parameters; "
                        f"got param with shape {tuple(p.size())}. Route 1D/4D+ "
                        "params (biases, norms, conv weights) to AdamW via "
                        "split_muon_adamw_params."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]

            group_config = {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "ns_steps": ns_steps,
                "eps": eps,
                "adjust_lr_fn": adjust_lr_fn,
            }

            # Classify ALL params upfront. Group FSDP_GATHER_2D by global shape.
            # CRITICAL: every rank must issue the same collective calls in the same
            # order, so we include ALL params (even grad=None) in the grouping and
            # skip the actual update for grad=None params inside the batch.
            fsdp_2d_groups: dict[tuple, list[Tensor]] = defaultdict(list)
            other_params: list[tuple[Tensor, str]] = []

            for p in group["params"]:
                kind = _classify_param(p)
                if kind == _KIND_FSDP_GATHER_2D:
                    global_shape = tuple(p.shape)  # DTensor .shape = global, same on all ranks
                    key = (global_shape, str(p.dtype))
                    fsdp_2d_groups[key].append(p)
                else:
                    if p.grad is None:
                        continue
                    if torch.is_complex(p):
                        raise RuntimeError("DistributedMuon does not support complex parameters")
                    if p.grad.is_sparse:
                        raise RuntimeError("DistributedMuon does not support sparse gradients")
                    other_params.append((p, kind))

            # --- Mega-batch path for FSDP_GATHER_2D ---
            # Sort by key to guarantee identical ordering across all ranks.
            for _key in sorted(fsdp_2d_groups.keys()):
                self._step_megabatch(fsdp_2d_groups[_key], group_config)

            # --- Per-param fallback for remaining kinds ---
            # Operate on local tensors to avoid DTensor dispatch issues
            # (torch.compile may produce non-DTensor grads that conflict with
            # DTensor in-place ops).
            for p, kind in other_params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]

                buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
                grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad

                buf_local.lerp_(grad_local, 1 - momentum)
                if nesterov:
                    update_local = grad_local.lerp(buf_local, momentum)
                else:
                    update_local = buf_local.clone()

                # Compute ortho on local tensors
                if kind == _KIND_LOCAL or kind == _KIND_MOE_LOCAL_3D:
                    ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)
                elif kind == _KIND_MOE_GATHER_3D:
                    assert isinstance(p, DTensor)
                    # Need full tensor for NS — gather, compute, slice back
                    full_update = _full_grad(
                        DTensor.from_local(update_local, device_mesh=p.device_mesh, placements=p.placements, run_check=False)
                    )
                    full_ortho = batched_newton_schulz(full_update, ns_coefficients, ns_steps, eps)
                    # Take this rank's local shard back
                    pg, ws, rk, sdim = _get_dtensor_shard_info(p)
                    global_size = p.shape[sdim]
                    chunk_floor = global_size // ws
                    rem = global_size % ws
                    start = rk * chunk_floor + min(rk, rem)
                    local_size = update_local.shape[sdim]
                    ortho_local = full_ortho.narrow(sdim, start, local_size).contiguous()
                else:
                    ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)

                lr_shape = p.shape[-2:] if p.ndim >= 2 else p.shape
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

                p_local = p.to_local() if isinstance(p, DTensor) else p
                if weight_decay != 0.0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)

        return loss

    def _step_megabatch(self, params: list[Tensor], config: dict) -> None:
        """Process a batch of same-shape FSDP_GATHER_2D params with batched comms + NS."""
        N = len(params)
        if N == 0:
            return

        momentum = config["momentum"]
        nesterov = config["nesterov"]
        ns_coefficients = config["ns_coefficients"]
        ns_steps = config["ns_steps"]
        eps = config["eps"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        adjust_lr_fn = config["adjust_lr_fn"]

        pg, world_size, rank, shard_dim = _get_dtensor_shard_info(params[0])

        # Compute adjusted_lr once (all params have same global shape)
        lr_shape = params[0].shape[-2:]
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

        # Process in sub-batches to bound peak memory
        for batch_start in range(0, N, _MEGABATCH_MAX_GROUP_SIZE):
            batch_params = params[batch_start:batch_start + _MEGABATCH_MAX_GROUP_SIZE]
            self._step_megabatch_chunk(
                batch_params, momentum, nesterov, ns_coefficients, ns_steps,
                eps, lr, weight_decay, adjusted_lr, pg, world_size, rank, shard_dim,
            )

    def _step_megabatch_chunk(
        self,
        params: list[Tensor],
        momentum: float,
        nesterov: bool,
        ns_coefficients: tuple,
        ns_steps: int,
        eps: float,
        lr: float,
        weight_decay: float,
        adjusted_lr: float,
        pg: Any,
        world_size: int,
        rank: int,
        shard_dim: int,
    ) -> None:
        """Core mega-batch logic for a chunk of same-shape params."""
        # Phase 1: Momentum update on LOCAL tensors (no DTensor ops).
        # This avoids issues with torch.compile which may produce non-DTensor grads.
        # Params with grad=None contribute zeros (they must still participate in
        # the collective to keep all ranks in sync).
        local_updates: list[Tensor] = []
        has_grad: list[bool] = []
        for p in params:
            if p.grad is None:
                # No grad — contribute zeros to the collective
                p_local = p.to_local() if isinstance(p, DTensor) else p
                local_updates.append(torch.zeros_like(p_local))
                has_grad.append(False)
                continue

            has_grad.append(True)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]

            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad

            buf_local.lerp_(grad_local, 1 - momentum)
            if nesterov:
                update_local = grad_local.lerp(buf_local, momentum)
            else:
                update_local = buf_local.clone()

            local_updates.append(update_local)

        # Phase 2: Stack and batched all-gather
        stacked_local = torch.stack(local_updates, dim=0)  # [N, local_M, K]
        del local_updates

        gather_dim = shard_dim + 1  # +1 for the batch dim we prepended
        original_local_size = stacked_local.size(gather_dim)

        # FSDP2 contiguous chunking may give different ranks different local sizes
        # (ceil vs floor when global_dim % world_size != 0). dist.all_gather
        # requires uniform sizes. Pad to max_local_size before gathering.
        global_dim_size = params[0].shape[shard_dim]  # DTensor .shape = global
        max_local_size = (global_dim_size + world_size - 1) // world_size
        needs_padding = (max_local_size != original_local_size)

        if needs_padding:
            pad_amount = max_local_size - original_local_size
            ndim = stacked_local.ndim
            pad_spec = [0] * (2 * ndim)
            pad_idx = 2 * (ndim - 1 - gather_dim)
            pad_spec[pad_idx + 1] = pad_amount
            stacked_local = torch.nn.functional.pad(stacked_local, pad_spec)

        gather_list = [torch.empty_like(stacked_local) for _ in range(world_size)]
        dist.all_gather(gather_list, stacked_local.contiguous(), group=pg)
        del stacked_local

        # Reconstruct full global tensor, stripping per-rank padding if needed.
        remainder = global_dim_size % world_size
        if remainder == 0:
            stacked_full = torch.cat(gather_list, dim=gather_dim)
        else:
            real_chunks = []
            for r in range(world_size):
                real_size = max_local_size if r < remainder else (global_dim_size // world_size)
                real_chunks.append(gather_list[r].narrow(gather_dim, 0, real_size))
            stacked_full = torch.cat(real_chunks, dim=gather_dim)
        del gather_list

        # Phase 3: Batched Newton-Schulz
        stacked_ortho = batched_newton_schulz(stacked_full, ns_coefficients, ns_steps, eps)
        del stacked_full

        # Phase 4: Local scatter + apply update
        chunk_floor = global_dim_size // world_size
        shard_start = rank * chunk_floor + min(rank, remainder)
        local_ortho_batch = stacked_ortho.narrow(
            gather_dim, shard_start, original_local_size
        ).contiguous()
        del stacked_ortho

        for i, p in enumerate(params):
            if not has_grad[i]:
                continue
            ortho_local = local_ortho_batch[i]
            p_local = p.to_local() if isinstance(p, DTensor) else p

            if weight_decay != 0.0:
                p_local.mul_(1 - lr * weight_decay)

            p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)

    @staticmethod
    def _compute_ortho(
        update: Tensor,
        kind: str,
        ns_coefficients: tuple[float, float, float],
        ns_steps: int,
        eps: float,
    ) -> Tensor:
        """Run Newton-Schulz on ``update`` according to its layout kind."""
        if kind == _KIND_LOCAL:
            return batched_newton_schulz(update, ns_coefficients, ns_steps, eps)

        if kind == _KIND_FSDP_GATHER_2D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)

        if kind == _KIND_MOE_LOCAL_3D:
            assert isinstance(update, DTensor)
            local = update._local_tensor
            local_ortho = batched_newton_schulz(local, ns_coefficients, ns_steps, eps)
            return DTensor.from_local(
                local_ortho,
                device_mesh=update.device_mesh,
                placements=update.placements,
                run_check=False,
            )

        if kind == _KIND_MOE_GATHER_3D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)

        raise ValueError(f"Unknown DistributedMuon kind: {kind!r}")

class CombinedOptimizer(Optimizer):
    """Drive several inner optimizers (Muon + AdamW) as a single optimizer.

    Unlike the upstream version this exposes a *standard* flat
    ``state_dict()`` / ``load_state_dict()`` over the concatenated param
    groups, so it serializes through lerobot's optimizer-state safetensors
    path exactly like a plain ``torch.optim`` optimizer. Parameter indices
    are remapped to each inner optimizer when loading.
    """

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one inner optimizer.")
        self.optimizers: list[Optimizer] = list(optimizers)
        self.defaults: dict[str, Any] = {}
        # Bookkeeping attributes normally created by Optimizer.__init__; torch's
        # profiler and hook registry read these without guarding.
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        self._optimizer_state_dict_pre_hooks = {}
        self._optimizer_state_dict_post_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}

    @property
    def param_groups(self):
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value) -> None:
        # LR schedulers mutate the shared param-group dicts in place;
        # wholesale reassignment is a no-op (mirrors upstream).
        pass

    @property
    def state(self):
        merged: dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Flat state dict over the concatenated groups, torch-style."""
        param_mappings: dict[int, int] = {}
        start_index = 0

        def pack_group(group: dict[str, Any]) -> dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {
                    id(p): i + start_index
                    for i, p in enumerate(group["params"])
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        packed_groups = [pack_group(g) for g in self.param_groups]
        # The merged state view is keyed by parameter objects; remap to indices.
        saved_state = {
            (param_mappings[id(k)] if id(k) in param_mappings else k): v
            for k, v in self.state.items()
        }
        return {"state": saved_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Distribute a flat state dict back into the inner optimizers."""
        saved_groups = state_dict["param_groups"]
        groups = self.param_groups
        if len(groups) != len(saved_groups):
            raise ValueError(
                f"Loaded state dict has {len(saved_groups)} param groups; "
                f"optimizer was built with {len(groups)}."
            )

        # Map flat indices back to parameters, sliced per inner optimizer. Each
        # child Optimizer owns its own index namespace, so the second child must
        # restart at zero rather than receive the global ids from `state_dict()`.
        child_groups: list[list[dict[str, Any]]] = [[] for _ in self.optimizers]
        group_owner: list[int] = []
        for child_id, opt in enumerate(self.optimizers):
            for _group in opt.param_groups:
                group_owner.append(child_id)

        global_to_child: list[tuple[int, int, int]] = []  # (global_idx, child_id, local_idx)
        child_local_idx = [0] * len(self.optimizers)
        global_idx = 0
        for gi, (group, saved_group) in enumerate(zip(groups, saved_groups, strict=True)):
            if len(group["params"]) != len(saved_group["params"]):
                raise ValueError(
                    f"Param group {gi} size mismatch: {len(saved_group['params'])} saved vs "
                    f"{len(group['params'])} current."
                )
            child_id = group_owner[gi]
            local_ids = list(range(child_local_idx[child_id], child_local_idx[child_id] + len(group["params"])))
            renumbered = dict(saved_group)
            renumbered["params"] = local_ids
            child_groups[child_id].append(renumbered)
            for local_idx in local_ids:
                global_to_child.append((global_idx, child_id, local_idx))
                global_idx += 1
            child_local_idx[child_id] += len(local_ids)

        for child_id, opt in enumerate(self.optimizers):
            child_state = {}
            for gidx, cid, local_idx in global_to_child:
                if cid == child_id and gidx in state_dict["state"]:
                    child_state[local_idx] = state_dict["state"][gidx]
            opt.load_state_dict({"state": child_state, "param_groups": child_groups[child_id]})


def is_muon_optimizer(optimizer: Any) -> bool:
    """Whether ``optimizer`` (or any wrapped/dict-nested one) contains Muon."""
    from collections.abc import Mapping

    seen = set()
    stack = [optimizer]
    while stack:
        opt = stack.pop()
        if id(opt) in seen:
            continue
        seen.add(id(opt))
        if isinstance(opt, DistributedMuon):
            return True
        if isinstance(opt, CombinedOptimizer):
            stack.extend(opt.optimizers)
        elif isinstance(opt, Mapping):
            stack.extend(opt.values())
        else:
            inner = getattr(opt, "optimizer", None)  # accelerate wrappers
            if inner is not None and inner is not opt:
                stack.append(inner)
    return False
