#!/usr/bin/env python3
"""
Distributed Training Utilities for π*₀.₆ RECAP

Supports:
- FSDP (Fully Sharded Data Parallel)
- DDP (Distributed Data Parallel)
- Checkpoint sharding and consolidation
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class DistributedManager:
    """Manage distributed training setup"""
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
    ):
        self.backend = backend
        self.init_method = init_method or "env://"
        self._initialized = False
        
    def initialize(self):
        """Initialize distributed process group"""
        if self._initialized:
            return
        
        if "RANK" not in os.environ:
            # Not in distributed mode
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            return
        
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
        )
        
        torch.cuda.set_device(self.local_rank)
        self._initialized = True
        
    def cleanup(self):
        """Cleanup distributed process group"""
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if current process is main"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronization barrier"""
        if self.world_size > 1:
            dist.barrier()
    
    def print_on_main(self, *args, **kwargs):
        """Print only on main process"""
        if self.is_main_process():
            print(*args, **kwargs)


def setup_fsdp(
    model: torch.nn.Module,
    device_id: int,
    mixed_precision: str = "bf16",
    strategy: str = "full_shard",
) -> FSDP:
    """
    Setup FSDP for model
    
    Args:
        model: PyTorch model
        device_id: GPU device id
        mixed_precision: bf16, fp16, or fp32
        strategy: full_shard, shard_grad_op, or no_shard
    
    Returns:
        FSDP wrapped model
    """
    # Mixed precision policy
    if mixed_precision == "bf16":
        mp_policy = torch.bfloat16
    elif mixed_precision == "fp16":
        mp_policy = torch.float16
    else:
        mp_policy = torch.float32
    
    # Strategy
    sharding_strategy = {
        "full_shard": "FULL_SHARD",
        "shard_grad_op": "SHARD_GRAD_OP",
        "no_shard": "NO_SHARD",
    }.get(strategy, "FULL_SHARD")
    
    # Wrap model
    fsdp_model = FSDP(
        model,
        device_id=device_id,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        limit_all_gathers=True,
    )
    
    return fsdp_model


def save_checkpoint_sharded(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    path: str,
    distributed_manager: Optional[DistributedManager] = None,
):
    """
    Save sharded checkpoint (for FSDP)
    
    Each rank saves its own shard
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
    }
    
    # Handle FSDP state dict
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            checkpoint['model_state'] = model.state_dict()
    else:
        checkpoint['model_state'] = model.state_dict()
    
    torch.save(checkpoint, path)
    
    if distributed_manager and distributed_manager.is_main_process():
        print(f"Saved sharded checkpoint to {path}")


def load_checkpoint_sharded(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: str,
) -> int:
    """Load sharded checkpoint"""
    checkpoint = torch.load(path)
    
    # Handle FSDP state dict
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and checkpoint.get('scheduler_state'):
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return checkpoint['step']


def consolidate_checkpoint(
    input_dir: str,
    output_path: str,
    distributed_manager: DistributedManager,
):
    """
    Consolidate sharded checkpoint into single file
    
    Must be called on all ranks with FSDP model
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    
    # This is a placeholder - actual implementation depends on
    # how checkpoint was saved and model structure
    pass
