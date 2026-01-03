"""
UMI-style relative actions with per-timestep normalization.

Two modes supported:
  Mode 1: Relative actions only (use_relative_state=False)
    - Actions converted to relative, state stays absolute
  Mode 2: Relative actions + state (use_relative_state=True, full UMI)
    - Both actions and state converted to relative

Per-timestep normalization (TRI LBM / BEHAVIOR style):
  Training: action_norm[t] = (action_rel[t] - mean[t]) / std[t]
  Inference: action_rel[t] = action_norm[t] * std[t] + mean[t]
"""

import torch
from pathlib import Path


class PerTimestepNormalizer:
    """Per-timestep normalization using precomputed dataset statistics."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
        self._cache = {}  # Cache for device/dtype converted tensors
    
    def _get_stats(self, device, dtype):
        """Get cached stats for device/dtype, or create and cache them."""
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = (
                self.mean.to(device, dtype),
                self.std.to(device, dtype),
            )
        return self._cache[key]
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._get_stats(x.device, x.dtype)
        if x.dim() == 3 and mean.dim() == 2:
            mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        return (x - mean) / (std + self.eps)
    
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._get_stats(x.device, x.dtype)
        if x.dim() == 3 and mean.dim() == 2:
            mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        return x * (std + self.eps) + mean
    
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps}, path)
    
    @classmethod
    def load(cls, path: Path | str) -> "PerTimestepNormalizer":
        data = torch.load(path, weights_only=True, map_location="cpu")
        return cls(data["mean"], data["std"], data.get("eps", 1e-8))


def compute_relative_action_stats(
    dataloader,
    state_key: str = "observation.state",
    num_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-timestep mean/std from relative actions."""
    all_rel = []
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        action, state = batch["action"], batch[state_key]
        current_pos = state[:, -1, :] if state.dim() == 3 else state
        min_dim = min(action.shape[-1], current_pos.shape[-1])
        rel = action.clone()
        rel[..., :min_dim] -= current_pos[:, None, :min_dim]
        all_rel.append(rel)
    
    all_rel = torch.cat(all_rel, dim=0)
    return all_rel.mean(dim=0), all_rel.std(dim=0).clamp(min=1e-6)


def convert_to_relative(
    batch: dict,
    state_key: str = "observation.state",
    convert_state: bool = True,
) -> dict:
    """
    Convert actions (and optionally state) to relative.
    
    Args:
        batch: Training batch with "action" and state_key
        state_key: Key for observation state
        convert_state: If True, also convert state to relative (full UMI mode)
    """
    if "action" not in batch or state_key not in batch:
        return batch
    
    action = batch["action"]
    state = batch[state_key]
    batch = batch.copy()
    
    # Get current position as reference
    current_pos = state[:, -1, :] if state.dim() == 3 else state
    
    # Convert state if requested
    if convert_state:
        if state.dim() == 3:
            batch[state_key] = state - current_pos[:, None, :]
        else:
            batch[state_key] = torch.zeros_like(state)
    
    # Convert actions to relative
    min_dim = min(action.shape[-1], current_pos.shape[-1])
    rel_action = action.clone()
    rel_action[..., :min_dim] -= current_pos[:, None, :min_dim]
    batch["action"] = rel_action
    
    return batch


# Backward compatibility alias
convert_to_relative_actions = convert_to_relative


def convert_state_to_relative(state: torch.Tensor) -> torch.Tensor:
    """Convert state to relative (for inference with use_relative_state=True)."""
    if state.dim() == 1:
        return torch.zeros_like(state)
    current_pos = state[-1, :] if state.dim() == 2 else state[:, -1, :]
    if state.dim() == 2:
        return state - current_pos[None, :]
    return state - current_pos[:, None, :]


def convert_from_relative_actions(
    relative_actions: torch.Tensor,
    current_pos: torch.Tensor,
) -> torch.Tensor:
    """Convert relative actions back to absolute for robot execution."""
    current_pos = current_pos.to(relative_actions.device, relative_actions.dtype)
    min_dim = min(relative_actions.shape[-1], current_pos.shape[-1])
    absolute = relative_actions.clone()
    
    if relative_actions.dim() == 2:
        absolute[..., :min_dim] += current_pos[:min_dim]
    elif relative_actions.dim() == 3:
        absolute[..., :min_dim] += current_pos[None, None, :min_dim]
    else:
        absolute[..., :min_dim] += current_pos[:min_dim]
    
    return absolute


def convert_from_relative_actions_dict(
    relative_actions: dict[str, float],
    current_pos: dict[str, float],
) -> dict[str, float]:
    """Convert relative actions back to absolute (dict version for inference)."""
    return {k: v + current_pos.get(k, 0.0) for k, v in relative_actions.items()}
