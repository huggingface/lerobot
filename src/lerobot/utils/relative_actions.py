"""
UMI-style relative action and state utilities.

Implements chunk-relative representation from the UMI paper:
"Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots"

For each inference step:
- Actions are relative to current position at chunk start (t0)
- State history is relative to current position (provides velocity info)

Training:
  action_relative[t] = action_absolute[t] - position_at_t0
  state_relative[t] = state_absolute[t] - current_position

Inference:
  action_absolute[t] = action_relative[t] + current_position
"""

import torch


def convert_to_relative(batch: dict, state_key: str = "observation.state") -> dict:
    """
    Convert absolute actions AND state to chunk-relative (UMI-style) for training.
    
    Following UMI paper PD2.1 and PD2.2:
    - Actions become relative to current position
    - State history becomes relative to current position (provides velocity info)
    
    Args:
        batch: Training batch containing:
            - "action": (batch_size, chunk_size, action_dim) absolute action targets
            - state_key: (batch_size, [n_obs_steps,] state_dim) observation state
        state_key: Key for the observation state in the batch
        
    Returns:
        Modified batch with relative actions and state
    """
    if "action" not in batch or state_key not in batch:
        return batch
    
    action = batch["action"]
    state = batch[state_key]
    
    batch = batch.copy()
    
    # Get current position (reference for relative conversion)
    # State shape: (batch, state_dim) or (batch, n_obs_steps, state_dim)
    if state.dim() == 3:
        current_pos = state[:, -1, :]  # (batch, state_dim)
        
        # Convert state history to relative (each timestep relative to current)
        # This gives velocity-like information to the policy
        relative_state = state.clone()
        relative_state = state - current_pos[:, None, :]
        batch[state_key] = relative_state
    else:
        current_pos = state  # (batch, state_dim)
        # Single timestep state becomes zeros (relative to itself)
        batch[state_key] = torch.zeros_like(state)
    
    # Convert actions to relative
    action_dim = action.shape[-1]
    state_dim = current_pos.shape[-1]
    min_dim = min(action_dim, state_dim)
    
    relative_action = action.clone()
    relative_action[..., :min_dim] = action[..., :min_dim] - current_pos[:, None, :min_dim]
    batch["action"] = relative_action
    
    return batch


# Alias for backward compatibility
convert_to_relative_actions = convert_to_relative


def convert_state_to_relative(
    state: torch.Tensor,
    current_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert absolute state to relative for inference.
    
    Args:
        state: State tensor, shape (state_dim,) or (n_obs_steps, state_dim)
        current_pos: Current position to use as reference. If None, uses last timestep of state.
        
    Returns:
        Relative state tensor
    """
    if current_pos is None:
        if state.dim() >= 2:
            current_pos = state[-1, :]  # Last timestep
        else:
            current_pos = state
    
    if state.dim() == 1:
        return torch.zeros_like(state)
    elif state.dim() == 2:
        # (n_obs_steps, state_dim)
        return state - current_pos[None, :]
    else:
        # (batch, n_obs_steps, state_dim)
        return state - current_pos[:, None, :]


def convert_from_relative_actions(
    relative_actions: torch.Tensor,
    current_pos: torch.Tensor | dict[str, float],
) -> torch.Tensor:
    """
    Convert relative actions back to absolute for robot execution.
    
    Args:
        relative_actions: Predicted relative actions, shape (chunk_size, action_dim) 
                         or (batch, chunk_size, action_dim)
        current_pos: Current robot position as tensor (action_dim,) or dict of joint positions
        
    Returns:
        Absolute actions for robot execution
    """
    if isinstance(current_pos, dict):
        # Convert dict to tensor, maintaining key order
        current_pos = torch.tensor(list(current_pos.values()), dtype=relative_actions.dtype)
    
    # Ensure current_pos is on same device
    current_pos = current_pos.to(relative_actions.device)
    
    # Match dimensions
    action_dim = relative_actions.shape[-1]
    pos_dim = current_pos.shape[-1] if current_pos.dim() > 0 else len(current_pos)
    min_dim = min(action_dim, pos_dim)
    
    absolute_actions = relative_actions.clone()
    
    if relative_actions.dim() == 2:
        # Shape: (chunk_size, action_dim)
        absolute_actions[..., :min_dim] = relative_actions[..., :min_dim] + current_pos[:min_dim]
    elif relative_actions.dim() == 3:
        # Shape: (batch, chunk_size, action_dim)
        absolute_actions[..., :min_dim] = relative_actions[..., :min_dim] + current_pos[None, None, :min_dim]
    else:
        # Shape: (action_dim,)
        absolute_actions[..., :min_dim] = relative_actions[..., :min_dim] + current_pos[:min_dim]
    
    return absolute_actions


def convert_from_relative_actions_dict(
    relative_actions: dict[str, float],
    current_pos: dict[str, float],
) -> dict[str, float]:
    """
    Convert relative actions back to absolute for robot execution (dict version).
    
    Args:
        relative_actions: Predicted relative actions as dict (e.g., {"joint_1.pos": 0.1, ...})
        current_pos: Current robot position as dict (e.g., {"joint_1.pos": 45.0, ...})
        
    Returns:
        Absolute actions dict for robot execution
    """
    absolute_actions = {}
    for key, rel_value in relative_actions.items():
        if key in current_pos:
            absolute_actions[key] = rel_value + current_pos[key]
        else:
            # Key not in current position, keep as-is (shouldn't happen normally)
            absolute_actions[key] = rel_value
    return absolute_actions

