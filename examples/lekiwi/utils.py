import rerun as rr
import numpy as np
import torch

def display_data(observation, arm_action, base_action):
    """Display all data in Rerun."""
    # Log observations
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation_{obs}", rr.Scalars(val))
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if len(val.shape) == 1:  # 1D array - log as individual scalars
                for i, v in enumerate(val):
                    rr.log(f"observation_{obs}_{i}", rr.Scalars(v))
            else:  # 2D or 3D array - log as image
                rr.log(f"observation_{obs}", rr.Image(val), static=True)

    # Log arm actions
    for act, val in arm_action.items():
        if isinstance(val, float):
            rr.log(f"action_{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action_{act}_{i}", rr.Scalars(v))

    # Log base actions
    for act, val in base_action.items():
        if isinstance(val, float):
            rr.log(f"base_action_{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"base_action_{act}_{i}", rr.Scalars(v))

