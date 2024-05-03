import einops
import numpy as np
import torch
from torch import Tensor


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}

    if isinstance(observations["pixels"], dict):
        imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
    else:
        imgs = {"observation.image": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)

        # sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255

        return_observations[imgkey] = img

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()

    return return_observations
