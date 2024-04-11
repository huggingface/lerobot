import einops
import torch

from lerobot.common.transforms import apply_inverse_transform


def preprocess_observation(observation, transform=None):
    # map to expected inputs for the policy
    obs = {}

    if isinstance(observation["pixels"], dict):
        imgs = {f"observation.images.{key}": img for key, img in observation["pixels"].items()}
    else:
        imgs = {"observation.image": observation["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img).float()
        # convert to (b c h w) torch format
        img = einops.rearrange(img, "b h w c -> b c h w")
        obs[imgkey] = img

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing requirement for "agent_pos"
    obs["observation.state"] = torch.from_numpy(observation["agent_pos"]).float()

    # apply same transforms as in training
    if transform is not None:
        for key in obs:
            obs[key] = torch.stack([transform({key: item})[key] for item in obs[key]])

    return obs


def postprocess_action(action, transform=None):
    action = action.to("cpu")
    # action is a batch (num_env,action_dim) instead of an item (action_dim),
    # we assume applying inverse transform on a batch works the same
    action = apply_inverse_transform({"action": action}, transform)["action"].numpy()
    assert (
        action.ndim == 2
    ), "we assume dimensions are respectively the number of parallel envs, action dimensions"
    return action
