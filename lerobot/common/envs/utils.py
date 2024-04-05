import einops
import torch

from lerobot.common.transforms import apply_inverse_transform


def preprocess_observation(observation, transform=None):
    # map to expected inputs for the policy
    obs = {
        "observation.image": torch.from_numpy(observation["pixels"]).float(),
        "observation.state": torch.from_numpy(observation["agent_pos"]).float(),
    }
    # convert to (b c h w) torch format
    obs["observation.image"] = einops.rearrange(obs["observation.image"], "b h w c -> b c h w")

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
