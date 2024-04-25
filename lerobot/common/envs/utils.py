import einops
import torch


def preprocess_observation(observation):
    # map to expected inputs for the policy
    obs = {}

    if isinstance(observation["pixels"], dict):
        imgs = {f"observation.images.{key}": img for key, img in observation["pixels"].items()}
    else:
        imgs = {"observation.image": observation["pixels"]}

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

        obs[imgkey] = img

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing requirement for "agent_pos"
    obs["observation.state"] = torch.from_numpy(observation["agent_pos"]).float()

    return obs


def postprocess_action(action):
    action = action.to("cpu").numpy()
    assert (
        action.ndim == 2
    ), "we assume dimensions are respectively the number of parallel envs, action dimensions"
    return action
