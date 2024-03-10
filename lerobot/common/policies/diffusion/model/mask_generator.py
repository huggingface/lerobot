import torch

from lerobot.common.policies.diffusion.model.module_attr_mixin import ModuleAttrMixin


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        # obs mask setup
        max_n_obs_steps=2,
        fix_obs_steps=True,
        # action mask
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape  # noqa: N806
        assert (self.action_dim + self.obs_dim) == D

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps + 1, size=(B,), generator=rng, device=device
            )

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (obs_steps > steps.T).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device)
            )
            action_mask = (action_steps > steps.T).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask

        return mask
