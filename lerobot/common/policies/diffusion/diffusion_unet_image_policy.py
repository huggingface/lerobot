"""Code from the original diffusion policy project.

Notes on how to load a checkpoint from the original repository:

In the original repository, run the eval and use a breakpoint to extract the policy weights.

```
torch.save(policy.state_dict(), "weights.pt")
```

In this repository, add a breakpoint somewhere after creating an equivalent policy and load in the weights:

```
loaded = torch.load("weights.pt")
aligned = {}
their_prefix = "obs_encoder.obs_nets.image.backbone"
our_prefix = "obs_encoder.key_model_map.image.backbone"
aligned.update({our_prefix + k.removeprefix(their_prefix): v for k, v in loaded.items() if k.startswith(their_prefix)})
their_prefix = "obs_encoder.obs_nets.image.pool"
our_prefix = "obs_encoder.key_model_map.image.pool"
aligned.update({our_prefix + k.removeprefix(their_prefix): v for k, v in loaded.items() if k.startswith(their_prefix)})
their_prefix = "obs_encoder.obs_nets.image.nets.3"
our_prefix = "obs_encoder.key_model_map.image.out"
aligned.update({our_prefix + k.removeprefix(their_prefix): v for k, v in loaded.items() if k.startswith(their_prefix)})
aligned.update({k: v for k, v in loaded.items() if k.startswith('model.')})
# Note: here you are loading into the ema model.
missing_keys, unexpected_keys = policy.ema_diffusion.load_state_dict(aligned, strict=False)
assert all('_dummy_variable' in k for k in missing_keys)
assert len(unexpected_keys) == 0
```

Then in that same runtime you can also save the weights with the new aligned state_dict:

```
policy.save("weights.pt")
```

Now you can remove the breakpoint and extra code and load in the weights just like with any other lerobot checkpoint.

"""

from typing import Dict

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from lerobot.common.policies.diffusion.model.conditional_unet1d import ConditionalUnet1D
from lerobot.common.policies.diffusion.model.mask_generator import LowdimMaskGenerator
from lerobot.common.policies.diffusion.model.module_attr_mixin import ModuleAttrMixin
from lerobot.common.policies.diffusion.model.multi_image_obs_encoder import MultiImageObsEncoder
from lerobot.common.policies.diffusion.model.normalizer import LinearNormalizer
from lerobot.common.policies.diffusion.pytorch_utils import dict_apply


class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                # **kwargs  # TODO(rcadene): in diffusion_policy, expected to be {}
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        nobs = obs_dict
        value = next(iter(nobs.values()))
        bsize, n_obs_steps = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_dim = self.obs_feature_dim
        assert self.n_obs_steps == n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(bsize, -1)
            # empty data for action
            cond_data = torch.zeros(size=(bsize, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(bsize, n_obs_steps, -1)
            cond_data = torch.zeros(size=(bsize, horizon, action_dim + obs_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, action_dim:] = nobs_features
            cond_mask[:, :n_obs_steps, action_dim:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond
        )

        action_pred = nsample[..., :action_dim]
        # get action
        start = n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    def compute_loss(self, obs_dict, action):
        nobs = obs_dict
        nactions = action
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
