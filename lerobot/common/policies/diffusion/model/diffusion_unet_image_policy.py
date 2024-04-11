import einops
import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.policies.diffusion.model.conditional_unet1d import ConditionalUnet1D
from lerobot.common.policies.diffusion.model.rgb_encoder import RgbEncoder
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class DiffusionUnetImagePolicy(nn.Module):
    """
    TODO(now): Add DDIM scheduler.

    Changes:  TODO(now)
      - Use single image encoder for now instead of generic obs_encoder. We may generalize again when/if
        needed. Code for a general observation encoder can be found at:
        https://github.com/huggingface/lerobot/blob/920e0d118b493e4cc3058a9b1b764f38ae145d8e/lerobot/common/policies/diffusion/model/multi_image_obs_encoder.py
      - Uses the observation as global conditioning for the Unet by default.
      - Does not do any inpainting (which would be applicable if the observation were not used to condition
        the Unet).
    """

    def __init__(
        self,
        cfg,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        film_scale_modulation=True,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.rgb_encoder = RgbEncoder(input_shape=shape_meta.obs.image.shape, **cfg.rgb_encoder)

        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=(action_dim + self.rgb_encoder.feature_dim) * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            film_scale_modulation=film_scale_modulation,
        )

        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps

        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        inpainting_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
    ):
        model = self.unet
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
            trajectory[inpainting_mask] = condition_data[inpainting_mask]

            # 2. predict model output
            model_output = model(
                trajectory,
                torch.full(trajectory.shape[:1], t, dtype=torch.long, device=trajectory.device),
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[inpainting_mask] = condition_data[inpainting_mask]

        return trajectory

    def predict_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
        }
        """
        assert set(batch).issuperset({"observation.state", "observation.image"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.n_obs_steps
        assert self.n_obs_steps == n_obs_steps

        # build input
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)
        # reshape back to B, Do
        # empty data for action
        cond_data = torch.zeros(size=(batch_size, self.horizon, self.action_dim), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)

        # `horizon` steps worth of actions (from the first observation).
        action = nsample[..., : self.action_dim]
        # Extract `n_action_steps` steps worth of action (from the current observation).
        start = n_obs_steps - 1
        end = start + self.n_action_steps
        action = action[:, start:end]

        return action

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)  # TODO(now) maybe this is (B, horizon, 1)
        }
        """
        assert set(batch).issuperset({"observation.state", "observation.image", "action", "action_is_pad"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.horizon
        assert n_obs_steps == self.n_obs_steps
        assert self.n_obs_steps == n_obs_steps

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = batch["action"]
        cond_data = trajectory

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Apply inpainting. TODO(now): implement?
        inpainting_mask = torch.zeros_like(trajectory, dtype=bool)
        noisy_trajectory[inpainting_mask] = cond_data[inpainting_mask]

        # Predict the noise residual
        pred = self.unet(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~inpainting_mask)

        if "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound[:, :, None].type(loss.dtype)

        return loss.mean()
