import einops
import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.policies.diffusion.model.conditional_unet1d import ConditionalUnet1D
from lerobot.common.policies.diffusion.model.rgb_encoder import RgbEncoder
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class DiffusionUnetImagePolicy(nn.Module):
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
    def conditional_sample(self, batch_size, global_cond=None, generator=None):
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def generate_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
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

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

        # run sampling
        sample = self.conditional_sample(batch_size, global_cond=global_cond)

        # `horizon` steps worth of actions (from the first observation).
        action = sample[..., : self.action_dim]
        # Extract `n_action_steps` steps worth of actions (from the current observation).
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
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "observation.image", "action", "action_is_pad"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.horizon
        assert n_obs_steps == self.n_obs_steps
        assert self.n_obs_steps == n_obs_steps

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        global_cond = torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

        trajectory = batch["action"]

        # Forward diffusion.
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The targe is either the original trajectory, or the noise.
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = eps
        elif pred_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
