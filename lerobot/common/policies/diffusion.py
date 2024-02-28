import copy

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class DiffusionPolicy(nn.Module):

    def __init__(
        self,
        cfg,
        cfg_noise_scheduler,
        cfg_rgb_model,
        cfg_obs_encoder,
        cfg_optimizer,
        cfg_ema,
        shape_meta: dict,
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
        self.cfg = cfg

        noise_scheduler = DDPMScheduler(**cfg_noise_scheduler)
        rgb_model = get_resnet(**cfg_rgb_model)
        obs_encoder = MultiImageObsEncoder(
            rgb_model=rgb_model,
            **cfg_obs_encoder,
        )

        self.diffusion = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            # parameters passed to step
            **kwargs,
        )

        self.device = torch.device("cuda")
        self.diffusion.cuda()

        self.ema = None
        if self.cfg.use_ema:
            self.ema = hydra.utils.instantiate(
                cfg_ema,
                model=copy.deepcopy(self.diffusion),
            )

        self.optimizer = hydra.utils.instantiate(
            cfg_optimizer,
            params=self.diffusion.parameters(),
        )

        # TODO(rcadene): modify lr scheduler so that it doesnt depend on epochs but steps
        self.global_step = 0

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=cfg.offline_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

    @torch.no_grad()
    def forward(self, observation, step_count):
        # TODO(rcadene): remove unused step_count
        del step_count

        obs_dict = {
            # c h w -> b t c h w (b=1, t=1)
            "image": observation["image"][None, None, ...],
            "agent_pos": observation["state"][None, None, ...],
        }
        out = self.diffusion.predict_action(obs_dict)

        # TODO(rcadene): add possibility to return >1 timestemps
        FIRST_ACTION = 0
        action = out["action"].squeeze(0)[FIRST_ACTION]
        return action

    def update(self, replay_buffer, step):
        self.diffusion.train()

        num_slices = self.cfg.batch_size
        batch_size = self.cfg.horizon * num_slices

        assert batch_size % self.cfg.horizon == 0
        assert batch_size % num_slices == 0

        def process_batch(batch, horizon, num_slices):
            # trajectory t = 64, horizon h = 16
            # (t h) ... -> t h ...
            batch = batch.reshape(num_slices, horizon)  # .transpose(1, 0).contiguous()

            out = {
                "obs": {
                    "image": batch["observation", "image"].to(self.device),
                    "agent_pos": batch["observation", "state"].to(self.device),
                },
                "action": batch["action"].to(self.device),
            }
            return out

        if self.cfg.balanced_sampling:
            batch = replay_buffer.sample(batch_size)
        else:
            batch = replay_buffer.sample()
        batch = process_batch(batch, self.cfg.horizon, num_slices)

        loss = self.diffusion.compute_loss(batch)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.step(self.diffusion)

        metrics = {
            "total_loss": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

        # TODO(rcadene): remove hardcoding
        # in diffusion_policy, len(dataloader) is 168 for a batch_size of 64
        if step % 168 == 0:
            self.global_step += 1

        return metrics

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)
