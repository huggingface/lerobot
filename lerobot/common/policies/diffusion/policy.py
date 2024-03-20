import copy
import time

import hydra
import torch

from lerobot.common.policies.abstract import AbstractPolicy
from lerobot.common.policies.diffusion.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from lerobot.common.policies.diffusion.model.lr_scheduler import get_scheduler
from lerobot.common.policies.diffusion.model.multi_image_obs_encoder import MultiImageObsEncoder
from lerobot.common.utils import get_safe_torch_device


class DiffusionPolicy(AbstractPolicy):
    def __init__(
        self,
        cfg,
        cfg_device,
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
        super().__init__(n_action_steps)
        self.cfg = cfg

        noise_scheduler = hydra.utils.instantiate(cfg_noise_scheduler)
        rgb_model = hydra.utils.instantiate(cfg_rgb_model)
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

        self.device = get_safe_torch_device(cfg_device)
        self.diffusion.to(self.device)

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
    def select_actions(self, observation, step_count):
        # TODO(rcadene): remove unused step_count
        del step_count

        obs_dict = {
            "image": observation["image"],
            "agent_pos": observation["state"],
        }
        out = self.diffusion.predict_action(obs_dict)
        action = out["action"]
        return action

    def update(self, replay_buffer, step):
        start_time = time.time()

        self.diffusion.train()

        num_slices = self.cfg.batch_size
        batch_size = self.cfg.horizon * num_slices

        assert batch_size % self.cfg.horizon == 0
        assert batch_size % num_slices == 0

        def process_batch(batch, horizon, num_slices):
            # trajectory t = 64, horizon h = 16
            # (t h) ... -> t h ...
            batch = batch.reshape(num_slices, horizon)  # .transpose(1, 0).contiguous()

            # |-1|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14| timestamps: 16
            #  |o|o|                                  observations: 2
            #  | |a|a|a|a|a|a|a|a|                    actions executed: 8
            #  |p|p|p|p|p|p|p|p|p|p|p| p| p| p| p| p| actions predicted: 16
            # note: we predict the action needed to go from t=-1 to t=0 similarly to an inverse kinematic model

            image = batch["observation", "image"]
            state = batch["observation", "state"]
            action = batch["action"]
            assert image.shape[1] == horizon
            assert state.shape[1] == horizon
            assert action.shape[1] == horizon

            if not (horizon == 16 and self.cfg.n_obs_steps == 2):
                raise NotImplementedError()

            # keep first 2 observations of the slice corresponding to t=[-1,0]
            image = image[:, : self.cfg.n_obs_steps]
            state = state[:, : self.cfg.n_obs_steps]

            out = {
                "obs": {
                    "image": image.to(self.device, non_blocking=True),
                    "agent_pos": state.to(self.device, non_blocking=True),
                },
                "action": action.to(self.device, non_blocking=True),
            }
            return out

        batch = replay_buffer.sample(batch_size)
        batch = process_batch(batch, self.cfg.horizon, num_slices)

        data_s = time.time() - start_time

        loss = self.diffusion.compute_loss(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.step(self.diffusion)

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "data_s": data_s,
            "update_s": time.time() - start_time,
        }

        # TODO(rcadene): remove hardcoding
        # in diffusion_policy, len(dataloader) is 168 for a batch_size of 64
        if step % 168 == 0:
            self.global_step += 1

        return info

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)
