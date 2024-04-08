import copy
import logging
import time
from collections import deque

import hydra
import torch
from torch import nn

from lerobot.common.policies.diffusion.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from lerobot.common.policies.diffusion.model.lr_scheduler import get_scheduler
from lerobot.common.policies.diffusion.model.multi_image_obs_encoder import MultiImageObsEncoder, RgbEncoder
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils import get_safe_torch_device


class DiffusionPolicy(nn.Module):
    name = "diffusion"

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
        super().__init__()
        self.cfg = cfg
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        noise_scheduler = hydra.utils.instantiate(cfg_noise_scheduler)
        rgb_model_input_shape = copy.deepcopy(shape_meta.obs.image.shape)
        if cfg_obs_encoder.crop_shape is not None:
            rgb_model_input_shape[1:] = cfg_obs_encoder.crop_shape
        rgb_model = RgbEncoder(input_shape=rgb_model_input_shape, **cfg_rgb_model)
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

        self.ema_diffusion = None
        self.ema = None
        if self.cfg.use_ema:
            self.ema_diffusion = copy.deepcopy(self.diffusion)
            self.ema = hydra.utils.instantiate(
                cfg_ema,
                model=self.ema_diffusion,
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

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.n_obs_steps),
            "observation.state": deque(maxlen=self.n_obs_steps),
            "action": deque(maxlen=self.n_action_steps),
        }

    @torch.no_grad()
    def select_action(self, batch, step):
        """
        Note: this uses the ema model weights if self.training == False, otherwise the non-ema model weights.
        """
        # TODO(rcadene): remove unused step
        del step
        assert "observation.image" in batch
        assert "observation.state" in batch
        assert len(batch) == 2

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            obs_dict = {
                "image": batch["observation.image"],
                "agent_pos": batch["observation.state"],
            }
            if self.training:
                out = self.diffusion.predict_action(obs_dict)
            else:
                out = self.ema_diffusion.predict_action(obs_dict)
            self._queues["action"].extend(out["action"].transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch, step):
        start_time = time.time()

        self.diffusion.train()

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
        missing_keys, unexpected_keys = self.load_state_dict(d, strict=False)
        if len(missing_keys) > 0:
            assert all(k.startswith("ema_diffusion.") for k in missing_keys)
            logging.warning(
                "DiffusionPolicy.load expected ema parameters in loaded state dict but none were found."
            )
        assert len(unexpected_keys) == 0
