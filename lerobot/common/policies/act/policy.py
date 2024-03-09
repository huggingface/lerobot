import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms

from lerobot.common.policies.act.detr_vae import build


def build_act_model_and_optimizer(cfg):
    model = build(cfg)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return model, optimizer


# def build_CNNMLP_model_and_optimizer(cfg):
#     parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()

#     for k, v in cfg.items():
#         setattr(args, k, v)

#     model = build_CNNMLP_model(args)
#     model.cuda()

#     param_dicts = [
#         {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
#         {
#             "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#             "lr": args.lr_backbone,
#         },
#     ]
#     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
#                                   weight_decay=args.weight_decay)

#     return model, optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ActionChunkingTransformerPolicy(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model, self.optimizer = build_act_model_and_optimizer(cfg)
        self.kl_weight = self.cfg.kl_weight
        logging.info(f"KL Weight {self.kl_weight}")

    def update(self, replay_buffer, step):
        del step

        start_time = time.time()

        self.train()

        num_slices = self.cfg.batch_size
        batch_size = self.cfg.horizon * num_slices

        assert batch_size % self.cfg.horizon == 0
        assert batch_size % num_slices == 0

        def process_batch(batch, horizon, num_slices):
            # trajectory t = 64, horizon h = 16
            # (t h) ... -> t h ...
            batch = batch.reshape(num_slices, horizon)

            image = batch["observation", "image", "top"]
            image = image[:, 0]  # first observation t=0
            # batch, num_cam, channel, height, width
            image = image.unsqueeze(1)
            assert image.ndim == 5
            image = image.float()

            state = batch["observation", "state"]
            state = state[:, 0]  # first observation t=0
            # batch, qpos_dim
            assert state.ndim == 2

            action = batch["action"]
            # batch, seq, action_dim
            assert action.ndim == 3
            assert action.shape[1] == horizon

            if self.cfg.n_obs_steps > 1:
                raise NotImplementedError()
                # # keep first n observations of the slice corresponding to t=[-1,0]
                # image = image[:, : self.cfg.n_obs_steps]
                # state = state[:, : self.cfg.n_obs_steps]

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

        loss = self.compute_loss(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.lr_scheduler.step()

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            # "lr": self.lr_scheduler.get_last_lr()[0],
            "lr": self.cfg.lr,
            "data_s": data_s,
            "update_s": time.time() - start_time,
        }

        return info

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)

    def compute_loss(self, batch):
        loss_dict = self._forward(
            qpos=batch["obs"]["agent_pos"],
            image=batch["obs"]["image"],
            actions=batch["action"],
        )
        loss = loss_dict["loss"]
        return loss

    @torch.no_grad()
    def forward(self, observation, step_count):
        # TODO(rcadene): remove unused step_count
        del step_count

        self.eval()

        # TODO(rcadene): remove unsqueeze hack to add bsize=1
        observation["image"] = observation["image"].unsqueeze(0)
        observation["state"] = observation["state"].unsqueeze(0)

        obs_dict = {
            "image": observation["image"],
            "agent_pos": observation["state"],
        }
        action = self._forward(qpos=obs_dict["agent_pos"], image=obs_dict["image"])
        return action

    def _forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        is_train_mode = actions is not None
        if is_train_mode:  # training time
            actions = actions[:, : self.model.num_queries]
            if is_pad is not None:
                is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = {}
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = all_l1.mean() if is_pad is None else (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:
            action, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return action


# class CNNMLPPolicy(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         model, optimizer = build_CNNMLP_model_and_optimizer(cfg)
#         self.model = model # decoder
#         self.optimizer = optimizer

#     def __call__(self, qpos, image, actions=None, is_pad=None):
#         env_state = None # TODO
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#         image = normalize(image)
#         if actions is not None: # training time
#             actions = actions[:, 0]
#             a_hat = self.model(qpos, image, env_state, actions)
#             mse = F.mse_loss(actions, a_hat)
#             loss_dict = dict()
#             loss_dict['mse'] = mse
#             loss_dict['loss'] = loss_dict['mse']
#             return loss_dict
#         else: # inference time
#             a_hat = self.model(qpos, image, env_state) # no action, sample from prior
#             return a_hat

#     def configure_optimizers(self):
#         return self.optimizer

# def kl_divergence(mu, logvar):
#     batch_size = mu.size(0)
#     assert batch_size != 0
#     if mu.data.ndimension() == 4:
#         mu = mu.view(mu.size(0), mu.size(1))
#     if logvar.data.ndimension() == 4:
#         logvar = logvar.view(logvar.size(0), logvar.size(1))

#     klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     total_kld = klds.sum(1).mean(0, True)
#     dimension_wise_kld = klds.mean(0)
#     mean_kld = klds.mean(1).mean(0, True)

#     return total_kld, dimension_wise_kld, mean_kld
