import logging
import time
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms
from torch import nn

from lerobot.common.policies.act.detr_vae import build
from lerobot.common.policies.utils import populate_queues


def build_act_model_and_optimizer(cfg):
    model = build(cfg)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return model, optimizer


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
    name = "act"

    def __init__(self, cfg, n_obs_steps, n_action_steps):
        super().__init__()
        self.cfg = cfg
        self.n_obs_steps = n_obs_steps
        if self.n_obs_steps > 1:
            raise NotImplementedError()
        self.n_action_steps = n_action_steps
        self.model, self.optimizer = build_act_model_and_optimizer(cfg)
        self.kl_weight = self.cfg.kl_weight
        logging.info(f"KL Weight {self.kl_weight}")

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.images.top": deque(maxlen=self.n_obs_steps),
            "observation.state": deque(maxlen=self.n_obs_steps),
            "action": deque(maxlen=self.n_action_steps),
        }

    def forward(self, batch, step):
        del step

        start_time = time.time()

        self.train()

        image = batch["observation.images.top"]
        # batch, num_cam, channel, height, width
        image = image.unsqueeze(1)
        assert image.ndim == 5

        state = batch["observation.state"]
        # batch, qpos_dim
        assert state.ndim == 2

        action = batch["action"]
        # batch, seq, action_dim
        assert action.ndim == 3

        preprocessed_batch = {
            "obs": {
                "image": image,
                "agent_pos": state,
            },
            "action": action,
        }

        data_s = time.time() - start_time

        loss = self.compute_loss(preprocessed_batch)
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
    def select_action(self, batch, step):
        assert "observation.images.top" in batch
        assert "observation.state" in batch
        assert len(batch) == 2

        self._queues = populate_queues(self._queues, batch)

        # TODO(rcadene): remove unused step_count
        del step

        self.eval()

        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            if self.n_obs_steps == 1:
                # hack to remove the time dimension
                for key in batch:
                    assert batch[key].shape[1] == 1
                    batch[key] = batch[key][:, 0]

            actions = self._forward(
                # TODO(rcadene): remove unsqueeze hack to add the "number of cameras" dimension
                image=batch["observation.images.top"].unsqueeze(1),
                qpos=batch["observation.state"],
            )

            if self.cfg.temporal_agg:
                # TODO(rcadene): implement temporal aggregation
                raise NotImplementedError()
                # all_time_actions[[t], t:t+num_queries] = action
                # actions_for_curr_step = all_time_actions[:, t]
                # actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                # actions_for_curr_step = actions_for_curr_step[actions_populated]
                # k = 0.01
                # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                # exp_weights = exp_weights / exp_weights.sum()
                # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

            # act returns a sequence of `n` actions, but we consider only
            # the first `n_action_steps` actions subset
            for i in range(self.n_action_steps):
                self._queues["action"].append(actions[:, i])

        action = self._queues["action"].popleft()
        return action

    def _forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        is_training = actions is not None
        if is_training:  # training time
            actions = actions[:, : self.model.num_queries]
            if is_pad is not None:
                is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = all_l1.mean() if is_pad is None else (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict["l1"] = l1
            if self.cfg.vae:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict["kl"] = total_kld[0]
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            else:
                loss_dict["loss"] = loss_dict["l1"]
            return loss_dict
        else:
            action, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return action
