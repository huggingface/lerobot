import logging

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
    def __init__(self, cfg):
        super().__init__()
        self.model, self.optimizer = build_act_model_and_optimizer(cfg)
        self.kl_weight = cfg.kl_weight
        logging.info(f"KL Weight {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        is_train_mode = actions is not None
        if is_train_mode:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = {}
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


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
