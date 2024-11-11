import torch
import einops
import numpy as np
import torch.nn as nn
from .base import AbstractSSL
from accelerate import Accelerator
from typing import Tuple, Dict, Optional
from ..transformer_encoder import TransformerEncoder, TransformerEncoderConfig
from ..ema import EMA


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L239
def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()


accelerator = Accelerator()


class DynaMoSSL(AbstractSSL):
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        window_size: int,
        feature_dim: int,
        projection_dim: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        covariance_reg_coef: float = 0.04,
        dynamics_loss_coef: float = 1.0,
        ema_beta: Optional[float] = None,  # None for SimSiam; float for EMA encoder
        beta_scheduling: bool = False,
        projector_use_ema: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        separate_single_views: bool = True,
    ):
        nn.Module.__init__(self)
        # avoid registering encoder/projector as submodules
        self.__dict__["encoder"] = encoder
        self.__dict__["projector"] = projector
        forward_dynamics_cfg = TransformerEncoderConfig(
            block_size=window_size,
            input_dim=feature_dim + projection_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            output_dim=feature_dim,
        )
        self.forward_dynamics = TransformerEncoder(forward_dynamics_cfg)
        self.forward_dynamics_optimizer = self.forward_dynamics.configure_optimizers(
            weight_decay=weight_decay,
            lr=lr,
            betas=betas,
        )
        self.forward_dynamics, self.forward_dynamics_optimizer = accelerator.prepare(
            self.forward_dynamics,
            self.forward_dynamics_optimizer,
        )
        self.covariance_reg_coef = covariance_reg_coef
        self.dynamics_loss_coef = dynamics_loss_coef
        self.ema_beta = ema_beta
        self.beta_scheduling = beta_scheduling
        self.projector_use_ema = projector_use_ema
        if self.ema_beta is not None:
            self.ema_encoder = EMA(self.encoder, self.ema_beta)
            if self.projector_use_ema:
                self.ema_projector = EMA(self.projector, self.ema_beta)
        self.separate_single_views = separate_single_views

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        obs_enc = self.encoder(obs)
        if self.ema_beta is not None:
            obs_target = self.ema_encoder(obs)  # use EMA encoder as target
            if self.projector_use_ema:
                obs_proj = self.ema_projector(obs_enc)
            else:
                obs_proj = self.projector(obs_enc)
        else:
            obs_target = obs_enc  # use SimSiam target
            obs_proj = self.projector(obs_enc)

        covariance_loss = self._covariance_reg_loss(obs_enc)
        dynamics_loss, dynamics_loss_components = self._forward_dyn_loss(
            obs_enc, obs_proj, obs_target, self.separate_single_views
        )
        total_loss = dynamics_loss + covariance_loss
        loss_components = {
            "total_loss": total_loss,
            **dynamics_loss_components,
            "covariance_loss": covariance_loss,
        }
        return obs_enc, obs_proj, total_loss, loss_components

    def _forward_dyn_loss(
        self,
        obs_enc: torch.Tensor,
        obs_proj: torch.Tensor,
        obs_target: torch.Tensor,
        separate_single_views: bool = True,
    ):
        V = obs_proj.shape[2]  # number of views
        total = torch.zeros(1, device=obs_enc.device)
        loss_components = {}
        if separate_single_views:
            for i in range(V):
                loss = self._forward_dyn_loss_one_pair(
                    obs_enc, obs_proj, obs_target, i, i
                )
                loss *= self.dynamics_loss_coef / V
                total += loss
                loss_components[f"dynamics_loss_{i}_{i}"] = loss
        else:
            total_view_pairs = V * (V - 1)  # w/ order
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    loss = self._forward_dyn_loss_one_pair(
                        obs_enc, obs_proj, obs_target, i, j
                    )
                    loss *= self.dynamics_loss_coef / total_view_pairs
                    total += loss
                    loss_components[f"dynamics_loss_{i}_{j}"] = loss
        loss_components["dynamics_loss_total"] = total
        if self.ema_beta is not None:
            loss_components["ema_beta"] = torch.Tensor([self.ema_encoder.beta]).to(
                obs_enc.device
            )
        return total, loss_components

    def _forward_dyn_loss_one_pair(
        self,
        obs_enc: torch.Tensor,
        obs_proj: torch.Tensor,
        obs_target: torch.Tensor,
        i: int,
        j: int,
    ):
        forward_dyn_input = torch.cat([obs_enc[:, :-1, j], obs_proj[:, 1:, i]], dim=-1)
        obs_enc_pred = self.forward_dynamics(forward_dyn_input)  # (N, T-1, E)
        loss = (
            1
            - torch.nn.functional.cosine_similarity(
                obs_enc_pred, obs_target[:, 1:, j].detach(), dim=-1
            ).mean()
        )
        return loss

    def _covariance_reg_loss(self, obs_enc: torch.Tensor):
        loss = off_diag_cov_loss(obs_enc)
        return loss * self.covariance_reg_coef

    def adjust_beta(self, epoch: int, max_epoch: int):
        if (self.ema_beta is None) or not self.beta_scheduling or (max_epoch == 0):
            return
        self.ema_encoder.beta = 1.0 - 0.5 * (
            1.0 + np.cos(np.pi * epoch / max_epoch)
        ) * (1.0 - self.ema_beta)
        if self.projector_use_ema:
            self.ema_projector.beta = 1.0 - 0.5 * (
                1.0 + np.cos(np.pi * epoch / max_epoch)
            ) * (1.0 - self.ema_beta)

    def step(self):
        self.forward_dynamics_optimizer.step()
        self.forward_dynamics_optimizer.zero_grad(set_to_none=True)
        if self.ema_beta is not None:
            self.ema_encoder.step(self.encoder)
            if self.projector_use_ema:
                self.ema_projector.step(self.projector)
