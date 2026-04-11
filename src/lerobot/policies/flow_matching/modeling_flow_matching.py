import math
from collections import deque
from typing import Optional, Unpack, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from lerobot.policies.pretrained import PreTrainedPolicy, ActionSelectKwargs
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig

# --- Begin Network Definitions ---
class SinusoidalPositionEmbedding(nn.Module):
    """
    Standard Positional Encoding for 1D sequences or continuous time (t).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] or [B, T] tensor of continuous values (like time or sequence index)
        Returns: [B, ..., dim] positional embeddings.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ActionFlowNet(nn.Module):
    """
    Standard Vector Field Estimator (Transformer-based).
    Maps (x_t, t, condition) -> _pred, representing the derivative of the trajectory.
    """
    def __init__(
        self, 
        action_dim: int, 
        cond_dim: int, 
        hidden_dim: int = 256, 
        num_layers: int = 6, 
        nheads: int = 8, 
        max_horizon: int = 500
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon

        # Time embedding (Continuous integration time t)
        self.time_emb = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Sequence position embedding (Spatial index within the trajectory chunk)
        self.pos_emb = nn.Embedding(max_horizon, hidden_dim)
        
        # State & Condition Projection
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dim_feedforward=hidden_dim * 4, 
            activation="gelu", 
            batch_first=True,
            norm_first=True # Standard practice for deep Transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Projection (predicts the derivative u of the action sequence)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, T, action_dim] The noisy trajectory state at time t.
        t: [B] The integration time.
        conditions: [B, cond_dim] The encoded conditional features (images + state).
        
        Returns:
            v_pred: [B, T, action_dim] The predicted velocity vector field.
        """
        B, T, D = x_t.shape
        device = x_t.device
        
        # 1. Process Time
        t_feat = self.time_mlp(self.time_emb(t)) # [B, hidden_dim]
        
        # 2. Process Sequence
        seq_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_feat = self.pos_emb(seq_idx) # [B, T, hidden_dim]
        
        # 3. Project Input Trajectory
        x_feat = self.action_proj(x_t) # [B, T, hidden_dim]
        
        # 4. Process Conditions (treat condition as the first sequence element)
        c_feat = self.cond_proj(conditions).unsqueeze(1) # [B, 1, hidden_dim]
        
        # Combine Condition [B, 1, H] and Actions [B, T, H] -> [B, 1+T, H]
        token_feat = torch.cat([c_feat, x_feat], dim=1)
        
        # Broadcast Time feature to sequence length -> [B, 1+T, H]
        time_token = t_feat.unsqueeze(1).expand(-1, 1+T, -1)
        
        # Pad Position feature to match the condition token -> [B, 1+T, H]
        # Condition token gets a zeroed positional embedding
        cond_pos = torch.zeros((B, 1, self.hidden_dim), device=device)
        total_pos = torch.cat([cond_pos, pos_feat], dim=1)
        
        # Final Transformer Input
        transformer_in = token_feat + time_token + total_pos
        
        # Forward pass
        latent = self.transformer(transformer_in) # [B, 1+T, H]
        
        # 5. Extract action latent (drop the condition token at index 0)
        action_latent = latent[:, 1:, :] # [B, T, H]
        
        # 6. Predict velocity
        v_pred = self.out_proj(action_latent) # [B, T, action_dim]
        
        return v_pred

# --- Begin Flow Matching Logic ---
class FlowMatchingEulerSolver:
    """
    A purely mathematical Euler Solver for Flow Matching.
    """
    def __init__(self, num_sampling_steps: int = 10, ts_start: float = 0.0, ts_end: float = 1.0):
        self.num_sampling_steps = num_sampling_steps
        self.ts_start = ts_start
        self.ts_end = ts_end
        self.dt = (self.ts_end - self.ts_start) / self.num_sampling_steps

    @torch.no_grad()
    def solve(
        self, 
        vector_field_fn, 
        x_0: torch.Tensor, 
        conditions: dict = None,
        omega: float = 1.0
    ) -> torch.Tensor:
        device = x_0.device
        x_t = x_0.clone()
        B = x_t.shape[0]

        for step in range(self.num_sampling_steps):
            t_val = self.ts_start + step * self.dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=x_t.dtype)
            
            if omega == 1.0:
                v_pred = vector_field_fn(x_t, t_tensor, conditions)
            else:
                v_cond = vector_field_fn(x_t, t_tensor, conditions)
                uncond_conditions = {k: None for k in conditions}
                v_uncond = vector_field_fn(x_t, t_tensor, uncond_conditions)
                v_pred = v_uncond + omega * (v_cond - v_uncond)

            x_t = x_t + v_pred * self.dt

        return x_t

def get_optimal_transport_target(x_1: torch.Tensor, x_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the target vector field (u) and interpolated state (x_t).
    """
    B = x_1.shape[0]
    device = x_1.device
    
    t = torch.rand((B,), device=device, dtype=x_1.dtype)
    t_broadcast = t.view(B, *([1]*(x_1.ndim - 1)))
    
    u = x_1 - x_0
    x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * x_1
    
    return x_t, u, t

# --- Begin Flow Matching Policy ---
class FlowMatchingPolicy(PreTrainedPolicy):
    """
    Vision-Language-Proprioception Wrapper for Flow Matching.
    This policy module manages image feature extraction (ResNet), 
    proprioceptive state embedding, and Classifier-Free Guidance dropping.
    """
    name = "flow_matching"
    config_class = FlowMatchingConfig
    
    def __init__(self, config: FlowMatchingConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        
        self.action_dim = config.action_dim
        self.qpos_dim = config.qpos_dim
        self.num_cameras = config.num_cameras
        self.hidden_dim = config.hidden_dim
        
        resnet = models.resnet18(weights=config.pretrained_backbone_weights)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj = nn.Linear(512, self.hidden_dim)

        self.qpos_kin_proj = nn.Linear(7, self.hidden_dim)
        self.qpos_force_proj = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.cond_dim = (self.num_cameras * self.hidden_dim) + self.hidden_dim + self.hidden_dim

        self.net = ActionFlowNet(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=config.depth,
            nheads=config.num_heads,
            max_horizon=config.max_horizon
        )
        
        self._action_queue = deque()
        self.reset()
        
    def reset(self):
        self._action_queue.clear()
        
    def get_optim_params(self) -> dict:
        return [
            {"params": [p for n, p in self.named_parameters() if "vision_backbone" not in n]},
            {
                "params": [p for n, p in self.named_parameters() if "vision_backbone" in n],
                "lr": 1e-5,
            }
        ]

    def _extract_inputs_from_batch(self, batch: dict[str, torch.Tensor]):
        qpos = batch.get(self.config.state_key)
        image_list = []
        for key in self.config.image_keys:
            if key in batch:
                image_list.append(batch[key])
                
        if len(image_list) > 0:
            images = torch.stack(image_list, dim=1)
        else:
            images = None
            
        return qpos, images

    def extract_condition(self, qpos: torch.Tensor, images: torch.Tensor, uncond_prob: float = 0.0) -> torch.Tensor:
        B = qpos.shape[0] if qpos is not None else images.shape[0]
        K = self.num_cameras
        device = qpos.device if qpos is not None else images.device
        
        if qpos is not None:
            qpos_kin = qpos[:, :7]
            qpos_force = qpos[:, 7:]
            qpos_kin_feat = self.qpos_kin_proj(qpos_kin)
            qpos_force_feat = self.qpos_force_proj(qpos_force)
        else:
            qpos_kin_feat = torch.zeros((B, self.hidden_dim), device=device)
            qpos_force_feat = torch.zeros((B, self.hidden_dim), device=device)
            
        if images is not None:
            flat_imgs = images.view(-1, 3, images.shape[-2], images.shape[-1])
            img_feats = self.vision_backbone(flat_imgs).squeeze(-1).squeeze(-1)
            img_feats = self.image_proj(img_feats).view(B, K * self.hidden_dim)
        else:
            img_feats = torch.zeros((B, K * self.hidden_dim), device=device)
            
        if uncond_prob > 0.0:
            drop_mask = (torch.rand(B, device=device) > uncond_prob).float().unsqueeze(1)
            img_feats = img_feats * drop_mask
            qpos_force_feat = qpos_force_feat * drop_mask
            
        condition_vector = torch.cat([qpos_kin_feat, qpos_force_feat, img_feats], dim=-1)
        return condition_vector

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        qpos, images = self._extract_inputs_from_batch(batch)
        actions = batch[self.config.action_key]
        is_pad = batch.get("action_is_pad", None)
        
        condition_vector = self.extract_condition(qpos, images, uncond_prob=self.config.uncond_prob)
        
        x_0 = torch.randn_like(actions)
        x_t, u_true, t_tensor = get_optimal_transport_target(x_1=actions, x_0=x_0)
        
        u_pred = self.net(x_t=x_t, t=t_tensor, conditions=condition_vector)
        
        loss = nn.functional.mse_loss(u_pred, u_true, reduction='none')
        
        if is_pad is not None:
            valid_mask = ~is_pad
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()
            
        return loss, {"l1": loss.detach(), "mse": loss.detach()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> torch.Tensor:
        qpos, images = self._extract_inputs_from_batch(batch)
        
        B = qpos.shape[0] if qpos is not None else images.shape[0]
        device = qpos.device if qpos is not None else images.device
        
        conditions = {"qpos": qpos, "images": images}
        
        def vector_field_wrapper(x_t, t_tensor, conds):
            is_uncond = conds.get("images") is None
            prob = 1.0 if is_uncond else 0.0
            cond_vec = self.extract_condition(qpos, images, uncond_prob=prob)
            return self.net(x_t, t_tensor, cond_vec)
            
        solver = FlowMatchingEulerSolver(num_sampling_steps=self.config.num_sampling_steps)
        x_0 = torch.randn((B, self.config.max_horizon, self.action_dim), device=device)
        action_seq = solver.solve(vector_field_fn=vector_field_wrapper, x_0=x_0, conditions=conditions, omega=self.config.omega)
        
        return action_seq

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            actions_chunk = self.predict_action_chunk(batch, **kwargs)
            for t in range(actions_chunk.shape[1]):
                self._action_queue.append(actions_chunk[:, t, :])
        return self._action_queue.popleft()
