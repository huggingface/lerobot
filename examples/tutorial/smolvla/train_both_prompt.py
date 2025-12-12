import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import math
import wandb
import os
import sys

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from LTA_Pool.datasets import BatchData, build_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C, H, W = 3, 512, 512
TOKENIZER_MAX_LEN = 48
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 2

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

class ConfigClass:
    batch_size: int = 256
    max_steps: int = 5000
    lr_start: float = 1e-4
    lr_end: float = 2.5e-6
    warmup_steps: int = 100
    weight_decay: float = 1e-6
    eps: float = 1e-8
    betas = (0.9, 0.95)
    model_id: str = "lerobot/smolvla_base"
    project_name: str = "smolvla-training"
    run_name: str = "both_captions"
    save_checkpoint_path: str = "/scratch2/autodp/smolvla_stuff/"

def build_normalizer(x_min=-0.0416, x_max=64.613, y_min=-44.581, y_max=36.627, out_min=-1.0, out_max=1.0):
    """
    Returns two functions:
        normalize(x, y)
        unnormalize(x_norm, y_norm)
        
    where x, y can be scalars, tensors, or arrays.
    """

    def normalize(x, y):
        # Scale x
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = x_norm * (out_max - out_min) + out_min

        # Scale y
        y_norm = (y - y_min) / (y_max - y_min)
        y_norm = y_norm * (out_max - out_min) + out_min

        return x_norm, y_norm

    def unnormalize(x_norm, y_norm):
        # Undo scaling for x
        x = (x_norm - out_min) / (out_max - out_min)
        x = x * (x_max - x_min) + x_min

        # Undo scaling for y
        y = (y_norm - out_min) / (out_max - out_min)
        y = y * (y_max - y_min) + y_min

        return x, y

    return normalize, unnormalize


def save_checkpoint(model_network, processor, cfg, global_step):
    save_directory = f"{cfg.save_checkpoint_path}checkpoints/{cfg.run_name}/step_{global_step}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    policy_model = model_network.smolVLA_model
    policy_model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)

def get_tokens(prompt):
    return processor(
        text=prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TOKENIZER_MAX_LEN
    ).to(DEVICE)

def log_debug_info(model, global_step):
    debug_dict = {}
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            debug_dict[f"weights/{name}"] = wandb.Histogram(param.data.cpu().float())
            if param.grad is not None:
                grad_data = param.grad.data.cpu().float()
                debug_dict[f"gradients/{name}"] = wandb.Histogram(grad_data)
                param_norm = grad_data.norm(2).item()
                total_grad_norm += param_norm ** 2
    total_grad_norm = total_grad_norm ** 0.5
    debug_dict["train/global_grad_norm"] = total_grad_norm
    wandb.log(debug_dict, step=global_step)

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup, total, lr_start, lr_end):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.step_num = 0

    def step(self):
        if self.step_num < self.warmup:
            lr = self.lr_start * self.step_num / self.warmup
        else:
            decay_step = self.step_num - self.warmup
            decay_total = self.total - self.warmup
            if decay_total <= 0:
                cosine = 1.0
            else:
                cosine = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
            lr = self.lr_end + (self.lr_start - self.lr_end) * cosine

        for g in self.opt.param_groups:
            g["lr"] = lr
        self.step_num += 1
        return lr

class ModelNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = SmolVLAConfig.from_pretrained(cfg.model_id)
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(C, H, W)),
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
        }
        self.smolVLA_model = SmolVLAPolicy.from_pretrained(cfg.model_id, config=config)

        old_out = self.smolVLA_model.model.action_out_proj
        new_out = nn.Linear(old_out.in_features, MAX_ACTION_DIM, bias=old_out.bias is not None)
        with torch.no_grad():
            new_out.weight.copy_(old_out.weight[:MAX_ACTION_DIM, :])
            if old_out.bias is not None:
                new_out.bias.copy_(old_out.bias[:MAX_ACTION_DIM])
        self.smolVLA_model.model.action_out_proj = new_out

        old_in = self.smolVLA_model.model.action_in_proj
        new_in = nn.Linear(MAX_ACTION_DIM, old_in.out_features, bias=old_in.bias is not None)
        with torch.no_grad():
            new_in.weight.copy_(old_in.weight[:, :MAX_ACTION_DIM])
            if old_in.bias is not None:
                new_in.bias.copy_(old_in.bias)
        self.smolVLA_model.model.action_in_proj = new_in

        self.smolVLA_model.config.max_action_dim = MAX_ACTION_DIM
        self.smolVLA_model.config.max_state_dim = MAX_STATE_DIM

    def forward(self, batch):
        return self.smolVLA_model(batch=batch)

def train_step(model_loss, batch, captions):
    model_loss.train()
    inputs = get_tokens(captions)

    img1 = batch.history.cam_front[:, 0:1, :, :, :].squeeze(1).permute(0, 3, 1, 2)
    gt_traj = batch.gt_traj[:, :-1, :] # (B,50,2)
    # Split X and Y
    x = gt_traj[:, :, 0]   # (B, 50)
    y = gt_traj[:, :, 1]   # (B, 50)

    normalize, unnormalize = build_normalizer()
    x_norm, y_norm = normalize(x, y)  # (B, 50), (B, 50)

    gt_traj = torch.stack((x_norm, y_norm), dim=2)  # (B, 50, 2)
    

    sample = {
        "observation.images.camera1": img1,
        "observation.state": batch.ego_curr_state,
        "action": gt_traj,
        "observation.language.tokens": inputs["input_ids"],
        "observation.language.attention_mask": inputs["attention_mask"].bool(),
    }
    batch_input = {k: v.to(DEVICE) for k, v in sample.items()}
    loss, _, _ = model_loss(batch_input)

    return loss

def main():
    cfg = ConfigClass()
    config_dict = {k: v for k, v in ConfigClass.__dict__.items() if not k.startswith('__')}
    wandb.init(project=cfg.project_name, name=cfg.run_name, config=config_dict)

    dataset_configs = [{"name": "nuscenes", "kwargs": {"load_images": True, "num_past_images": 6, "image_res": (512, 512)}}]
    dataloader = build_dataloader(dataset_configs=dataset_configs, batch_size=cfg.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    model_loss = ModelNetwork(cfg).to(DEVICE)

    for name, param in model_loss.named_parameters():
        param.requires_grad = name.startswith("smolVLA_model.model.action")

    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model_loss.parameters()),
        lr=cfg.lr_start,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps
    )

    scheduler = WarmupCosineScheduler(
        optimiser,
        warmup=cfg.warmup_steps,
        total=cfg.max_steps,
        lr_start=cfg.lr_start,
        lr_end=cfg.lr_end
    )

    global_step = 0

    while global_step <= cfg.max_steps:
        for batch in dataloader:
            #breakpoint()
            prompts = [batch.captions1, batch.captions2]
            for i in prompts:
            
                loss = train_step(model_loss, batch, i)
                loss.backward()

                if global_step % 200 == 0:
                    log_debug_info(model_loss, global_step)

                optimiser.step()
                curr_lr = scheduler.step()
                optimiser.zero_grad()
                
                if global_step % 200 == 0:
                    save_checkpoint(model_loss, processor, cfg, global_step)
                wandb.log({"train/loss": loss.item(), "train/lr": curr_lr, "global_step": global_step})
                
                global_step += 1
                if global_step >= cfg.max_steps:
                    break
    
    wandb.finish()
    

if __name__ == "__main__":
    main()
