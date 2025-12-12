import torch
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from LTA_Pool.datasets import BatchData, build_dataloader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C, H, W = 3, 512, 512
TOKENIZER_MAX_LEN = 48
MAX_STATE_DIM = 32

from training_loop_smolVLA import ConfigClass, ModelNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Update this path to the specific step folder you want to load
CHECKPOINT_PATH = "/scratch2/autodp/smolvla_stuff/checkpoints/latest7_with_only_norm/step_1800" 

TOKENIZER_MAX_LEN = 48
IMAGE_SIZE = (512, 512)
STATE_DIM = 6  # (x, y, z, roll, pitch, yaw) or (v, a, w...) depending on your data

def get_tokens(prompt, processor):
    return processor(
        text=prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TOKENIZER_MAX_LEN
    ).to(DEVICE)

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

def load_model(checkpoint_dir):
    """
    Loads the model and processor. 
    Because you used save_pretrained, this automatically respects the 
    MAX_ACTION_DIM=2 and MAX_STATE_DIM=32 configuration.
    """
    print(f"Loading model from: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")

    # Load Processor (for text)
    processor = AutoProcessor.from_pretrained(checkpoint_dir)

    # Load Policy Model
    # The config.json here contains the architecture changes you made (action_dim=2)
    model = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    model = model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully.")
    return model, processor

def main():
    cfg = ConfigClass
    
    dataset_configs = [{"name": "nuscenes", "kwargs": {"load_images": True, "num_past_images": 6, "image_res": (512, 512)}}]
    

    dataloader = build_dataloader(dataset_configs=dataset_configs, batch_size=10, shuffle=True)
    model, processor = load_model(CHECKPOINT_PATH)
    batch = next(iter(dataloader))
    inputs = get_tokens(batch.captions1, processor)
    img1 = batch.history.cam_front[:, 0:1, :, :, :].squeeze(1).permute(0, 3, 1, 2)
    gt_traj = batch.gt_traj[:, 1:, :]
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
    
    inference_input = {k: v for k, v in batch_input.items() if k != "action"}
    
    v_t = model.predict_action_chunk(inference_input)
    
    # Convert tensors to numpy (batch-first)
    gt_traj_np = gt_traj.detach().cpu().numpy()      # (B, 50, 2)
    v_t_np = v_t.detach().cpu().numpy()              # (B, 50, 2)

    B = gt_traj_np.shape[0]

    for i in range(B):
        # Ground truth (i-th example)
        gt_x = gt_traj_np[i, :, 0]
        gt_y = gt_traj_np[i, :, 1]

        # Prediction (i-th example)
        pred_x = v_t_np[i, :, 0]
        pred_y = v_t_np[i, :, 1]

        plt.figure()

        # Ground truth trajectory (bold green)
        plt.plot(gt_x, gt_y, linewidth=3, color='green', label='ground truth')

        # Predicted trajectory (dark blue)
        plt.plot(pred_x, pred_y, linewidth=2, color='darkblue', label='predicted')

        # Starting point
        plt.scatter(pred_x[0], pred_y[0], s=80, color='black', marker='o', label='start')

        # Ending point
        plt.scatter(pred_x[-1], pred_y[-1], s=80, color='red', marker='x', label='end')

        # Caption string → safe filename
        caption_str = batch.captions1[i].replace(" ", "_")[:50]
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Trajectory Example {i}: Ground Truth vs Prediction. Captions: {{{caption_str}}}")

        plt.legend()

        

        plt.savefig(f"v_t_plot_example_{i}.png", bbox_inches='tight')
        plt.close()
    
    
    # state_dict = load_file(f"{CHECKPOINT_PATH}/model.safetensors")
    # for i in state_dict.keys():
    #     if i.startswith("model.action"):
    #         print(i)
    #breakpoint()
    #model_loss = ModelNetwork(cfg).to(DEVICE)
    
    
    
if __name__ == "__name__":
    main()
    
    
    
    
    
    
    


if __name__=="__main__":
    main()