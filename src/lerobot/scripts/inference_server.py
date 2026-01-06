import torch
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path
import traceback
import sys

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

app = FastAPI()

# Global state
policy = None
preprocessor = None
postprocessor = None
device = torch.device("cuda")

class InferenceRequest(BaseModel):
    wrist_image: List[int]
    proprio: List[float]
    shape_wrist: List[int]
    reset: Optional[bool] = False

@app.on_event("startup")
def load_model():
    global policy, preprocessor, postprocessor
    ckpt_path = Path("outputs/train/v122_final/checkpoints/100000/pretrained_model")
    
    print(f"Loading policy from {ckpt_path}...")
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()

    # Load official processors (automatically handles stats.json and cropping)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt_path
    )
    policy.reset()
    print("Policy and Processors loaded. System Ready.")

@app.post("/infer")
async def infer(req: InferenceRequest):
    try:
        if req.reset:
            policy.reset()

        raw_pixels = np.array(req.wrist_image, dtype=np.int32)
        bgr_img = raw_pixels.reshape(req.shape_wrist)
        # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 1) raw obs to observation
        img_t = torch.from_numpy(bgr_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        state_t = torch.tensor(req.proprio, dtype=torch.float32).unsqueeze(0).to(device)

        raw_obs = {
            "observation.state": state_t,
            "observation.images.gripper": img_t
        }

        # 2) Preprocessor
        obs_processed = preprocessor(raw_obs)

        # sys.last_obs_processed = {k: v.detach().cpu() for k, v in obs_processed.items()}

        # 3) Inference
        with torch.inference_mode():
            # Use select_action for a single step, or predict_action_chunk for the 16-step horizon
            action_norm = policy.select_action(obs_processed)

            # 4) Post-processing (Un-normalization)
            action_phys = postprocessor(action_norm)
            
            actions_np = action_phys.cpu().numpy()

        return {
            "actions": actions_np.flatten().tolist(),
            "shape": list(actions_np.shape)
        }

    except Exception as e:
        full_stack = traceback.format_exc()
        
        print("---------- INFERENCE STACK TRACE ----------")
        print(full_stack)
        print("-------------------------------------------")
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)