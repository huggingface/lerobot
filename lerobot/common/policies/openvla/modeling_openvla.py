# lerobot/common/policies/openvla/modeling_openvla.py

from typing import Dict
import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.utils.utils import get_safe_torch_device
from PIL import Image

class OpenVLAPolicy(nn.Module, PyTorchModelHubMixin):
    """OpenVLA Policy wrapper for LeRobot framework"""
    
    name = "openvla"

    def __init__(self, config, dataset_stats=None):
        super().__init__()
        self.config = config

        # Define action ranges for SO-100 robot
        # These values should be adjusted based on your robot's specifications
        self.action_ranges = {
            'position': {  
                'min': np.array([-0.02, -0.02, -0.02]),  # 2cm max movement per step
                'max': np.array([0.02, 0.02, 0.02])      
            },
            'orientation': {  
                'min': np.array([-0.1, -0.1, -0.1]),     # ~5.7 degrees max rotation per step
                'max': np.array([0.1, 0.1, 0.1])      
            },
            'gripper': {
                'min': 0.0,  # Closed
                'max': 1.0   # Open
            }
        }
        # Get MPS device if available
        self.device = get_safe_torch_device("mps")

        # Initialize OpenVLA model and processor 
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # Load model with torch.float32 for MPS compatibility
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # Use float32 instead of bfloat16 for MPS
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.to(self.device)


    def reset(self):
        """Called when environment is reset"""
        pass

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Training forward pass - not implemented for OpenVLA as we use pretrained model"""
        raise NotImplementedError("OpenVLA policy is inference-only")

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Unnormalize OpenVLA actions to SO-100 robot ranges
        normalized_action: array of shape (7,) in range [-1, 1]
        returns: array of shape (6,) in robot's actual ranges
        """

        # Add safety checks
        if np.any(np.abs(normalized_action) > 1.0):
            print("WARNING: Input actions outside [-1,1] range:", normalized_action)
            normalized_action = np.clip(normalized_action, -1.0, 1.0)

        # Split into components
        pos_norm = normalized_action[:3]
        rot_norm = normalized_action[3:6]
        grip_norm = normalized_action[6]

        # Unnormalize position (from [-1,1] to actual position deltas)
        pos_range = self.action_ranges['position']
        pos_unnorm = (
            pos_norm + 1  # From [-1,1] to [0,2]
        ) / 2 * (pos_range['max'] - pos_range['min']) + pos_range['min']

        # Unnormalize orientation (from [-1,1] to actual angle deltas)
        rot_range = self.action_ranges['orientation']
        rot_unnorm = (
            rot_norm + 1
        ) / 2 * (rot_range['max'] - rot_range['min']) + rot_range['min']

        # Unnormalize gripper (from [-1,1] to [0,1])
        grip_range = self.action_ranges['gripper']
        grip_unnorm = (grip_norm + 1) / 2 * (grip_range['max'] - grip_range['min']) + grip_range['min']

        # Combine into final action
        # For SO-100, we'll concatenate position and orientation 
        # Note: gripper is handled separately in your robot control
        action = np.concatenate([pos_unnorm, rot_unnorm])
        
        return action

    @torch.no_grad
    def select_action(self, observation: Dict[str, Tensor]) -> Tensor:
        """Run inference to select next action"""
        # Print observation dictionary
        print("\nObservation Dictionary:")
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value)}")

        # Find the camera image key
        image_key = next(
            (key for key in observation.keys() if "image" in key),
            None
        )
        if image_key is None:
            raise KeyError(f"No image key found in observation. Available keys: {observation.keys()}")

        # Format prompt with task instruction
        prompt = f"In: What action should the robot take to {self.config.instruction}?\nOut:"
        print(f"prompt: {prompt}, image key: {image_key}")
        # Get image and process it
        image = observation[image_key]
        # Remove batch dim and convert to PIL
        image = image.squeeze(0)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype('uint8'))

        # Get model inputs and ensure correct types
        inputs = self.processor(prompt, image)
        
        # Convert input_ids to long (int64) type and move to device
        inputs["input_ids"] = inputs["input_ids"].long().to(self.device)
        
        # Convert attention mask to appropriate type and device 
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        # Convert pixel_values to float32 and move to device
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float32, device=self.device)

        # Get normalized actions from OpenVLA
        # Predict action
        actions = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        print("Raw normalized actions:", actions)  # Should be in [-1, 1]

        # Unnormalize for SO-100
        unnorm_actions = self.unnormalize_action(actions)
        print("Unnormalized actions for SO-100:", unnorm_actions)

        # Convert numpy array to torch tensor
        actions = torch.from_numpy(unnorm_actions)

        print("\nActions as tensor:", actions)
        print("Actions shape:", actions.shape)
        print("Actions dtype:", actions.dtype)

        return actions

    def to(self, device):
        """Override to() to handle device movement"""
        super().to(device)
        self.device = device
        self.model.to(device)
        return self
