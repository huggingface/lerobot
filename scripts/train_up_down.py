#!/usr/bin/env python3
"""
Training script for LeRobot using the NLTuan/up-down dataset.
This script uses the lerobot.scripts.lerobot_train.train function with a custom configuration.
"""

import sys
from pathlib import Path

# Add the src directory to the python path so we can import lerobot
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train

def main():
    # Define the training configuration
    # We use draccus to parse CLI arguments, but we can also set them programmatically.
    # Here we create a default config and override the necessary fields.
    
    # Note: TrainPipelineConfig expects a list of arguments if we want to use the parser.
    # Alternatively, we can instantiate it directly if we know all the required fields.
    # For simplicity and to match the 'lerobot-train' command behavior, we'll use the parser-wrapped train function.
    
    # The 'lerobot-train' command typically takes arguments like:
    # --dataset.repo_id NLTuan/up-down
    # --policy.type act (or another policy type)
    # --steps 100000
    
    # We can pass these as sys.argv to the train function which is wrapped by @parser.wrap()
    
    # Default arguments for the up-down dataset
    # -------------------------------------------------------------------------
    # SAVING LOCATION:
    # By default, outputs will be saved to: outputs/train/YYYY-MM-DD/HH-MM-SS_up-down-smolvla/
    # Checkpoints will be in: outputs/train/.../checkpoints/{step}/pretrained_model
    # -------------------------------------------------------------------------
    
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--dataset.repo_id", "NLTuan/up-down",
            "--dataset.video_backend", "pyav",
            "--policy.type", "smolvla",
            "--policy.pretrained_path", "lerobot/smolvla_base", # To start from a local checkpoint, change this to the path
            "--steps", "5000",
            "--batch_size", "8",
            "--eval_freq", "1000",
            "--save_freq", "1000",
            "--log_freq", "100",
            "--job_name", "up-down-smolvla",
            "--device", "cuda",
            # To resume an interrupted training, add:
            # "--resume", "true",
            # "--checkpoint_path", "path/to/your/checkpoint/folder",
        ])
    
    # NOTE: If you want to start a NEW training session but use your fine-tuned weights 
    # as the starting point (e.g. with a new dataset), simply set:
    # --policy.pretrained_path to the path of your previous checkpoint's 'pretrained_model' folder.
    
    print(f"Starting training with arguments: {sys.argv[1:]}")
    
    try:
        # Call the train function from lerobot.scripts.lerobot_train
        # Since it's wrapped with @parser.wrap(), it will parse sys.argv
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
