#!/usr/bin/env python

"""
Configuration file for Brewie dataset recording.

This file contains all settings for dataset recording.
Modify values in this file before running record.py

IMPORTANT: hf_token is now obtained from HUGGINGFACE_TOKEN environment variable
or from command line arguments. Do not store tokens in code!

USAGE EXAMPLES:

1. With environment variable (recommended):
   export HUGGINGFACE_TOKEN=your_token_here
   python examples/brewie/record_with_config.py

2. With command line argument:
   python examples/brewie/record_with_config.py --hf-token your_token_here

3. Interactive input (token will be requested at startup):
   python examples/brewie/record_with_config.py

4. Alternative environment variable:
   export HF_TOKEN=your_token_here
   python examples/brewie/record_with_config.py

SECURITY:
- Never commit tokens to code
- Use environment variables for production
- Tokens are entered hidden (not displayed in terminal)
"""

import os
import sys
import getpass
from dataclasses import dataclass
from typing import Optional

def get_hf_token() -> str:
    """
    Get HuggingFace token from environment variable or prompt user.
    
    Token retrieval order:
    1. HUGGINGFACE_TOKEN environment variable
    2. HF_TOKEN environment variable
    3. Command line argument --hf-token
    4. Interactive input (hidden)
    
    Returns:
        str: HuggingFace token
        
    Raises:
        ValueError: If token not found and user cancelled input
    """
    def validate_token(token: str) -> str:
        """Validate HuggingFace token format."""
        if not token or not token.strip():
            raise ValueError("Token cannot be empty")
        
        token = token.strip()
        
        # Basic validation of HuggingFace token format
        if not token.startswith("hf_"):
            print("Warning: Token does not start with 'hf_'. Make sure this is a valid HuggingFace token.")
        
        if len(token) < 10:
            raise ValueError("Token is too short. Check token correctness.")
        
        return token
    
    # 1. Check environment variables
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return validate_token(token)
    
    # 2. Check command line arguments
    if "--hf-token" in sys.argv:
        try:
            token_index = sys.argv.index("--hf-token")
            if token_index + 1 < len(sys.argv):
                token = sys.argv[token_index + 1]
                return validate_token(token)
        except (ValueError, IndexError):
            pass
    
    # 3. Prompt user
    print("HuggingFace token not found in environment variables.")
    print("Set HUGGINGFACE_TOKEN environment variable or enter token:")
    print("  export HUGGINGFACE_TOKEN=your_token_here")
    print()
    
    try:
        token = getpass.getpass("Enter your HuggingFace token (hidden input): ")
        return validate_token(token)
    except KeyboardInterrupt:
        print("\nInput cancelled by user")
        raise ValueError("Token not provided")

@dataclass
class RecordingConfig:
    """Configuration for Brewie dataset recording."""
    
    # =============================================================================
    # HUGGINGFACE SETTINGS
    # =============================================================================
    
    # Your HuggingFace credentials
    hf_username: str = "your_username"  # Replace with your username
    # hf_token is now obtained dynamically from environment variables or input
    
    # Dataset name (will be created as username/dataset_name)
    dataset_name: str = "hit_detection"
    
    # =============================================================================
    # RECORDING SETTINGS
    # =============================================================================
    
    # Number of episodes to record
    num_episodes: int = 5
    
    # Recording frequency (frames per second)
    # WARNING: After adding new sensors, it's recommended to reduce FPS
    # to prevent video synchronization issues
    fps: int = 20  # Reduced from 30 to 20 for stability
    
    # Duration of each episode in seconds
    episode_time_sec: int = 30
    
    # Reset time between episodes in seconds
    reset_time_sec: int = 5
    
    # =============================================================================
    # TASK DESCRIPTION
    # =============================================================================
    
    # Description of the task the robot will perform
    task_description: str = "Brewie robot manipulation demonstration"
    
    # Additional metadata
    task_category: str = "manipulation"  # manipulation, pick_place, assembly, etc.
    difficulty_level: str = "beginner"   # beginner, intermediate, advanced
    
    # =============================================================================
    # ROBOT SETTINGS
    # =============================================================================
    
    # ROS connection parameters
    ros_master_ip: str = "localhost"
    ros_master_port: int = 9090
    
    # Safety settings
    max_relative_target: float = 50.0  # Maximum relative movement per step
    servo_duration: float = 0.1        # Duration for servo movements
    
    # =============================================================================
    # DATASET SETTINGS
    # =============================================================================
    
    # Use video in dataset
    use_videos: bool = True
    
    # Number of threads for image recording
    image_writer_threads: int = 4
    
    # =============================================================================
    # ADDITIONAL SETTINGS
    # =============================================================================
    
    # Display data during recording
    display_data: bool = True 
    
    # Session name for visualization
    session_name: str = "brewie_record"
    
    # Automatically push to Hub after recording
    auto_push_to_hub: bool = True
    
    # Continue recording in existing dataset (add new episodes)
    resume_existing_dataset: bool = False
    
    def get_hf_token(self) -> str:
        """
        Get HuggingFace token for this configuration.
        
        Returns:
            str: HuggingFace token 
        """
        return get_hf_token()
    
    # =============================================================================
    # PREDEFINED CONFIGURATIONS
    # =============================================================================
    
    @classmethod
    def quick_demo(cls) -> "RecordingConfig":
        """Quick demo - 2 short episodes."""
        return cls(
            hf_username="forroot",  # REQUIRED: replace with your username
            # hf_token is obtained dynamically from environment variables
            ros_master_ip="192.168.20.21",
            ros_master_port=9090,
            num_episodes=2,
            fps=20, 
            episode_time_sec=15,
            reset_time_sec=3,
            task_description="Fast demo of robot movements",
            task_category="demo",
            difficulty_level="beginner",
            resume_existing_dataset=True
        )
    
    @classmethod
    def detection_aim(cls) -> "RecordingConfig":
        """racking and aiming at an enemy robot for fire. FAST MODE"""
        return cls(
            hf_username="forroot",  # REQUIRED: replace with your username
            dataset_name ="detection_aim",
            ros_master_ip="192.168.20.21",
            ros_master_port=9090,
            num_episodes=2,
            fps=20, 
            episode_time_sec=15,
            reset_time_sec=3,
            task_description="Tracking and aiming at an enemy robot for fire",
            task_category="aim",
            difficulty_level="beginner",
            resume_existing_dataset=True
        )
    @classmethod
    def hit_detection(cls) -> "RecordingConfig":
        return cls(
            hf_username="forroot",  # REQUIRED: replace with your username
            dataset_name ="TERST2",
            ros_master_ip="192.168.20.23",
            ros_master_port=9090,
            num_episodes=2,
            fps=20, 
            episode_time_sec=2,
            reset_time_sec=3,
            task_description="Testing observation of hit data FIRE button True = human hit verification",
            task_category="hit",
            difficulty_level="beginner",
            resume_existing_dataset=False
        )

    @classmethod
    def resume_demo(cls) -> "RecordingConfig":
        """Demonstration of continuing recording in existing dataset."""
        return cls(
            hf_username="forroot",  # REQUIRED: replace with your username
            # hf_token is obtained dynamically from environment variables
            ros_master_ip="192.168.20.21",
            ros_master_port=9090,
            num_episodes=3,
            episode_time_sec=20,
            reset_time_sec=5,
            task_description="Additional episodes for existing dataset",
            task_category="demo",
            difficulty_level="beginner",
            resume_existing_dataset=True  # Enable resume recording mode
        )
    
    @classmethod
    def full_dataset(cls) -> "RecordingConfig":
        """Full dataset - many episodes for training."""
        return cls(
            num_episodes=20,
            episode_time_sec=60,
            reset_time_sec=10,
            task_description="Full set of demonstrations for training",
            task_category="manipulation",
            difficulty_level="intermediate"
        )
    
    @classmethod
    def pick_place_task(cls) -> "RecordingConfig":
        """Configuration for pick and place task."""
        return cls(
            num_episodes=10,
            episode_time_sec=45,
            reset_time_sec=8,
            task_description="Grasping and placing objects",
            task_category="pick_place",
            difficulty_level="intermediate"
        )
    
    @classmethod
    def assembly_task(cls) -> "RecordingConfig":
        """Configuration for assembly task."""
        return cls(
            num_episodes=15,
            episode_time_sec=90,
            reset_time_sec=15,
            task_description="Assembly of parts by robot",
            task_category="assembly",
            difficulty_level="advanced"
        )
    
    @classmethod
    def optimized_with_sensors(cls) -> "RecordingConfig":
        """Optimized configuration for working with new sensors."""
        return cls(
            hf_username="forroot",  # REQUIRED: replace with your username
            ros_master_ip="192.168.20.21",
            ros_master_port=9090,
            num_episodes=5,
            episode_time_sec=30,
            reset_time_sec=5,
            fps=15,  # Reduced frequency for stability
            task_description="Optimized recording with new sensors",
            task_category="demo",
            difficulty_level="beginner",
            resume_existing_dataset=True,
            image_writer_threads=2,  # Fewer threads for stability
            use_videos=True
        )

# =============================================================================
# CONFIGURATION SELECTION
# =============================================================================

# Choose one of the predefined configurations or create your own
#config = RecordingConfig.optimized_with_sensors()  # Recommended for new sensors
config = RecordingConfig.hit_detection()
# config = RecordingConfig.resume_demo()  # For continuing recording in existing dataset
# config = RecordingConfig.full_dataset()
# config = RecordingConfig.pick_place_task()
# config = RecordingConfig.assembly_task()

# Or create your own configuration
'''
config = RecordingConfig(
    hf_username="your_username",  # REQUIRED: replace with your username
    # hf_token is obtained automatically from environment variables
    dataset_name="brewie_my_task",
    num_episodes=5,
    episode_time_sec=30,
    task_description="My task for Brewie robot",
    resume_existing_dataset=False  # True to continue recording in existing dataset
)
'''