from dataclasses import dataclass

@dataclass 
class OpenVLAConfig:
    """Configuration for OpenVLA policy"""
    model_name: str = "openvla/openvla-7b"
    instruction: str = "pick up the blue object and drop in the bin" # Task instruction for prompting
