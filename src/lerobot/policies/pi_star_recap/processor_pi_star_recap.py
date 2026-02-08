#!/usr/bin/env python3
"""
Data Processor for π*₀.₆ RECAP Policy

Handles RECAP-specific data formats including:
- Data type labeling (demo, auto, intervention)
- Advantage computation preparation
- Multi-modal observation processing
"""

import torch
from typing import Any
import numpy as np

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import unravel_index
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_pi_star_recap import PiStarRECAPConfig


class PiStarRECAPProcessor:
    """
    Data processor for π*₀.₆ RECAP
    
    Extends LeRobot's standard processing with RECAP-specific features:
    - Data type tracking
    - Reward and done signal handling
    - Advantage-related preprocessing
    """
    
    def __init__(self, config: PiStarRECAPConfig):
        self.config = config
    
    def prepare_recap_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Prepare batch for RECAP training
        
        Adds RECAP-specific fields:
        - data_types: str indicating source (demo/auto/intervention)
        - rewards: scalar reward signal
        - dones: episode termination flag
        """
        # Standard LeRobot batch already contains observations and actions
        processed = {}
        
        # Copy standard fields
        for key in [OBS_IMAGES, OBS_STATE, ACTION]:
            if key in batch:
                processed[key] = batch[key]
        
        # Add RECAP-specific fields with defaults if not present
        batch_size = len(batch[ACTION])
        device = batch[ACTION].device
        
        # Data types (for RECAP weighted sampling)
        if "data_types" in batch:
            processed["data_types"] = batch["data_types"]
        else:
            # Default to demo if not specified
            processed["data_types"] = ["demo"] * batch_size
        
        # Rewards (for IQL)
        if "rewards" in batch:
            processed["rewards"] = batch["rewards"]
        else:
            # Use dummy rewards if not in dataset
            processed["rewards"] = torch.zeros(batch_size, 1, device=device)
        
        # Done flags (for IQL)
        if "dones" in batch:
            processed["dones"] = batch["dones"]
        else:
            # Assume not done if not specified
            processed["dones"] = torch.zeros(batch_size, 1, device=device)
        
        # Intervention mask (optional, for tracking intervention points)
        if "intervention_masks" in batch:
            processed["intervention_masks"] = batch["intervention_masks"]
        
        return processed
    
    def compute_advantages_for_logging(
        self,
        batch: dict[str, torch.Tensor],
        policy: PreTrainedPolicy,
    ) -> dict[str, float]:
        """
        Compute advantages for logging/analysis
        
        This is called during evaluation to track advantage statistics
        """
        with torch.no_grad():
            # Get Q-values
            context = policy._encode_observations(batch)
            actions = batch[ACTION]
            
            q_values = torch.stack([q(context, actions) for q in policy.q_networks], dim=0)
            q_value = q_values.min(dim=0)[0]
            
            # Get V-values
            v_value = policy.v_network(context)
            
            # Compute advantage
            advantage = q_value - v_value
        
        return {
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item(),
            "advantage_min": advantage.min().item(),
            "advantage_max": advantage.max().item(),
            "q_value_mean": q_value.mean().item(),
            "v_value_mean": v_value.mean().item(),
        }


def create_recap_dataset_summary(dataset) -> dict:
    """
    Create summary of RECAP dataset composition
    
    Analyzes the distribution of data types in the dataset
    """
    summary = {
        "total_episodes": len(dataset),
        "demo_count": 0,
        "auto_count": 0,
        "intervention_count": 0,
    }
    
    # Count data types if available
    for episode in dataset:
        data_type = episode.get("data_type", "demo")
        if data_type == "demo":
            summary["demo_count"] += 1
        elif data_type == "auto":
            summary["auto_count"] += 1
        elif data_type == "intervention":
            summary["intervention_count"] += 1
    
    return summary
