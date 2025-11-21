#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("widow_ai_leader")
@dataclass
class WidowAILeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm (IP address for Trossen arms)
    port: str
    
    # Trossen arm model to use
    model: str = "V0_LEADER"

    # Velocity limit scale for safety (1.0 = 100% of max velocity, 0.5 = 50% of max velocity)
    # Scales maximum joint velocities on connection. Set to None to skip scaling.
    velocity_limit_scale: float | None = None

    # Effort feedback gain for haptic feedback
    effort_feedback_gain: float = 0.1
