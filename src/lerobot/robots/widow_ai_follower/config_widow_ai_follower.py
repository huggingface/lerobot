#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("widow_ai_follower")
@dataclass
class WidowAIFollowerConfig(RobotConfig):
    # Port to connect to the arm (IP address for Trossen arms)
    port: str
    
    # Trossen arm model to use
    model: str = "V0_FOLLOWER"  # Options: "V0_LEADER", "V0_FOLLOWER"

    # Velocity limit scale for safety (1.0 = 100% of max velocity, 0.5 = 50% of max velocity)
    # Scales maximum joint velocities on connection. Set to None to skip scaling.
    velocity_limit_scale: float | None = None

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: float | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Enable effort sensing to include effort measurements in observations
    effort_sensing: bool = False
