# src/lerobot/robots/bi_piper_follower/config_bi_piper_follower.py
#!/usr/bin/env python

from dataclasses import dataclass
from ..config import RobotConfig


@RobotConfig.register_subclass("bi_piper_follower")
@dataclass
class BiPiperFollowerConfig(RobotConfig):
    left_port: str = "can0"
    right_port: str = "can1"
    cameras: dict = None  # opcional; {} si no usas cámaras