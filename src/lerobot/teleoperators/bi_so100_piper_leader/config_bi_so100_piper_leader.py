# src/lerobot/teleoperators/bi_so100_piper_leader/config_bi_so100_piper_leader.py
#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_so100_piper_leader")
@dataclass
class BiSO100PiperLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str