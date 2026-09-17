"""Legacy nominal-refinement diffusion planner for SafeDiff-VLA, kept only to reproduce past
experiments -- not the main path. See `..modeling_safediff_vla` for the current `temporal_decoder`
architecture. Registered as policy type `"safediff_vla_legacy"`
(`lerobot-train --policy.type=safediff_vla_legacy ...`).
"""

from .configuration_legacy_diffusion import LegacySafeDiffVLAConfig

__all__ = ["LegacySafeDiffVLAConfig"]
