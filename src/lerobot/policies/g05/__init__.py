"""OpenGalaxea G0.5 policy integration."""

from .configuration_g05 import G05Config
from .processor_g05 import make_g05_pre_post_processors

__all__ = ["G05Config", "make_g05_pre_post_processors"]
