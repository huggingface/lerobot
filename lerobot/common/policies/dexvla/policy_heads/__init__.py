from transformers import AutoConfig, AutoModel

from .configuration_scaledp import ScaleDPPolicyConfig
from .configuration_unet_diffusion import UnetDiffusionPolicyConfig
from .modeling_scaledp import ScaleDP
from .modeling_unet_diffusion import ConditionalUnet1D


def register_policy_heads():
    AutoConfig.register("scale_dp_policy", ScaleDPPolicyConfig)
    AutoConfig.register("unet_diffusion_policy", UnetDiffusionPolicyConfig)
    AutoModel.register(ScaleDPPolicyConfig, ScaleDP)
    AutoModel.register(UnetDiffusionPolicyConfig, ConditionalUnet1D)
