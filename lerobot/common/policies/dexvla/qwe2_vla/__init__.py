from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_qwen2_vla import Qwen2VLAConfig
from .modeling_qwen2_vla import Qwen2VLForConditionalGenerationForVLA


def register_qwen2_vla():
    AutoConfig.register("qwen2_vla", Qwen2VLAConfig)
    AutoModelForCausalLM.register(Qwen2VLAConfig, Qwen2VLForConditionalGenerationForVLA)
