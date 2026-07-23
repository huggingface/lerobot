"""Configuration for the pretrained nanoVLM-460M value-function experiment."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@RewardModelConfig.register_subclass("nanovlm_value_function")
@dataclass
class NanoVLMVFConfig(RewardModelConfig):
    nanovlm_pretrained_path: str = "lusxvr/nanoVLM-460M-8k"
    nanovlm_code_path: str = "third_party/nanoVLM"
    tokenizer_path: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    image_resolution: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 256
    num_value_bins: int = 201
    value_support_min: float = -1.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 0.75
    target_method: str = "dirac_delta"
    use_one_hot_terminal: bool = True
    value_dropout: float = 0.0
    freeze_vision_encoder: bool = True
    freeze_multimodal_projector: bool = True
    freeze_language_model: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {"VISUAL": NormalizationMode.IDENTITY}
    )

    def validate_features(self) -> None:
        if not any(feature.type == FeatureType.VISUAL for feature in self.input_features.values()):
            raise ValueError("NanoVLMVFConfig requires visual input features")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=500,
            num_decay_steps=40000,
            peak_lr=1e-4,
            decay_lr=1e-6,
        )
