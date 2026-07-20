from dataclasses import dataclass
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

@PreTrainedConfig.register_subclass("pi05_polarization")
@dataclass
class PI05PolarizationConfig(PI05Config):
    polfem_checkpoint_path: str = ""
    polfem_canonical_resolution: int = 256
    polar_embed_dim: int = 2048

    def validate_features(self) -> None:
        super().validate_features()
        required = {"observation.polar000", "observation.polar045",
                    "observation.polar090", "observation.polar135"}
        missing = required - set(self.input_features)
        if missing:
            raise ValueError(f"PI05Polarization requires these input_features: {missing}")