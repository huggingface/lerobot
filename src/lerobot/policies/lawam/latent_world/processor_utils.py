from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from transformers import AutoProcessor


@dataclass(frozen=True)
class LatentWorldProcessorSpec:
    model_id: str
    placeholder_token: str
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True


def build_latent_world_processor_spec(*, policy_cfg: Any, vlm_model_id: str) -> LatentWorldProcessorSpec:
    return LatentWorldProcessorSpec(
        model_id=str(vlm_model_id),
        placeholder_token=str(policy_cfg.latent_action_placeholder_token),
        cache_dir=None if getattr(policy_cfg, "hf_cache_dir", None) is None else str(policy_cfg.hf_cache_dir),
    )


def configure_latent_world_processor(
    processor: Any,
    *,
    placeholder_token: str,
) -> tuple[Any, Any, int]:
    if processor.chat_template is None and getattr(getattr(processor, "tokenizer", None), "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template

    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [str(placeholder_token)]})  # type: ignore[attr-defined]
    placeholder_token_id = int(tokenizer.convert_tokens_to_ids(str(placeholder_token)))
    if placeholder_token_id < 0:
        raise ValueError(f"Invalid placeholder token id for `{placeholder_token}`.")

    return processor, tokenizer, placeholder_token_id


def load_latent_world_processor(
    spec: LatentWorldProcessorSpec,
) -> tuple[Any, Any, int]:
    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        cache_dir=spec.cache_dir,
        trust_remote_code=bool(spec.trust_remote_code),
    )
    return configure_latent_world_processor(
        processor,
        placeholder_token=spec.placeholder_token,
    )
