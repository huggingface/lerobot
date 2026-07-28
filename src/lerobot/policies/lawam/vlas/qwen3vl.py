"""Qwen3-VL loading and freezing helpers for LaWAM."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


def load_qwen3vl(
    model_id: str,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, AutoProcessor]:
    cache_path = str(cache_dir) if cache_dir is not None else None
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_path,
        trust_remote_code=True,
    )
    if config.model_type != "qwen3_vl":
        raise ValueError(
            "The LeRobot LaWAM adapter supports Qwen3-VL backbones only; "
            f"got model_type={config.model_type!r} for {model_id!r}."
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_path,
        trust_remote_code=True,
    )
    if processor.chat_template is None and getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    ):
        processor.chat_template = processor.tokenizer.chat_template

    base_kwargs = {
        "cache_dir": cache_path,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    errors = []
    for attention_backend in ("flash_attention_2", "sdpa", None):
        variant = {"attn_implementation": attention_backend} if attention_backend is not None else {}
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                **base_kwargs,
                **variant,
            )
            logger.info(
                "Loaded Qwen3-VL %s with attention backend %s.",
                model_id,
                attention_backend or "default",
            )
            return model, processor
        except Exception as exc:
            errors.append(f"{attention_backend or 'default'}: {type(exc).__name__}: {exc}")

    details = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(f"Failed to load Qwen3-VL {model_id!r}:\n{details}") from None


def remove_lm_head(vlm: nn.Module) -> None:
    """Drop the unused causal language-model output head."""
    if hasattr(vlm, "lm_head"):
        del vlm.lm_head


def _qwen3vl_components(vlm: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    model = getattr(vlm, "model", None)
    if model is None:
        return None, None
    return getattr(model, "visual", None), getattr(model, "language_model", None)


def keep_first_n_llm_layers(vlm: nn.Module, num_layers: int) -> None:
    _, language_model = _qwen3vl_components(vlm)
    layers = getattr(language_model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise AttributeError(
            "Qwen3-VL language model does not expose a ModuleList at `model.language_model.layers`."
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if num_layers < len(layers):
        del layers[num_layers:]


def unfreeze_last_n_llm_layers(vlm: nn.Module, num_layers: int) -> None:
    _, language_model = _qwen3vl_components(vlm)
    layers = getattr(language_model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise AttributeError(
            "Qwen3-VL language model does not expose a ModuleList at `model.language_model.layers`."
        )
    if num_layers <= 0:
        return
    for layer in layers[-num_layers:]:
        layer.requires_grad_(True)


def freeze_qwen3vl(
    vlm: nn.Module,
    *,
    freeze_vision_backbone: bool,
    freeze_llm_backbone: bool,
    freeze_embedding: bool = False,
    unfreeze_vision_merger: bool = False,
) -> None:
    visual, language_model = _qwen3vl_components(vlm)
    if visual is None or language_model is None:
        raise AttributeError("Expected a Qwen3-VL model with `model.visual` and `model.language_model`.")

    if freeze_vision_backbone:
        visual.requires_grad_(False)
        if unfreeze_vision_merger:
            merger = getattr(visual, "merger", None)
            if merger is not None:
                merger.requires_grad_(True)
            deepstack_mergers = getattr(visual, "deepstack_merger_list", None)
            if deepstack_mergers is not None:
                deepstack_mergers.requires_grad_(True)

    if freeze_llm_backbone:
        language_model.requires_grad_(False)

    embeddings = vlm.get_input_embeddings()
    if embeddings is not None:
        embeddings.requires_grad_(not freeze_embedding)
