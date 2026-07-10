"""
Minimal HuggingFace VLM loader + freeze helpers for latent-world VLA.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor

import logging

logger = logging.getLogger(__name__)


def _optional_loader(class_name: str):
    try:
        return getattr(transformers, class_name, None)
    except Exception as exc:
        extra_hint = ""
        exc_msg = str(exc)
        if class_name == "Qwen3_5ForConditionalGeneration" and (
            "causal_conv1d" in exc_msg or "torchCheckFail" in exc_msg or "libc10.so" in exc_msg
        ):
            extra_hint = (
                " Likely cause: the Qwen3.5 Mamba/causal-conv1d native extension is incompatible with the "
                "current PyTorch/CUDA runtime on this rank."
            )
        logger.warning(
            "[vlm_auto] Skipping optional transformers loader `%s` because import failed: %s%s",
            class_name,
            exc,
            extra_hint,
        )
        return None


def _dedupe_loaders(loaders):
    unique = []
    seen = set()
    for loader in loaders:
        if loader is None:
            continue
        loader_name = getattr(loader, "__name__", str(loader))
        if loader_name in seen:
            continue
        seen.add(loader_name)
        unique.append(loader)
    return unique


def _build_loader_candidates(model_id, cache_dir=None):
    preferred_loader_names = []
    model_type = ""
    try:
        cfg = AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            trust_remote_code=True,
        )
        preferred_loader_names.extend(getattr(cfg, "architectures", []) or [])
        model_type = str(getattr(cfg, "model_type", "") or "")
        if model_type == "qwen3_5":
            preferred_loader_names.append("Qwen3_5ForConditionalGeneration")
        elif model_type == "qwen3_vl":
            preferred_loader_names.append("Qwen3VLForConditionalGeneration")
        elif model_type == "qwen2_5_vl":
            preferred_loader_names.append("Qwen2_5_VLForConditionalGeneration")
    except Exception as exc:
        logger.warning("[vlm_auto] Failed to inspect model config for `%s`: %s", model_id, exc)

    loaders = [_optional_loader(name) for name in preferred_loader_names]

    if model_type == "qwen3_5":
        loaders.extend(
            [
                _optional_loader("Qwen3_5ForConditionalGeneration"),
                AutoModelForCausalLM,
            ]
        )
    elif model_type in {"qwen3_vl", "qwen2_5_vl"}:
        loaders.extend(
            [
                _optional_loader("Qwen3VLForConditionalGeneration"),
                _optional_loader("Qwen2_5_VLForConditionalGeneration"),
                _optional_loader("AutoModelForImageTextToText"),
                AutoModelForCausalLM,
            ]
        )
    else:
        loaders.extend(
            [
                _optional_loader("Qwen3_5ForConditionalGeneration"),
                _optional_loader("Qwen3VLForConditionalGeneration"),
                _optional_loader("Qwen2_5_VLForConditionalGeneration"),
                _optional_loader("InternVLForConditionalGeneration"),
                _optional_loader("AutoModelForImageTextToText"),
                AutoModelForSeq2SeqLM,
                AutoModelForCausalLM,
            ]
        )
    return _dedupe_loaders(loaders), model_type, preferred_loader_names


def load_vlm_auto(model_id, cache_dir=None, dtype: torch.dtype = torch.bfloat16):
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=True,
    )
    if processor.chat_template is None and getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    ):
        processor.chat_template = processor.tokenizer.chat_template

    loaders, model_type, preferred_loader_names = _build_loader_candidates(model_id, cache_dir=cache_dir)
    if model_type == "qwen3_5" and not any(
        getattr(loader, "__name__", "") == "Qwen3_5ForConditionalGeneration" for loader in loaders
    ):
        raise RuntimeError(
            "Failed to import `Qwen3_5ForConditionalGeneration` for "
            f"`{model_id}`. This model depends on the Qwen3.5 Mamba/causal-conv1d native extension, "
            "and the current rank cannot load that extension. Please reinstall/rebuild the environment so "
            "`causal_conv1d_cuda` matches the current PyTorch/CUDA runtime."
        ) from None
    base_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    # Qwen3.5 has shown unstable FlashAttention-2 behavior during LIBERO eval
    # (illegal memory access inside HF flash attention integration). Prefer
    # SDPA there, keep eager as a safer fallback, and only try FA2 last.
    if model_type == "qwen3_5":
        loader_variants = [
            {"attn_implementation": "sdpa"},
            {"attn_implementation": "eager"},
            {"attn_implementation": "flash_attention_2"},
            {},
        ]
    else:
        loader_variants = [
            {"attn_implementation": "flash_attention_2"},
            {"attn_implementation": "sdpa"},
            {},
        ]

    errors = []
    for loader in loaders:
        loader_name = getattr(loader, "__name__", str(loader))
        for variant in loader_variants:
            try:
                vlm = loader.from_pretrained(
                    model_id,
                    **base_kwargs,
                    **variant,
                )
                if variant:
                    logger.info("[vlm_auto] Loaded `%s` with `%s` using %s", model_id, loader_name, variant)
                else:
                    logger.info("[vlm_auto] Loaded `%s` with `%s`", model_id, loader_name)
                return vlm, processor
            except Exception as exc:
                variant_label = variant if variant else {"attn_implementation": None}
                errors.append(f"{loader_name} {variant_label}: {type(exc).__name__}: {exc}")

    error_msg = "\n".join(f"  - {msg}" for msg in errors[-12:])
    qwen35_hint = ""
    if model_type == "qwen3_5":
        qwen35_hint = (
            "\nHint: `qwen3_5` requires the Qwen3.5 Mamba/causal-conv1d native extension. "
            "Your current runtime failed to load it, which is usually caused by a mismatched "
            "`causal-conv1d` build vs. the active PyTorch/CUDA environment."
        )
    raise RuntimeError(
        f"Failed to load VLM `{model_id}` via generic loaders.{qwen35_hint}\nRecent loader errors:\n{error_msg}"
    ) from None


def remove_lm_head(vlm) -> bool:
    """Drop the causal LM output head when the VLA only consumes backbone hidden states."""
    if not hasattr(vlm, "lm_head") or getattr(vlm, "lm_head") is None:
        return False
    delattr(vlm, "lm_head")
    logger.info("[vlm_auto.remove_lm_head] Removed unused VLM `lm_head`.")
    return True


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _resolve_llm_module(vlm):
    candidate_llm_paths = [
        "language_model",
        "model.language_model",
        "text_model",
        "model.text_model",
        "transformer",
        "model.decoder",
        "decoder",
        "model",
    ]
    for path in candidate_llm_paths:
        llm = _get_nested_attr(vlm, path)
        if llm is not None:
            return llm
    return None


def _resolve_llm_layers_container(llm_module):
    candidate_layer_paths = [
        "layers",
        "h",
        "language_model.layers",
        "language_model.model.layers",
        "language_model.decoder.layers",
        "language_model.decoder.layer",
        "language_model.transformer.h",
        "model.layers",
        "model.decoder.layers",
        "model.encoder.layers",
        "decoder.layers",
        "decoder.layer",
        "encoder.layers",
        "encoder.layer",
        "transformer.h",
        "transformer.layers",
        "transformer.blocks",
        "transformer.block",
        "blocks",
        "block",
    ]

    for path in candidate_layer_paths:
        candidate = _get_nested_attr(llm_module, path)
        if isinstance(candidate, (list, nn.ModuleList)):
            return candidate, path

    return None, None


def _keep_first_n_llm_layers(llm_module, n: int) -> bool:
    if n is None or n <= 0:
        return False

    layers_container, layers_path = _resolve_llm_layers_container(llm_module)
    if layers_container is None:
        logger.warning(
            "[vlm_auto._keep_first_n_llm_layers] Failed to locate LLM layers container for pruning."
        )
        return False

    num_layers = len(layers_container)
    keep_n = int(n)
    if keep_n >= num_layers:
        logger.info(
            "[vlm_auto._keep_first_n_llm_layers] Requested keep_first=%d layers but model only has %d via `%s`; leaving unchanged.",
            keep_n,
            num_layers,
            layers_path,
        )
        return True

    while len(layers_container) > keep_n:
        if isinstance(layers_container, nn.ModuleList):
            del layers_container[-1]
        else:
            layers_container.pop(-1)

    logger.info(
        "[vlm_auto._keep_first_n_llm_layers] Pruned LLM layers via `%s`: kept first %d / %d layers.",
        layers_path,
        keep_n,
        num_layers,
    )
    return True


def _unfreeze_last_n_llm_layers(llm_module, n: int) -> bool:
    if n is None or n <= 0:
        return False

    layers_container, layers_path = _resolve_llm_layers_container(llm_module)
    if layers_container is None:
        logger.warning("[vlm_auto._unfreeze_last_n_llm_layers] Failed to locate LLM layers container.")
        return False

    num_layers = len(layers_container)
    n_layers = min(int(n), num_layers)
    if n_layers <= 0:
        return False

    for layer in list(layers_container)[-n_layers:]:
        try:
            layer.requires_grad_(True)
        except Exception:
            logger.debug("Failed to unfreeze an LLM layer.", exc_info=True)
            continue

    logger.info(
        "[vlm_auto._unfreeze_last_n_llm_layers] Unfroze last %d / %d LLM layers via `%s`.",
        n_layers,
        num_layers,
        layers_path,
    )
    return True


def freeze_qwen3vl(
    vlm,
    freeze_vision_backbone,
    freeze_llm_backbone,
    freeze_last_llm_layer,
    freeze_embedding: bool = False,
    unfreeze_vision_merger: bool = False,
):
    visual = _get_nested_attr(vlm, "model.visual") or _get_nested_attr(vlm, "visual")
    if freeze_vision_backbone and visual is not None:
        try:
            visual.requires_grad_(False)
        except Exception:
            logger.debug("Failed to freeze Qwen vision backbone.", exc_info=True)

        if unfreeze_vision_merger:
            unfroze_any = False
            try:
                if hasattr(visual, "merger"):
                    visual.merger.requires_grad_(True)
                    unfroze_any = True
            except Exception:
                logger.debug("Failed to unfreeze Qwen vision merger.", exc_info=True)
            try:
                if hasattr(visual, "deepstack_merger_list"):
                    visual.deepstack_merger_list.requires_grad_(True)
                    unfroze_any = True
            except Exception:
                logger.debug("Failed to unfreeze Qwen deepstack merger.", exc_info=True)
            if unfroze_any:
                logger.info(
                    "[freeze_qwen3vl] Kept Qwen vision merger trainable (unfreeze_vision_merger=True)"
                )

    language_model = _get_nested_attr(vlm, "model.language_model") or _get_nested_attr(vlm, "language_model")
    if freeze_llm_backbone:
        if language_model is not None:
            try:
                language_model.requires_grad_(False)
            except Exception:
                logger.debug("Failed to freeze Qwen language model.", exc_info=True)
        else:
            try:
                vlm.requires_grad_(False)
            except Exception:
                logger.debug("Failed to freeze Qwen VLM module.", exc_info=True)

        if not freeze_embedding:
            try:
                emb = None
                if hasattr(vlm, "get_input_embeddings"):
                    emb = vlm.get_input_embeddings()
                if emb is None and language_model is not None and hasattr(language_model, "embed_tokens"):
                    emb = language_model.embed_tokens
                if emb is not None:
                    emb.requires_grad_(True)
            except Exception:
                logger.debug("Failed to keep Qwen input embeddings trainable.", exc_info=True)

        if not freeze_last_llm_layer:
            try:
                if hasattr(vlm, "lm_head") and vlm.lm_head is not None:
                    vlm.lm_head.requires_grad_(True)
            except Exception:
                logger.debug("Failed to keep Qwen LM head trainable.", exc_info=True)

    if freeze_embedding:
        try:
            emb = None
            if hasattr(vlm, "get_input_embeddings"):
                emb = vlm.get_input_embeddings()
            if emb is None and language_model is not None and hasattr(language_model, "embed_tokens"):
                emb = language_model.embed_tokens
            if emb is not None:
                emb.requires_grad_(False)
        except Exception:
            logger.debug("Failed to freeze Qwen input embeddings.", exc_info=True)

    if freeze_last_llm_layer:
        try:
            if hasattr(vlm, "lm_head") and vlm.lm_head is not None:
                vlm.lm_head.requires_grad_(False)
        except Exception:
            logger.debug("Failed to freeze Qwen LM head.", exc_info=True)
