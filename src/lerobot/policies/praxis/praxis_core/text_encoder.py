# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/text_encoder.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Frozen text encoder: the language model of Qwen3.5-2B.

The encoder turns a task string into conditioning tokens for the video
backbone. It runs the Hugging Face ``transformers`` model and returns one
hidden state per token. The calls follow the original implementation:

* The tokenizer is the ``AutoProcessor`` of the checkpoint. The text gets no
  chat template. The tokenizer adds its special tokens (Qwen3.5 adds none).
* A batch is padded to the longest prompt, then to a multiple of 32 tokens.
* The model is ``AutoModel`` in the checkpoint dtype (bfloat16). The output
  is ``hidden_states[-3]``: the output of decoder layer 21 of 24, before the
  final norm.
* The padding positions are removed from each output.

The linear-attention layers of Qwen3.5 use the ``flash-linear-attention`` and
``causal-conv1d`` kernels when these packages are installed. The validation
runs used them. Without them ``transformers`` uses slower PyTorch code with
different rounding. Bitwise parity needs the same package versions:
``transformers`` 5.2.0, ``flash-linear-attention`` 0.5.0 and
``causal-conv1d`` 1.6.1.

The ``flash-linear-attention`` kernels pick their launch settings by timing
(Triton autotuning) in each new process, and some settings change the output
bits. :func:`pin_fla_kernels` gives each of these kernels one fixed setting:
the pick of the validation runs (:data:`FLA_KERNEL_CONFIGS`).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:  # pragma: no cover - the policy checks the package at construction
    AutoModel = None
    AutoProcessor = None

HIDDEN_LAYER = -3
PAD_TO_MULTIPLE_OF = 32
MAX_BATCH_SIZE = 2

# The launch setting (kwargs, num_warps, num_stages) of each autotuned
# flash-linear-attention kernel that the text encoder runs. These are the
# picks of the validation runs on an H100.
FLA_KERNEL_CONFIGS: dict[str, tuple[dict[str, int], int, int]] = {
    "fla.modules.l2norm.l2norm_fwd_kernel": ({"BT": 8}, 2, 3),
    "fla.modules.fused_norm_gate.layer_norm_gated_fwd_kernel": ({"BT": 16}, 16, 3),
    "fla.ops.utils.cumsum.chunk_local_cumsum_scalar_kernel": ({}, 4, 3),
    "fla.ops.gated_delta_rule.chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel": (
        {"BK": 64},
        2,
        3,
    ),
    "fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd_kernel": ({}, 4, 3),
    "fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64": (
        {"BV": 32},
        4,
        3,
    ),
    "fla.ops.common.chunk_o.chunk_fwd_kernel_o": ({"BK": 32, "BV": 32}, 2, 3),
}

logger = logging.getLogger(__name__)


def pin_fla_kernels() -> dict[str, str]:
    """Give each kernel of :data:`FLA_KERNEL_CONFIGS` its one launch setting.

    A Triton autotuner with one setting runs that setting and times nothing.
    A kernel stays autotuned when this ``flash-linear-attention`` version
    does not have it or does not offer its setting. Return the fixed settings
    by kernel name (empty without the package).
    """
    try:
        from triton.runtime import Autotuner
    except ImportError:  # no Triton: the fast path is off and there is nothing to pin
        return {}

    pinned: dict[str, str] = {}
    for path, (kwargs, num_warps, num_stages) in FLA_KERNEL_CONFIGS.items():
        module, name = path.rsplit(".", 1)
        try:
            kernel = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
        while not isinstance(kernel, Autotuner) and hasattr(kernel, "fn"):
            kernel = kernel.fn
        same = [
            c
            for c in getattr(kernel, "configs", [])
            if (c.kwargs, c.num_warps, c.num_stages) == (kwargs, num_warps, num_stages)
        ]
        if not same:
            logger.warning("%s: no launch setting %s to fix", name, kwargs)
            continue
        kernel.configs = same[:1]
        pinned[name] = str(same[0])
    return pinned


def _repair_language_model_weights(model, path: Path, loading_info: dict) -> None:
    """Load the language-model tensors that ``from_pretrained`` reported missing.

    LeRobot change. With ``transformers`` 5.4-5.5 ``AutoModel`` builds ``Qwen3_5Model``
    with the text stack under ``language_model.*``, but it renames the checkpoint's
    ``model.language_model.*`` tensors to ``model.*`` before matching, reports all 320
    of them as missing and initializes them at random (silently: the text encoder is
    then a random network). This function reads each missing ``language_model.X``
    tensor from the safetensors shards under its stored name ``model.language_model.X``
    and copies it into the model. ``transformers`` 5.2 needs no repair (no missing
    keys). Anything still missing afterwards raises ``RuntimeError``.
    """
    missing = sorted(loading_info.get("missing_keys", []))
    if not missing:
        return
    from safetensors import safe_open

    wanted = {"model." + key: key for key in missing}
    state = model.state_dict()
    loaded: set[str] = set()
    with torch.no_grad():
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                for stored in handle.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                    target = wanted.get(stored)
                    if target is None:
                        continue
                    tensor = handle.get_tensor(stored)
                    if tuple(state[target].shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"{target}: shape {tuple(state[target].shape)} != stored {tuple(tensor.shape)}"
                        )
                    state[target].copy_(tensor.to(state[target].dtype))
                    loaded.add(target)
    still_missing = sorted(set(missing) - loaded)
    if still_missing:
        raise RuntimeError(
            f"{path}: {len(still_missing)} text encoder tensors are missing from the checkpoint "
            f"(first: {still_missing[:3]}); the transformers version does not load this Qwen3.5 layout"
        )
    logger.warning(
        "transformers %s reported %d missing Qwen3.5 language-model tensors; loaded them from their stored names.",
        __import__("transformers").__version__,
        len(loaded),
    )


class TextEncoder:
    """Qwen3.5-2B as a frozen text encoder."""

    def __init__(
        self,
        path: str | Path,
        device: str | torch.device = "cuda",
        hidden_layer: int = HIDDEN_LAYER,
        pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        self.device = torch.device(device)
        self.hidden_layer = hidden_layer
        self.max_batch_size = max_batch_size
        self.kernel_configs = pin_fla_kernels()
        logger.info("text encoder kernel launch settings: %s", self.kernel_configs)
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.call_kwargs = {
            "padding": True,
            "truncation": False,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": "pt",
        }
        model, loading_info = AutoModel.from_pretrained(
            str(path), local_files_only=True, dtype="auto", output_loading_info=True
        )
        _repair_language_model_weights(model, Path(path), loading_info)
        self.model = model.to(self.device).eval().requires_grad_(False)

    def tokenize(self, prompts: Sequence[str]) -> dict[str, Tensor]:
        """Tokenize one batch. Returns ``input_ids`` and ``attention_mask``."""
        return self.processor(text=list(prompts), **self.call_kwargs).to(self.device)

    @property
    def pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad: object = tokenizer.pad_token_id
        if not isinstance(pad, int):
            raise ValueError(f"the text encoder's tokenizer has no integer pad token, got {pad!r}")
        return pad

    @torch.no_grad()
    def encode_token_ids(self, token_ids: Tensor) -> Tensor:
        """Return the ``(1, S, 2048)`` embedding of one instruction given as its token ids ``(S,)``.

        The ids are padded on the right to a multiple of ``pad_to_multiple_of`` with the pad
        token, as :meth:`tokenize` pads one prompt, so the result equals ``encode([prompt])[0]``.
        """
        ids = token_ids.reshape(-1).to(torch.int64)
        multiple = max(self.pad_to_multiple_of, 1)
        padded_len = -(-ids.shape[0] // multiple) * multiple
        input_ids = torch.full((1, padded_len), self.pad_token_id, dtype=torch.int64)
        input_ids[0, : ids.shape[0]] = ids
        attention_mask = torch.zeros((1, padded_len), dtype=torch.int64)
        attention_mask[0, : ids.shape[0]] = 1
        return self.encode_tokens(input_ids, attention_mask)[0]

    @torch.no_grad()
    def encode_tokens(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """Return one ``(1, S_i, 2048)`` bfloat16 tensor per row of a tokenized batch.

        ``input_ids`` and ``attention_mask`` are ``(B, L)`` as :meth:`tokenize` returns them
        (right padding to a multiple of ``pad_to_multiple_of``). Row ``i`` keeps its
        ``attention_mask.sum()`` first positions.
        """
        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        hidden = self.model(**batch, output_hidden_states=True).hidden_states
        states = hidden[self.hidden_layer]
        lengths = batch["attention_mask"].sum(dim=-1).cpu().tolist()
        return [states[j : j + 1, : lengths[j]] for j in range(states.shape[0])]

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """Return one ``(1, S_i, 2048)`` bfloat16 tensor per prompt.

        An empty or blank prompt becomes a single space, as in the original
        serve code. The prompts run in batches of ``max_batch_size``.
        """
        texts = [p if p.strip() else " " for p in prompts]
        out: list[Tensor] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = self.tokenize(texts[start : start + self.max_batch_size])
            out.extend(self.encode_tokens(batch["input_ids"], batch["attention_mask"]))
        return out
