#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-time debug helpers for PI052's language head."""

import logging
from typing import Any


def print_debug_text_predictions(policy: Any, batch: dict[str, Any], step: int, n_samples: int = 5) -> None:
    """Print supervised text predictions and token accuracy for up to ``n_samples`` rows."""
    # Unwrap distributed wrappers that do not proxy custom policy methods.
    inner = policy
    while hasattr(inner, "module") and not hasattr(inner, "debug_text_predictions"):
        inner = inner.module
    if not hasattr(inner, "debug_text_predictions"):
        logging.warning(
            "LEROBOT_DEBUG_PREDS_EVERY set but policy %s has no "
            "debug_text_predictions method — skipping dump.",
            type(inner).__name__,
        )
        return
    try:
        debug = inner.debug_text_predictions(batch, max_samples=n_samples)
    except Exception as exc:  # noqa: BLE001
        logging.warning("debug_text_predictions failed: %s", exc, exc_info=True)
        return
    if not debug:
        logging.warning(
            "debug_text_predictions returned no supervised samples — current batch has no text labels."
        )
        return
    policy = inner  # used below for select_message-style decoding parity

    # Build a tokenizer for decoding — match training side exactly.
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: PLC0415
            register_paligemma_loc_tokens,
        )

        tok_name = getattr(policy.config, "tokenizer_name", None) or "google/paligemma-3b-pt-224"
        tokenizer = register_paligemma_loc_tokens(AutoTokenizer.from_pretrained(tok_name))
    except Exception as exc:  # noqa: BLE001
        logging.warning("debug preds: tokenizer load failed: %s", exc)
        return

    ids = debug["input_ids"]
    labels = debug["labels"]
    preds = debug["predictions"]
    attn = debug["attention_mask"]

    n = ids.shape[0]
    print(
        f"\n========== STEP {step} DEBUG PREDICTIONS ({n} samples) ==========",
        flush=True,
    )
    for s in range(n):
        a = attn[s].tolist()
        real = sum(a)
        sid = ids[s].tolist()
        sl = labels[s].tolist()
        sp = preds[s].tolist()
        prompt = tokenizer.decode(sid[:real], skip_special_tokens=False)
        print(f"\n  --- sample {s + 1}/{n} ---", flush=True)
        print(f"  prompt: {prompt!r}", flush=True)

        # Ground-truth target (the contiguous supervised label span).
        sup_ids = [int(sid[i]) for i in range(real) if sl[i] != -100]
        if sup_ids:
            print(
                f"  target  (ground truth)        : {tokenizer.decode(sup_ids, skip_special_tokens=False)!r}",
                flush=True,
            )

        # Training-side teacher-forced argmax on the same prompt+target.
        n_sup = n_ok = 0
        teacher_chars: list[int] = []
        for i in range(1, real):
            label = sl[i]
            if label == -100:
                continue
            n_sup += 1
            pred = int(sp[i - 1])
            teacher_chars.append(pred)
            if label == pred:
                n_ok += 1
        teacher_text = tokenizer.decode(teacher_chars, skip_special_tokens=False) if teacher_chars else ""
        acc = n_ok / max(n_sup, 1)
        print(
            f"  training argmax (teacher-fed) : {teacher_text!r}   acc={n_ok}/{n_sup}={acc:.1%}",
            flush=True,
        )
    print("=" * 60 + "\n", flush=True)
