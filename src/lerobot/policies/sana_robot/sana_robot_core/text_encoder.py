# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/utils/text_encoder.py. Import paths are rewritten to the LeRobot package;
# the dtype keyword of transformers 5.22.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TEXT_TOKENS = 300


def load_text_encoder(model_path: str, device, dtype=torch.bfloat16):
    """Load the text encoder of the policies, the decoder of Gemma 2 2B.

    Args:
        model_path: The directory of the Gemma weights.
        device: The device to run on.
        dtype: The dtype of the weights.

    Returns:
        A function ``encode(prompt_rows)``. ``prompt_rows`` holds, per sample, the prompts of its text groups; the
        function returns the float32 embeddings [B, G, 1, 300, C_text], zero on the padding, and the token mask
        [B, G, 300]. Every distinct prompt is encoded once.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).get_decoder().to(device).eval()

    @torch.no_grad()
    def encode(prompt_rows) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompt for row in prompt_rows for prompt in row]
        distinct = list(dict.fromkeys(prompts))
        tokens = tokenizer(
            distinct, max_length=MAX_TEXT_TOKENS, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        used = int(tokens.attention_mask.sum(1).max())
        hidden = model(tokens.input_ids[:, :used], attention_mask=tokens.attention_mask[:, :used])[0]
        embeddings = hidden.new_zeros(len(distinct), MAX_TEXT_TOKENS, hidden.shape[-1])
        embeddings[:, :used] = hidden
        position = {prompt: index for index, prompt in enumerate(distinct)}
        index = torch.tensor([position[prompt] for prompt in prompts], device=device)
        shape = (len(prompt_rows), len(prompt_rows[0]))
        embeddings = embeddings[index].unflatten(0, shape).unsqueeze(2).float()
        return embeddings, tokens.attention_mask[index].unflatten(0, shape)

    return encode
