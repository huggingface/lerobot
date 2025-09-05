#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0+FAST: Efficient Action Tokenization for Vision-Language-Action Models

[Paper](https://huggingface.co/papers/2501.09747)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.
Disclaimer: It is not expected to perform as well as the original implementation.

Example of finetuning the pi0+FAST pretrained model (`pi0_fast_base` in `openpi`):
```bash
lerobot-train \
--policy.path=lerobot/pi0fast_base \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of training the pi0+FAST neural network with from scratch:
```bash
lerobot-train \
--policy.type=pi0fast \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = PI0FASTPolicy.from_pretrained("lerobot/pi0fast_base")
```

"""

from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from scipy.fft import idct
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import HybridCache, StaticCache
from transformers.models.auto import CONFIG_MAPPING
from transformers import LogitsProcessorList

from lerobot.constants import ACTION, OBS_STATE, OBS_ENV_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.smolandfast.configuration_smolandfast import SMOLANDFASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


class SMOLANDFASTPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FAST tokenizer and model to train and run inference within LeRobot."""

    config_class = SMOLANDFASTConfig
    name = "smolandfast"

    def __init__(
        self,
        config: SMOLANDFASTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained("gpt2")
        self.model = SMOLANDFAST(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for PI0FAST")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model.generate_actions(batch)

            actions = actions[:, : self.config.n_action_steps]

            original_action_dim = self.config.action_feature.shape[
                0
            ]  # self.config.max_action_dim  # self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss_dict = self.model.forward(batch)
        return loss_dict["loss"], loss_dict


class SMOLANDFAST(nn.Module):
    def __init__(self, config: SMOLANDFASTConfig):
        super().__init__()
        self.config = config

        # TODO: move tokenizers in Policy
        fast_tokenizer_path = "physical-intelligence/fast"
        llm_path = "gpt2"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self.fast_skip_tokens = self.config.fast_skip_tokens
        self.max_input_seq_len = self.config.max_input_seq_len
        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[
            0
        ]  # self.config.max_action_dim  # self.config.action_feature.shape[0]
        precision = config.precision
        torch_precision = PRECISION.get(precision, torch.float32)

        # TODO: CHANGE TO GENERAL APPROACH. CURRENT SOLLUTION FOR GPT2
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        
        self.pad_token_id = (
            self.llm_tokenizer.pad_token_id
            if hasattr(self.llm_tokenizer, "pad_token_id")
            else self.llm_tokenizer.eos_token_id
        )
        self.eos_token_id = self.llm_tokenizer.eos_token_id

        # cfg = GPT2Config()
        self.llm = GPT2LMHeadModel.from_pretrained(llm_path)

        # change important stuff in bf16
        params_to_change_dtype = [
            "language_model",
            "vision_tower",
            "multi_modal",
        ]
    
        for name, param in self.llm.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch_precision)

        # TODO: Remove this once we bump transformers to >4.52.0 because the attribute will be removed
        # AttributeError: 'PaliGemmaConfig' object has no attribute 'ignore_index'
        # self.ignore_index = self.llm.config.ignore_index # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.padding_side = self.config.padding_side


    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        paligemma_tokens = self.llm_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return paligemma_tokens
    
    def _paligemma_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        fast_tokens = self.llm_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return fast_tokens
    
    def fast_tokenizer_wrapper(self, actions_norm):
        """
        A wrapper for self.fast_tokenizer that ensures batch processing,
        conversion to PyTorch tensors, and returns a dictionary without padding.
        """
        batch_tokens = self.fast_tokenizer(actions_norm)

        max_len = max([len(seq) for seq in batch_tokens])
        for seq in batch_tokens:
            seq.extend([0]*(max_len - len(seq)))
        fast_tokens = torch.tensor(batch_tokens, dtype=torch.long)

        mask = torch.where(
                fast_tokens == 0,
                0,
                1,
            ).to(torch.long)
        return fast_tokens, mask


    def create_obs_prefix_tokens(self, state, env, lang_text):
        device = state.device

        bins = torch.linspace(-1, 1, self.config.n_state_bins + 1, device=device)[:-1]

        discretized_state = torch.bucketize(state, bins) - 1
        discretized_env = torch.bucketize(env, bins) - 1

        prefix_texts = []

        for txt, disc_st, disc_env in zip(lang_text, discretized_state, discretized_env, strict=False):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(str(val.item()) for val in disc_st)
            env_str = " ".join(str(val.item()) for val in disc_env)
            prefix_texts.append(f"Task: {cleaned}, State: {state_str}, Env: {env_str}, Action: ")

        prefix_out = self.llm_tokenizer(
            prefix_texts, add_special_tokens=True, return_tensors="pt", padding="longest", truncation=False
        )
    
        prefix_ids = prefix_out["input_ids"].to(device)
        prefix_mask = prefix_out["attention_mask"].to(device)

        return prefix_ids, prefix_mask

    def create_action_tokens(self, actions):
        bsize = actions.shape[0]
        device = actions.device

        fast_tokens, act_mask = self.fast_tokenizer_wrapper(actions.cpu())

        act_ids = self._act_tokens_to_paligemma_tokens(fast_tokens)
        act_ids = act_ids.to(device)
        act_mask = act_mask.to(device)

        # Replace action with 0 to pad tokens
        mapped_fast_pad = self.llm_tokenizer.vocab_size - 1 - self.fast_skip_tokens  # fast_id=0
        act_ids = torch.where(act_ids == mapped_fast_pad, self.pad_token_id, act_ids)
        # then append eos (distinct from pad)
        eos_token = torch.full((bsize, 1), self.eos_token_id, dtype=torch.long, device=device)
        eos_mask = torch.tensor([1], dtype=torch.long, device=device).expand(bsize, -1)

        # TODO: here we first add PAD tokens, and after EOS token, I think it should be oposite
        act_ids = torch.cat([act_ids, eos_token], dim=1).to(device)
        act_mask = torch.cat([act_mask, eos_mask], dim=1).to(device)
        return act_ids, act_mask

    def create_input_tokens(self, state, env, lang_text, actions=None):
        device = state.device

        prefix_ids, prefix_mask = self.create_obs_prefix_tokens(state=state,
                                                                env=env,
                                                                lang_text=lang_text)

        if actions is not None:
            act_ids, act_mask = self.create_action_tokens(actions=actions)

            final_ids = torch.cat([prefix_ids, act_ids], dim=1).to(device)
            final_mask = torch.cat([prefix_mask, act_mask], dim=1).to(device)
        else:
            final_ids = prefix_ids.to(device)
            final_mask = prefix_mask.to(device)

        padded_output = {"input_ids": final_ids,
                         "attention_mask": final_mask}
        # define tensor of padding lengths
        prefix_lens = prefix_mask.sum(dim=1, keepdim=True)
        seq_mask = (padded_output["attention_mask"] != 0)
        loss_mask = (seq_mask.cumsum(dim=1) > prefix_lens) & seq_mask
        padded_output["loss_mask"] = loss_mask
        return padded_output

    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        padded_outs = self.create_input_tokens(
            state=batch[OBS_STATE],
            env=batch[OBS_ENV_STATE],
            lang_text=batch["task"] if "task" in batch else "",
            actions=batch[ACTION],
        )

        outputs = self.llm.forward(
            input_ids=padded_outs["input_ids"],
            attention_mask=padded_outs["attention_mask"],
            use_cache=False,
        )

        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Shift left for next-step prediction
        logits = logits[:, :-1, :]
        targets = padded_outs["input_ids"][:, 1:].to(device)  # Shift targets
        loss_mask = padded_outs["loss_mask"][:, 1:].to(device)  # Ensure correct shape

        # Compute per-token loss
        token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # Apply loss mask
        token_loss = token_loss * loss_mask.reshape(-1)

        # Compute final loss
        loss = token_loss.sum() / torch.clamp(loss_mask.sum(), min=1)

        # Return loss dictionary
        loss_dict = {"ce_loss": loss.item(), "loss": loss}
        return loss_dict

    def decode_actions_with_fast(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        relaxed_decoding: bool = True,
    ) -> np.array:
        """
        Adapt original decoding in FAST to always return actions instead of zeros.
        """
        self.time_horizon = (
            time_horizon or self.fast_tokenizer.time_horizon or self.fast_tokenizer.called_time_horizon
        )
        self.action_dim = (
            action_dim or self.fast_tokenizer.action_dim or self.fast_tokenizer.called_action_dim
        )

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert self.time_horizon is not None and self.action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                if relaxed_decoding:
                    # Expected sequence length
                    expected_seq_len = self.time_horizon * self.action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    # Apply truncation if too long
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # Truncate on the right
                    # Apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (
                    self.time_horizon,
                    self.action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
                )
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    def generate_actions(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        padded_outs = self.create_input_tokens(
            state=batch[OBS_STATE],
            env=batch[OBS_ENV_STATE],
            lang_text=batch["task"] if "task" in batch else "",
            actions=None,
        )

        input_len = padded_outs["input_ids"].shape[1]

        def make_fast_band_processor(low, high, eos_id):
            def processor(input_ids, scores):
                # Everything outside [low, high] and eos_id → -inf
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[:, low:high+1] = False
                mask[:, eos_id] = False
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
                return scores
            return processor
        # Example usage in generate_actions
        fast_vocab_size = self.fast_tokenizer.bpe_tokenizer.vocab_size
        high = self.llm_tokenizer.vocab_size - 1 - self.fast_skip_tokens
        low = high - (fast_vocab_size - 1)

        processors = LogitsProcessorList([make_fast_band_processor(low, high, self.eos_token_id)])

        output_tokens = self.llm.generate(
            input_ids=padded_outs["input_ids"],
            attention_mask=padded_outs["attention_mask"],
            use_cache=self.config.use_cache,
            max_new_tokens=self.config.max_decoding_steps,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            logits_processor=processors,
        )
        gemma_action_tokens = output_tokens[:,input_len:]

        # replace gemma pad tokens with fast pad tokens
        gemma_action_tokens = torch.where(
                    gemma_action_tokens == self.pad_token_id,
                    self.llm_tokenizer.vocab_size - 1 - self.fast_skip_tokens,
                    gemma_action_tokens,
                )

        fast_action_tokens = self._paligemma_tokens_to_act_tokens(gemma_action_tokens).tolist() 
        # remove fast pad tokens
        for seq in fast_action_tokens:
            while seq and seq[-1] == 0:
                seq.pop()

        decoded_actions = torch.tensor([
                        self.decode_actions_with_fast(
                            [tok],
                            time_horizon=self.action_horizon,
                            action_dim=self.action_dim,
                            relaxed_decoding=self.config.relaxed_action_decoding,
                        ).squeeze(0) for tok in fast_action_tokens
                ], dtype=torch.float32, device=device)
        return decoded_actions
